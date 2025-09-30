import os
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from mne import io
import logging
import pandas as pd
import gc
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import traceback
from torch.utils.data import Dataset
from tqdm import tqdm

@dataclass
class TransformerConfig:
    sequence_length: int = 256
    sampling_rate: int = 256
    overlap: float = 0.25
    n_channels: int = None
    normalization: str = "standard"
    filter_low: float = 0.5
    filter_high: float = 50.0

    # Memory optimization settings
    max_sequences_per_file: int = 1000  # Limit sequences per file to save memory
    batch_process_size: int = 10  # Process files in smaller batches


class EEGSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, metadata: Optional[pd.DataFrame] = None):
        """
        Converts preprocessed EEG sequences into a format that PyTorch DataLoader can use for batch training.

        Args:
            sequences: Shape (n_samples, sequence_length, n_channels)
            labels: Shape (n_samples,)
            metadata: Optional metadata for each sequence
        """

        # converts numpy arrays to PyTorch tensors
        self.sequences = torch.FloatTensor(sequences) # for sequences (neural network input)
        self.labels = torch.LongTensor(labels) # for labels (classification targets)
        self.metadata = metadata # optional metadata for sequence tracking

    def __len__(self):
        return len(self.sequences) # returns the total number of sequences

    def __getitem__(self, index):
        return {
            'sequence' : self.sequences[index], # EEG data tensor for one sequence
            'label' : self.labels[index], # seizure classification (0 or 1)
            'index' : index # position in the dataset for tracking
        }


class TransformerProcessor:
    """
    Preprocessing pipeline for EEG data to be used with the Transformer architecture.
    Converts raw EDF files into sequences suitable for CNN-Transformer training.
    """

    def __init__(self, config: TransformerConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.channel_names = []
        self.logger = self.setup_logging()


    def load_raw_edf(self, file_path: str) -> Tuple[np.ndarray, List[str], int]:
        """
        Load raw EEG data from EDF file

        :return:
            Tuple[np.ndarray, List[str], int]
            np.ndarray - The EEG signal data (typically shape: channels × time_samples)
            List[str] - Channel names/labels (e.g., ['Fp1-F7', 'F7-T7', 'T7-P7', ...])
            int - Sampling rate (256hz)
        """
        # Segment → Create time windows
        # Sequence → Group windows into sequences for transformer

        self.logger.debug(f"Loading EDF file: {file_path}")

        # Load EDF → Extract raw EEG signals
        raw = io.read_raw_edf(file_path, preload=True, verbose=False)

        # Filters → Apply frequency filters (0.5-50 Hz)
        if self.config.filter_low or self.config.filter_high:
            raw.filter(l_freq=self.config.filter_low, h_freq=self.config.filter_high, verbose=False)

        # Resample if needed
        if raw.info['sfreq'] != self.config.sampling_rate:
            raw.resample(self.config.sampling_rate, verbose=False)

        # Get data and channel names
        data = raw.get_data() # Shape: (n_channels, n_samples)
        channel_names = raw.ch_names
        sampling_rate = int(raw.info['sfreq'])

        return data, channel_names, sampling_rate


    def create_sequences_from_raw(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create overlapping sequences from continuous EEG data for transformer training using a sliding window approach.

        With overlapping (sliding window) we will have more sequences and the sliding window ensures we capture
        Pre-seizure patterns, Seizure onset (critical transition moments) and Post-seizure patterns

        Args:
           data: Shape (n_channels, n_samples)
           labels: Optional labels for each sample

        Returns:
            sequences: Shape (n_sequences, sequence_length, n_channels)
            sequence_labels: Shape (n_sequences,)
        """

        n_channels, n_samples = data.shape

        # Calculate sliding window parameters
        seq_len = self.config.sequence_length # 256 windows
        step_size = int(seq_len * (1 - self.config.overlap))

        # Calculate number of sequences
        n_sequences = max(1, (n_samples - seq_len) // step_size + 1)

        sequences = np.zeros((n_sequences, seq_len, n_channels))
        sequence_labels = np.zeros(n_sequences) if labels is not None else None

        for i in range(n_sequences):
            start_idx = i * step_size
            end_idx = min(start_idx + seq_len, n_samples)

            # Handle edge case where we don't have enough samples
            if end_idx - start_idx < seq_len:
                # Pad with the last available samples
                seq_data = data[:, max(0, n_samples - seq_len):n_samples]
            else:
                seq_data = data[:, start_idx:end_idx]

            # Transpose to get (sequence_length, n_channels)
            sequences[i] = seq_data.T

            # Label: majority vote or presence of seizure in sequence
            if labels is not None:
                if end_idx - start_idx < seq_len:
                    seq_labels = labels[max(0, n_samples - seq_len):n_samples]
                else:
                    seq_labels = labels[start_idx:end_idx]
                sequence_labels[i] = 1 if np.any(seq_labels) else 0

        return sequences, sequence_labels


    def normalize_sequences(self, sequences: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        This method applies feature scaling to EEG sequences to ensure consistent input ranges for the transformer model.

        EEG channels have different scales:

        Some channels: ±10 μV amplitude
        Other channels: ±100 μV amplitude
        Without normalization, transformer attention focuses on high-amplitude channels

        StandardScaler ensures:
        All channels have equal importance in attention mechanisms
        Faster convergence during training
        Better gradient flow through the network
        """

        # Reshape
        original_shape = sequences.shape  # 3D sequence -  (n_sequences, sequence_length, n_channels)
        sequences_2d = sequences.reshape(-1, original_shape[-1])  # reshape into 2D format - (n_sequences * sequence_length, n_channels)

        if fit: # if fit=True, will compute the mean/std from data and transform
            sequences_normalized = self.scaler.fit_transform(sequences_2d)
        else:
            sequences_normalized = self.scaler.transform(sequences_2d)

        return sequences_normalized.reshape(original_shape) # restore the original shape


    def process_single_recording(self,
                                 file_path: str,
                                 seizure_intervals: Optional[List[Tuple[float, float]]] = None) -> Dict:
        """Process a single EDF recording into transformer-ready sequences with proper labels """

        try:
            # Load raw data
            data, channel_names, fs = self.load_raw_edf(file_path)

            # Store channel info
            if not self.channel_names:
                self.channel_names = channel_names
                self.config.n_channels = len(channel_names)

            # Create sample-wise labels if seizure intervals provided
            labels = None
            if seizure_intervals:
                n_samples = data.shape[1]
                labels = np.zeros(n_samples)

                for start_sec, end_sec in seizure_intervals:
                    start_sample = int(start_sec * fs)
                    end_sample = int(end_sec * fs)
                    end_sample = min(end_sample, n_samples)
                    if start_sample < n_samples:
                        labels[start_sample:end_sample] = 1

            # Create sequences
            max_seq = getattr(self.config, 'max_sequences_per_file', None)
            sequences, seq_labels = self.create_sequences_from_raw(data, labels)

            # Free memory immediately
            del data
            if labels is not None:
                del labels

            return {
                'sequences': sequences,
                'labels': seq_labels if seq_labels is not None else np.zeros(len(sequences)),
                'channel_names': channel_names,
                'file_path': file_path,
                'success': True,
                'n_sequences': len(sequences),
                'n_seizure_sequences': np.sum(seq_labels) if seq_labels is not None else 0
            }

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return {
                'sequences': None,
                'labels': None,
                'channel_names': None,
                'file_path': file_path,
                'success': False,
                'error': str(e)
            }

    def process_dataset(self,
                        data_dir: str,
                        seizure_info: Dict[str, List[List[float]]],
                        output_dir: str) -> Dict:

        """
        This method is the main orchestrator that processes an entire directory of EDF files into a unified, transformer-ready dataset.
        """

        os.makedirs(output_dir, exist_ok=True)

        batch_size = getattr(self.config, 'batch_process_size', 8)

        # Find EDF files
        edf_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.edf'):
                    file_path = os.path.join(root, file)
                    recording_id = os.path.splitext(os.path.relpath(file_path, data_dir).replace(os.sep, '_'))[0]
                    edf_files.append((file_path, recording_id))

        self.logger.info(f"Found {len(edf_files)} EDF files, processing in batches of {batch_size}")

        # Process and save each batch immediately
        batch_files_created = []
        total_sequences = 0
        seizure_sequences = 0

        for batch_idx in range(0, len(edf_files), batch_size):
            batch_files = edf_files[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1

            self.logger.info(f"Processing batch {batch_num}/{(len(edf_files) - 1) // batch_size + 1}")

            batch_sequences = []
            batch_labels = []
            batch_metadata = []

            # Process current batch
            for file_path, recording_id in tqdm(batch_files, desc=f"Batch {batch_num}"):
                seizure_intervals = seizure_info.get(recording_id, None)
                result = self.process_single_recording(file_path, seizure_intervals)

                if result['success']:
                    batch_sequences.append(result['sequences'])
                    batch_labels.append(result['labels'])

                    for j in range(len(result['sequences'])):
                        batch_metadata.append({
                            'recording_id': recording_id,
                            'sequence_idx': j,
                            'file_path': file_path,
                            'has_seizure': bool(result['labels'][j]),
                            'batch_num': batch_num
                        })

                    total_sequences += result['n_sequences']
                    seizure_sequences += result['n_seizure_sequences']

            # Save batch immediately if we have data
            if batch_sequences:
                batch_combined_seq = np.concatenate(batch_sequences, axis=0)
                batch_combined_labels = np.concatenate(batch_labels, axis=0)

                # Normalize this batch
                if batch_num == 1:  # Fit scaler on first batch
                    batch_combined_seq = self.normalize_sequences(batch_combined_seq, fit=True)
                else:  # Transform subsequent batches
                    batch_combined_seq = self.normalize_sequences(batch_combined_seq, fit=False)

                # Save batch to disk
                batch_file = os.path.join(output_dir, f'batch_{batch_num:03d}.npz')
                np.savez_compressed(
                    batch_file,
                    sequences=batch_combined_seq,
                    labels=batch_combined_labels,
                    metadata=batch_metadata
                )

                batch_files_created.append(batch_file)
                self.logger.info(f"✓ Saved batch {batch_num}: {len(batch_combined_seq)} sequences")

                # Clear memory immediately
                del batch_sequences, batch_labels, batch_metadata
                del batch_combined_seq, batch_combined_labels

                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        # Save processing info
        with open(os.path.join(output_dir, 'batch_info.pkl'), 'wb') as f:
            pickle.dump({
                'batch_files': batch_files_created,
                'total_sequences': total_sequences,
                'seizure_sequences': seizure_sequences
            }, f)

        # Save other metadata
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(os.path.join(output_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)

        with open(os.path.join(output_dir, 'channel_names.pkl'), 'wb') as f:
            pickle.dump(self.channel_names, f)

        self.logger.info(f"Processing complete! {len(batch_files_created)} batch files created")

        return {
            'total_sequences': total_sequences,
            'seizure_sequences': seizure_sequences
        }


    def save_processed_data(self, sequences: np.ndarray, labels: np.ndarray,
                            metadata: pd.DataFrame, output_dir: str):
        """
        Save processed data and configuration
        Serialization - Converting objects to binary files
        """

        # Save sequences and labels as numpy arrays (more efficient)
        np.save(os.path.join(output_dir, 'sequences.npy'), sequences)
        np.save(os.path.join(output_dir, 'labels.npy'), labels)

        # Save metadata
        metadata.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)

        # Save scaler and config
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f: # contains computed statistics from training data
            pickle.dump(self.scaler, f)

        with open(os.path.join(output_dir, 'config.pkl'), 'wb') as f: # stores hyperparameters
            pickle.dump(self.config, f)

        # Save channel names - EEG channel order: ['Fp1-F7', 'F7-T7', 'T7-P7'..]
        with open(os.path.join(output_dir, 'channel_names.pkl'), 'wb') as f:
            pickle.dump(self.channel_names, f)

        self.logger.info(f"Processed data saved to {output_dir}")

        # Save summary statistics
        summary = {
            'total_sequences': len(sequences),
            'seizure_sequences': int(np.sum(labels)),
            'sequence_shape': sequences.shape,
            'channels': len(self.channel_names),
            'channel_names': self.channel_names,
            'sequence_length': self.config.sequence_length,
            'sampling_rate': self.config.sampling_rate,
            'overlap': self.config.overlap
        }

        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(os.path.join(output_dir, 'dataset_summary.csv'), index=False)


    def load_processed_data(self, data_dir: str) -> Dict:
        """
        Load previously processed data from saved files back into memory for model training or analysis
        """

        self.logger.info(f"Loading processed data from {data_dir}")

        sequences = np.load(os.path.join(data_dir, 'sequences.npy')) # Loads: (n_sequences, sequence_length, n_channels) - transformer-ready EEG data
        labels = np.load(os.path.join(data_dir, 'labels.npy')) # Loads: (n_sequences,) - seizure labels (0/1) for each sequence
        metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv')) # Loads: DataFrame with recording_id, sequence_idx, file_path info

        with open(os.path.join(data_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)

        with open(os.path.join(data_dir, 'config.pkl'), 'rb') as f:
            self.config = pickle.load(f)

        with open(os.path.join(data_dir, 'channel_names.pkl'), 'rb') as f:
            self.channel_names = pickle.load(f)

        self.logger.info(f"Loaded dataset: {sequences.shape}")
        self.logger.info(f"Seizure ratio: {np.mean(labels) * 100:.1f}%")

        return {
            'sequences': sequences,
            'labels': labels,
            'metadata': metadata,
            'config': self.config
        }

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        return logging.getLogger(__name__)

    def debug_seizure_labels(self, output_dir: str = "processed_data/transformer_sequences"):
        """Debug why no seizure sequences were found"""

        print("🔍 Debugging seizure labels...")

        # Load batch info
        try:
            with open(os.path.join(output_dir, 'batch_info.pkl'), 'rb') as f:
                batch_info = pickle.load(f)
        except FileNotFoundError:
            print(f"❌ No batch_info.pkl found in {output_dir}")
            return

        print(f"Found {len(batch_info['batch_files'])} batch files")
        print(f"Expected: {batch_info['total_sequences']} total, {batch_info['seizure_sequences']} seizure\n")

        total_seizure_found = 0

        for batch_file in batch_info['batch_files']:
            if not os.path.exists(batch_file):
                print(f"❌ Missing batch file: {batch_file}")
                continue

            data = np.load(batch_file, allow_pickle=True)
            sequences = data['sequences']
            labels = data['labels']
            metadata = data['metadata']

            seizure_count = np.sum(labels)
            total_seizure_found += seizure_count

            print(f"{os.path.basename(batch_file)}: {seizure_count} seizure sequences out of {len(labels)}")

            if seizure_count > 0:
                seizure_indices = np.where(labels == 1)[0]
                for idx in seizure_indices[:3]:
                    meta = metadata[idx]
                    print(f"  ✓ Seizure in {meta['recording_id']}, sequence {meta['sequence_idx']}")

        print(f"\n📊 Summary:")
        print(f"   - Total seizure sequences found: {total_seizure_found}")
        print(f"   - Expected seizure sequences: {batch_info['seizure_sequences']}")

        if total_seizure_found == 0:
            print("\n🚨 No seizure sequences found! Checking recording IDs...")

            # Show sample recording IDs from first batch
            if batch_info['batch_files']:
                first_batch = np.load(batch_info['batch_files'][0], allow_pickle=True)
                metadata = first_batch['metadata']
                unique_recordings = set()
                for meta in metadata[:10]:
                    unique_recordings.add(meta['recording_id'])

                print("\n📁 Recording IDs found in data:")
                for rec_id in sorted(unique_recordings):
                    print(f"   - '{rec_id}'")


def main():

    config = TransformerConfig(
        sequence_length=256,
        sampling_rate = 256,
        overlap = 0.25,
        normalization = 'standard',
        filter_low = 0.5,
        filter_high = 50.0,
        max_sequences_per_file=1000,  # Limit sequences per file
        batch_process_size=6  # Process only 6 files at a time
    )

    seizure_info = {
        "chb01_chb01_03": [[2996, 3036]],
        "chb01_chb01_04": [[1467, 1494]],
        "chb01_chb01_15": [[1732, 1772]],
        "chb01_chb01_16": [[1015, 1066]],
        "chb01_chb01_18": [[1720, 1810]],
        "chb01_chb01_21": [[327, 420]],
        "chb01_chb01_26": [[1862, 1963]],
        "chb02_chb02_16": [[130, 212]],
        "chb02_chb02_16+": [[2972, 3053]],
        "chb02_chb02_19": [[3369, 3378]],
        "chb03_chb03_01": [[362, 414]],
        "chb03_chb03_02": [[731, 796]],
        "chb03_chb03_03": [[432, 501]],
        "chb03_chb03_04": [[2162, 2214]],
        "chb03_chb03_34": [[1982, 2029]],
        "chb03_chb03_35": [[2592, 2656]],
        "chb03_chb03_36": [[1725, 1778]]
    }

    dataset = 'dataset'
    output_dir = "processed_data/transformer_sequences"

    preprocessor = TransformerProcessor(config)

    try:
        # Process dataset
        print(f"Processing EDF files from: {dataset}")
        print(f"Output will be saved to: {output_dir}")
        print(f"Configuration:")
        print(f"  - Sequence length: {config.sequence_length} samples (1 second)")
        print(f"  - Overlap: {config.overlap * 100:.0f}%")
        print(f"  - Sampling rate: {config.sampling_rate} Hz")
        print(f"  - Normalization: {config.normalization}")
        print()

        result = preprocessor.process_dataset(
            data_dir=dataset,
            seizure_info=seizure_info,
            output_dir=output_dir
        )

        print("\n" + "=" * 50)
        print("Preprocessing completed successfully!")
        print(f"Dataset Statistics:")
        print(f"   - Total sequences: {result['total_sequences']:,}")
        print(f"   - Seizure sequences: {result['seizure_sequences']:,}")
        if result['total_sequences'] > 0:
            seizure_ratio = result['seizure_sequences'] / result['total_sequences'] * 100
            print(f"   - Seizure ratio: {seizure_ratio:.1f}%")

        # Debug seizure labeling if no seizures found
        if result['seizure_sequences'] == 0:
            print("\n⚠️  No seizure sequences found - running debug...")
            preprocessor.debug_seizure_labels(output_dir)

            print("\n💡 To fix this, check if your seizure_info keys match the recording IDs")
        else:
            seizure_ratio = result['seizure_sequences'] / result['total_sequences'] * 100
            print(f"   - Seizure ratio: {seizure_ratio:.1f}%")

        print(f"\nData saved to: {output_dir}")
        print(f"Files created:")
        print(f"   - batch_001.npz, batch_002.npz, ... (sequence data)")
        print(f"   - batch_info.pkl (batch file list)")
        print(f"   - config.pkl, scaler.pkl, channel_names.pkl")

    except Exception as e:
            print(f"Preprocessing failed: {str(e)}")
            traceback.print_exc()


if __name__ == '__main__':
    main()