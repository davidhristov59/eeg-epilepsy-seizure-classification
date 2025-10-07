import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging
import pyeeg
from pyeeg import bin_power
pyeeg.bin_power = bin_power
from scipy import stats
from scipy.signal import welch, argrelextrema
from typing import Optional, List, Tuple, Dict
from mne import io
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings('ignore')


@dataclass
class EEGFeatures:
    temporal: Dict[str, float]
    spectral: Dict[str, float]
    nonlinear: Dict[str, float]


class SignalProcessor:
    def __init__(self, sampling_rate: int = 256):
        self.fs = sampling_rate
        self.freq_bands = np.array([1, 5, 10, 15, 20, 25])
        self.nyquist = self.fs / 2

    def extract_temporal_features(self, signals: np.ndarray) -> Dict[str, np.ndarray]:
        means = np.mean(signals, axis=1)
        variances = np.var(signals, axis=1)
        skewnesses = stats.skew(signals, axis=1)
        kurtoses = stats.kurtosis(signals, axis=1)
        rms_values = np.sqrt(np.mean(np.square(signals), axis=1))
        zero_crossings = np.sum(np.diff(np.signbit(signals).astype(int), axis=1) != 0, axis=1)
        peak_amps = np.max(np.abs(signals), axis=1)

        peak_counts = np.zeros(signals.shape[0])
        for i in range(signals.shape[0]):
            peaks = argrelextrema(signals[i], np.greater)[0]
            peak_counts[i] = len(peaks)

        return {
            "mean": means,
            "variance": variances,
            "skewness": skewnesses,
            "kurtosis": kurtoses,
            "rms": rms_values,
            "zero_crossings": zero_crossings,
            "peak_amp": peak_amps,
            "peak_count": peak_counts,
        }

    def extract_spectral_features(self, signals: np.ndarray) -> Dict[str, np.ndarray]:
        n_channels = signals.shape[0]
        results = {
            "total_power": np.zeros(n_channels),
            "median_freq": np.zeros(n_channels),
            "peak_freq": np.zeros(n_channels),
            "spec_entropy": np.zeros(n_channels),
        }
        for i, band in enumerate(self.freq_bands[:-1]):
            results[f"band_power_{band}"] = np.zeros(n_channels)

        for ch_idx in range(n_channels):
            signal = signals[ch_idx]
            freqs, psd = welch(signal, fs=self.fs, nperseg=min(256, len(signal)//4))
            results["total_power"][ch_idx] = np.sum(psd)
            cumsum_psd = np.cumsum(psd)
            median_idx = np.where(cumsum_psd >= np.sum(psd)/2)[0]
            results["median_freq"][ch_idx] = freqs[median_idx[0]] if len(median_idx) > 0 else 0
            results["peak_freq"][ch_idx] = freqs[np.argmax(psd)]
            psd_norm = psd / np.sum(psd)
            results["spec_entropy"][ch_idx] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

            try:
                power, power_ratio = bin_power(signal, self.freq_bands, self.fs)
                for i, band in enumerate(self.freq_bands[:-1]):
                    results[f"band_power_{band}"][ch_idx] = power_ratio[i]
            except:
                for i, band in enumerate(self.freq_bands[:-1]):
                    results[f"band_power_{band}"][ch_idx] = 0

        return results

    def extract_nonlinear_features(self, signals: np.ndarray) -> Dict[str, np.ndarray]:
        n_channels = signals.shape[0]
        results = {
            "hjorth_mobility": np.zeros(n_channels),
            "hjorth_complexity": np.zeros(n_channels),
            "hfd": np.zeros(n_channels),
            "pfd": np.zeros(n_channels),
            "hurst": np.zeros(n_channels)
        }

        for ch_idx in range(n_channels):
            signal = signals[ch_idx]
            try:
                mobility, complexity = pyeeg.hjorth(signal)
                results["hjorth_mobility"][ch_idx] = mobility
                results["hjorth_complexity"][ch_idx] = complexity
                results["hfd"][ch_idx] = pyeeg.hfd(signal, Kmax=3)
                results["pfd"][ch_idx] = pyeeg.pfd(signal)
                results["hurst"][ch_idx] = pyeeg.hurst(signal)
            except:
                results["hjorth_mobility"][ch_idx] = 0
                results["hjorth_complexity"][ch_idx] = 0
                results["hfd"][ch_idx] = 1.0
                results["pfd"][ch_idx] = 1.0
                results["hurst"][ch_idx] = 0.5

        return results

    def extract_all_features_batch(self, signals: np.ndarray, channel_names: List[str]) -> Dict[str, float]:
        temporal = self.extract_temporal_features(signals)
        spectral = self.extract_spectral_features(signals)
        nonlinear = self.extract_nonlinear_features(signals)

        features = {}
        for ch_idx, channel in enumerate(channel_names):
            for feature_name, values in temporal.items():
                features[f"{channel}_{feature_name}"] = float(values[ch_idx])
            for feature_name, values in spectral.items():
                features[f"{channel}_{feature_name}"] = float(values[ch_idx])
            for feature_name, values in nonlinear.items():
                features[f"{channel}_{feature_name}"] = float(values[ch_idx])
        return features


class EEGProcessor:
    def __init__(self, processor: SignalProcessor, epoch_length: int = 10, step_size: int = 5):
        self.processor = processor
        self.epoch_length = epoch_length
        self.step_size = step_size
        self.logger = logging.getLogger(__name__)

    def load_and_filter(self, file_path: str) -> io.Raw:
        raw = io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.filter(l_freq=0.25, h_freq=25, verbose=False)
        return raw

    def process_recording(self, file_path: str,
                          seizure_intervals: Optional[List[Tuple[float, float]]] = None) -> pd.DataFrame:
        raw = self.load_and_filter(file_path)

        start_time = 0
        epochs = []
        epoch_times = []

        while start_time <= raw.times[-1] - self.epoch_length:
            epoch_times.append(start_time)
            start_time += self.step_size

        total_epochs = len(epoch_times)
        self.logger.info(f"Processing {total_epochs} epochs (step_size={self.step_size}s)")

        batch_size = min(100, total_epochs)

        with tqdm(total=total_epochs, desc=f"Processing {os.path.basename(file_path)}", unit="epoch") as pbar:
            for batch_start in range(0, total_epochs, batch_size):
                batch_end = min(batch_start + batch_size, total_epochs)
                batch_epochs = []

                for i in range(batch_start, batch_end):
                    start_time = epoch_times[i]
                    start, stop = raw.time_as_index([start_time, start_time + self.epoch_length])
                    data = raw[:, start:stop][0]

                    features = {"start_time": start_time}
                    channel_features = self.processor.extract_all_features_batch(data, raw.ch_names)
                    features.update(channel_features)

                    if seizure_intervals:
                        features["seizure"] = any(
                            start_time < end and start_time + self.epoch_length > start
                            for start, end in seizure_intervals
                        )
                    else:
                        features["seizure"] = 0

                    batch_epochs.append(features)
                    pbar.update(1)

                epochs.extend(batch_epochs)

        return pd.DataFrame(epochs)


def process_single_file(args):
    file_path, recording_id, seizure_intervals, output_dir = args
    try:
        start_time = time.time()

        parts = recording_id.split("_")
        subject_id = parts[0] if parts else "unknown"

        signal_processor = SignalProcessor()
        eeg_processor = EEGProcessor(signal_processor, epoch_length=10, step_size=5)

        features_df = eeg_processor.process_recording(file_path, seizure_intervals)
        features_df["subject"] = subject_id
        features_df["recording_id"] = recording_id
        if "seizure" not in features_df.columns:
            features_df["seizure"] = 0

        safe_recording_id = recording_id.replace("+", "_plus")
        output_path = os.path.join(output_dir, f"{safe_recording_id}.csv")

        if os.path.exists(output_path):
            return f"⏭ {recording_id}: CSV already exists, skipping"

        features_df.to_csv(output_path, index=False)

        elapsed = time.time() - start_time
        return f"✓ {recording_id}: {len(features_df)} epochs in {elapsed:.1f}s -> {output_path}"

    except Exception as e:
        return f"✗ {recording_id}: ERROR - {str(e)}"


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    print("Running preprocessing script...")

    # Set project root for consistent paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'dataset')
    output_dir = os.path.join(project_root, 'processed_data')
    os.makedirs(output_dir, exist_ok=True)

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

    # Find EDF files
    edf_files = []
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith(".edf"):
                file_path = os.path.join(root, filename)
                recording_id = os.path.splitext(os.path.relpath(file_path, data_dir).replace(os.sep, "_"))[0]

                safe_recording_id = recording_id.replace("+", "_plus")
                output_path = os.path.join(output_dir, f"{safe_recording_id}.csv")

                if os.path.exists(output_path):
                    logger.info(f"⏭ Skipping {recording_id} - CSV already exists")
                    continue

                edf_files.append((file_path, recording_id, seizure_info.get(recording_id), output_dir))

    edf_files.sort(key=lambda x: x[1])
    logger.info(f"Found {len(edf_files)} EDF files to process")
    logger.info(f"Available CPU cores: {os.cpu_count()}")

    if not edf_files:
        logger.info("All files already processed!")
        return

    logger.info("Processing order:")
    for i, (_, recording_id, _, _) in enumerate(edf_files[:5], 1):
        logger.info(f"   {i}. {recording_id}")
    if len(edf_files) > 5:
        logger.info(f"   ... and {len(edf_files) - 5} more")

    max_workers = min(os.cpu_count() - 1, len(edf_files), 4)
    logger.info(f"Processing {len(edf_files)} files using {max_workers} workers")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_single_file, edf_files),
                            total=len(edf_files), desc="Processing files", unit="file"))

    for result in results:
        logger.info(result)

    elapsed_time = time.time() - time.time()
    logger.info(f"Total processing time: {elapsed_time:.1f} seconds")
    logger.info("All files processed!")


if __name__ == '__main__':
    main()
