import logging
import numpy as np
import psutil
import torch
import os

from dataclasses import dataclass
from typing import List, Dict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from preprocessing_transformer import TransformerProcessor, EEGSequenceDataset
from transformer_models import CNNTransformer, ModelConfig

@dataclass
class TransformerConfig:
    # Model Architecture
    input_dim: int = 23 # Number of EEG channels
    sequence_length: int = 256 # Length of each EEG sequence
    d_model: int = 128 # Embedding dimension for the transformer, all layers in the transformer use 128-dimensional vectors
    nhead: int = 8 # Number of attention heads in the multi-head attention mechanism
    num_layers: int = 4 # Number of transformer encoder layers
    cnn_channels: List[int] = None
    dropout: float = 0.1 # Dropout rate for regularization, 10% of neurons are randomly dropped during training


    # Training Hyperparameters
    batch_size: int = 16 # number of samples per gradient update (one forward pass)
    num_epochs: int = 50
    learning_rate: float = 1e-4 # step size for weight updates during training
    weight_decay: float = 1e-5 # L2 regularization to prevent overfitting, penalizes large weights
    patience: int = 5 # Early stopping patience

    # Data split
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42

    # Paths
    processed_data_dir: str = "processed_data/transformer_sequences"
    output_dir: str = "output/transformer_results"
    model_save_path: str = "models/cnn_transformer_best.pth"

    # MPS
    use_mixed_precision: bool = False
    pin_memory: bool = False
    num_workers: int = 0

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [64, 128, 256] # 3-layer CNN with increasing channels, sets if the cnn_channels are not provided during initialization
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class CNNTransformerTrainingPipeline:
    """
    Pipeline to handle initialization, data loading, model training, and evaluation for CNN-Transformer on EEG data.
    """

    def __init__(self, config: TransformerConfig):
        self.config = config
        self.device, self.device_name = DeviceManager.get_optimal_device()
        self.logger = self.setup_logger()

        # Model Components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.early_stopping = EarlyStopping(patience=self.config.patience)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        self.logger.info(f"Device: {self.device_name}")
        self.logger.info(f"System: {DeviceManager.get_memory_info(self.device)}")


    def load_data(self) -> Dict[str, DataLoader]:
        """
        Load and preprocess EEG data, returning DataLoaders for training, validation, and testing.
        """

        self.logger.info("Loading processed data...")

        processor = TransformerProcessor(config=self.config)
        dataset = processor.load_processed_data(data_dir=self.config.processed_data_dir)

        sequences = dataset['sequences']
        labels = dataset['labels']

        self.logger.info(f"Loaded data shape: {sequences.shape}")
        self.logger.info(f"Memory usage: ~{sequences.nbytes / 1024 ** 2:.0f} MB")
        self.logger.info(f"Total samples: {len(sequences):,}")
        self.logger.info(f"Seizure samples: {np.sum(labels):,} ({np.mean(labels) * 100:.1f}%)")

        # sequences.shape = (num_samples, sequence_length, num_channels)
        self.config.sequence_length = sequences.shape[1]
        self.config.input_dim = sequences.shape[2]

        # Split data with stratification
        x_temp, x_test, y_temp, y_test = train_test_split(
            sequences, labels,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=labels # use when the dataset is imbalanced , it ensures that all splits maintain original class distribution
        )

        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_temp, y_temp, # use the remainder
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=y_temp # keep seizure ratio
        )

        self.logger.info(f"Data splits:")
        self.logger.info(f"  Train: {len(x_train):,} samples ({np.mean(y_train) * 100:.1f}% seizure)")
        self.logger.info(f"  Val:   {len(x_val):,} samples ({np.mean(y_val) * 100:.1f}% seizure)")
        self.logger.info(f"  Test:  {len(x_test):,} samples ({np.mean(y_test) * 100:.1f}% seizure)")

        # Create Datasets
        train_dataset = EEGSequenceDataset(x_train, y_train)
        val_dataset = EEGSequenceDataset(x_val, y_val)
        test_dataset = EEGSequenceDataset(x_test, y_test)

        # Create DataLoaders (loading and batching data during training and evaluation)

        data_loaders = {
            "train" : DataLoader(
                dataset=train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory and self.device.type != "mps", # false for MAC
                persistent_workers = True if self.config.num_workers > 0 else False
            ),

            "val" : DataLoader(
                dataset=val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory and self.device.type != "mps",  # false for MAC
                persistent_workers=True if self.config.num_workers > 0 else False
            ),

            "test" : DataLoader(
                dataset=test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory and self.device.type != "mps",  # false for MAC
                persistent_workers=True if self.config.num_workers > 0 else False
            )
        }

        return data_loaders


    def build_model(self):
        """
        Build the CNN-Transformer model, optimizer, loss function, and learning rate scheduler.
        """

        self.logger.info("Building CNN-Transformer model...")

        # Create model config
        model_config = ModelConfig(
            input_dim = self.config.input_dim,
            sequence_length = self.config.sequence_length,
            d_model = self.config.d_model,
            nhead = self.config.nhead,
            num_layers = self.config.num_layers,
            cnn_channels = self.config.cnn_channels,
            dropout = self.config.dropout
        )

        # Create the model
        self.model = CNNTransformer()








    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)


class DeviceManager:
    """Handles device detection and optimization for Mac/MPS"""

    @staticmethod
    def get_optimal_device():
        """Get the best available device with Mac optimization"""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            device_name = "Apple Silicon GPU (MPS)"
            print(f"Using {device_name}")

            # Set optimal CPU threads for MPS
            cpu_count = psutil.cpu_count()
            torch.set_num_threads(min(8, cpu_count))
            print(f"CPU threads: {torch.get_num_threads()}")

        else: # if no MPS available, fallback to CPU
            device = torch.device("cpu")
            device_name = "CPU"

            # Optimize for ARM CPU
            cpu_count = psutil.cpu_count()
            optimal_threads = min(8, cpu_count)
            torch.set_num_threads(optimal_threads)
            print(f"Using {device_name} with {optimal_threads} threads")

        return device, device_name

    @staticmethod
    def clear_memory(device):
        """Clear memory cache based on device"""
        if device.type == "mps":
            torch.mps.empty_cache()

    @staticmethod
    def get_memory_info(device):
        """Get memory information if available"""
        if device.type == "mps":
            return "MPS memory management handled automatically"

        else:
            memory = psutil.virtual_memory()
            return f"RAM: {memory.used / 1024 ** 3:.1f}GB / {memory.total / 1024 ** 3:.1f}GB ({memory.percent:.1f}% used)"

