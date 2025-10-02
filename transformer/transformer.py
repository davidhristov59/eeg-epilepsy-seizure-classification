import logging
import numpy as np
import torch
import os
import time
import pickle
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from preprocessing_transformer import TransformerProcessor, EEGSequenceDataset
from transformer_models import CNNTransformer, ModelConfig
from mps_configuration import DeviceManager

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
        self.logger = self._setup_logging()

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
        Load and preprocess EEG data from batch files, returning DataLoaders for training, validation, and testing.
        """
        self.logger.info("Loading processed data from batch files...")

        data_dir = self.config.processed_data_dir

        # Load batch info
        batch_info_path = os.path.join(data_dir, 'batch_info.pkl')
        if not os.path.exists(batch_info_path):
            raise FileNotFoundError(f"batch_info.pkl not found in {data_dir}")

        with open(batch_info_path, 'rb') as f:
            batch_info = pickle.load(f)

        self.logger.info(f"Found {len(batch_info['batch_files'])} batch files")

        # Load all batches
        total_samples = 0
        all_labels_list = []

        for batch_file in tqdm(batch_info['batch_files'], desc="Scanning batches"):
            if not os.path.exists(batch_file):
                self.logger.warning(f"Batch file not found: {batch_file}")
                continue

            data = np.load(batch_file, allow_pickle=True)
            labels = data['labels']
            all_labels_list.append(labels)
            total_samples += len(labels)


            if total_samples == len(labels):  # First batch
                self.config.sequence_length = data['sequences'].shape[1]
                self.config.input_dim = data['sequences'].shape[2]


        # Concatenate only labels for stratified split
        all_labels = np.concatenate(all_labels_list, axis=0)
        all_indices = np.arange(total_samples)

        self.logger.info(f"Total samples: {total_samples:,}")
        self.logger.info(f"Seizure samples: {np.sum(all_labels):,} ({np.mean(all_labels) * 100:.1f}%)")

        # Split indices with stratification
        idx_temp, idx_test, y_temp, y_test = train_test_split(
            all_indices, all_labels,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=all_labels
        )

        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
        idx_train, idx_val, y_train, y_val = train_test_split(
            idx_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=y_temp
        )

        self.train_labels = y_train

        self.logger.info(f"Data splits:")
        self.logger.info(f"  Train: {len(idx_train):,} samples ({np.mean(y_train) * 100:.1f}% seizure)")
        self.logger.info(f"  Val:   {len(idx_val):,} samples ({np.mean(y_val) * 100:.1f}% seizure)")
        self.logger.info(f"  Test:  {len(idx_test):,} samples ({np.mean(y_test) * 100:.1f}% seizure)")

        # Create Datasets
        train_dataset = EEGSequenceDataset(batch_info['batch_files'], idx_train.tolist())
        val_dataset = EEGSequenceDataset(batch_info['batch_files'], idx_val.tolist())
        test_dataset = EEGSequenceDataset(batch_info['batch_files'], idx_test.tolist())

        # Create DataLoaders
        data_loaders = {
            "train": DataLoader(
                dataset=train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory and self.device.type != "mps",
                persistent_workers=True if self.config.num_workers > 0 else False
            ),

            "val": DataLoader(
                dataset=val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory and self.device.type != "mps",
                persistent_workers=True if self.config.num_workers > 0 else False
            ),

            "test": DataLoader(
                dataset=test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory and self.device.type != "mps",
                persistent_workers=True if self.config.num_workers > 0 else False
            )
        }

        return data_loaders


    def build_model(self, data_loaders: Dict[str, DataLoader]):
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
        self.model = CNNTransformer(model_config).to(self.device)

        # Count parameters
        total_params = self.model.count_parameters(self.model)
        model_size_mb = total_params * 4 / 1024 ** 2 # calculates the memory size in MB
        self.logger.info(f"Model: {total_params:,} parameters (~{model_size_mb:.1f} MB)")

        self.logger.info("Calculating class weights from training labels...")
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.train_labels),
            y=self.train_labels
        )
        class_weights = torch.FloatTensor(class_weights).to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            params = self.model.parameters(),
            lr = self.config.learning_rate,
            weight_decay = self.config.weight_decay, # L2 regularization, penalizes large weights
            eps = 1e-8 # to prevent division by zero in AdamW algorithm
        )

        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss(weight = class_weights) # for binary classification

        # Create learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( # reduce LR when a metric has stopped improving
            optimizer = self.optimizer,
            mode = 'min',
            factor = 0.5,
            patience = 3,
            min_lr= 1e-7
        )

        self.logger.info("Model and training components initialized")


    def train_epoch(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Train the model for one epoch - complete pass through the data.
        """

        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(data_loader, desc="Training", leave=False,
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # Batch processing
        for batch_idx, (sequences, labels) in enumerate(progress_bar):
            try:
                sequences = sequences.to(self.device, non_blocking=(self.device.type == "mps"))
                labels = labels.to(self.device, non_blocking=(self.device.type == "mps"))

                # Zero gradients - clears old gradients from the last step
                self.optimizer.zero_grad()

                # Forward pass - compute model output, gets predictions
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)

                # Backward pass - calculate gradients
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Update weights - updates model parameters
                self.optimizer.step()

                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                current_acc = 100 * correct / total
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.1f}%',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })

                # Clear cache periodically for MPS
                if batch_idx % 50 == 0 and batch_idx > 0:
                    DeviceManager.clear_memory(self.device)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.warning(f"Memory warning at batch {batch_idx}")
                    DeviceManager.clear_memory(self.device)
                    continue
                else:
                    raise e

        epoch_loss = running_loss / len(data_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc


    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float, Dict]:
        """
        Validate the model for one epoch - complete pass through the validation data.
        """

        self.model.eval() # will disable dropout and batchnorm layers

        running_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad(): # to disable gradient computation for validation
            progress_bar = tqdm(dataloader, desc="Validating", leave=False)

            for sequences, labels in progress_bar:
                sequences = sequences.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Forward pass
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collect for detailed metrics
                probabilities = torch.softmax(outputs, dim=1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())

                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct / total:.1f}%'
                })

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total

        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities)

        return epoch_loss, epoch_acc, metrics


    def _calculate_metrics(self, true_labels: List[int], predictions: List[int], probabilities: List[float]) -> Dict:
        """
        Calculate detailed metrics like precision, recall, F1-score.
        """

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )

        precision_classes, recall_classes, f1_classes, _ = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )

        roc_auc = roc_auc_score(true_labels, probabilities)

        cm = confusion_matrix(true_labels, predictions)

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'seizure_precision': precision_classes[1] if len(precision_classes) > 1 else 0,
            'seizure_recall': recall_classes[1] if len(recall_classes) > 1 else 0,
            'seizure_f1': f1_classes[1] if len(f1_classes) > 1 else 0,
        }

        return metrics

    def load_model(self):
        """Load model checkpoint"""

        if not os.path.exists(self.config.model_save_path):
            raise FileNotFoundError(f"Model file not found: {self.config.model_save_path}")

        checkpoint = torch.load(self.config.model_save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load training history
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_accuracies = checkpoint['val_accuracies']

        self.logger.info(f"Model loaded from {self.config.model_save_path}")


    def save_model(self):
        """
        Save model checkpoint
        """

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'device_name': self.device_name,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'epoch': len(self.train_losses)
        }

        torch.save(checkpoint, self.config.model_save_path)


    def train(self, dataloader: Dict[str, DataLoader]) -> Dict:
        """
        Full training loop with early stopping and model saving.
        """

        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Device: {self.device_name}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Learning rate: {self.config.learning_rate}")

        best_val_loss = float('inf')
        best_metrics = None
        start_time = time.time()

        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()

            # Clear memory at start of epoch
            DeviceManager.clear_memory(device = self.device)

            # Training
            train_loss, train_acc = self.train_epoch(dataloader['train'])

            # Validation
            val_loss, val_acc, val_metrics = self.validate_epoch(dataloader['val'])

            # Update learning rate
            self.scheduler.step(val_loss)

            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            epoch_time = time.time() - epoch_start_time

            memory_info = DeviceManager.get_memory_info(self.device)
            self.logger.info(
                f"Epoch {epoch + 1:3d}/{self.config.num_epochs} | "
                f"Train: {train_loss:.4f}/{train_acc:5.1f}% | "
                f"Val: {val_loss:.4f}/{val_acc:5.1f}% | "
                f"F1: {val_metrics['f1_score']:.3f} | AUC: {val_metrics['roc_auc']:.3f} | "
                f"Time: {epoch_time:4.1f}s"
            )

            if epoch % 10 == 0:
                self.logger.info(f"Memory info: {memory_info}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics.copy()
                self.save_model() # Save best model
                self.logger.info(f"New best model saved (Val Loss: {val_loss:.4f})")

            # Early stopping check
            if self.early_stopping(val_loss):
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break


        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/60:.1f} minutes")

        return best_metrics


    def evaluate(self, dataloader: DataLoader) -> Dict:

        """
        Evaluate the trained model on the test set
        """

        self.load_model()

        DeviceManager.clear_memory(self.device)

        # Evaluate
        test_loss, test_acc, test_metrics = self.validate_epoch(dataloader)

        self.logger.info("TEST SET RESULTS")
        self.logger.info(f"Test Loss: {test_loss:.4f}%")
        self.logger.info(f"Test Accuracy: {test_acc:.4f}%")
        self.logger.info(f"Test F1 Score:  {test_metrics['f1']:.3f}")
        self.logger.info(f"Test AUC:  {test_metrics['auc']:.3f}")

        return test_metrics


    def save_results(self, test_metrics: Dict):
        """
        Save training results and training history to CSV files
        """

        results = {
            'model': 'CNN-Transformer',
            'device': self.device_name,
            'test_accuracy': test_metrics.get('accuracy', 0), # use precision as proxy for accuracy, if not try 'accuracy'
            'test_precision': test_metrics.get('precision', 0),
            'test_roc_auc': test_metrics.get('roc_auc', 0),
            'test_recall': test_metrics.get('recall', 0),
            'test_f1_score': test_metrics.get('f1_score', 0),
            'seizure_precision': test_metrics.get('seizure_precision', 0),
            'seizure_recall': test_metrics.get('seizure_recall', 0),
            'seizure_f1': test_metrics.get('seizure_f1', 0),
            'total_params': self.model.count_parameters(self.model) if self.model else 0,
            'training_epochs': len(self.train_losses),
            'best_val_loss': min(self.val_losses) if self.val_losses else 0,
            'final_epoch': len(self.train_losses) if self.train_losses else 0,
            'final_lr': self.optimizer.param_groups[0]['lr'],
            'batch_size': self.config.batch_size,
            'd_model': self.config.d_model,
            'num_layers': self.config.num_layers
        }
        df = pd.DataFrame([results])
        results_file = os.path.join(self.config.output_dir, 'transformer_results.csv')
        df.to_csv(results_file, index=False)

        # Save training history
        training_history = pd.DataFrame({
            'epoch': list(range(1, len(self.train_losses))),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_accuracy': self.train_accuracies,
            'val_accuracy': self.val_accuracies
        })
        history_file = os.path.join(self.config.output_dir, 'training_history.csv')
        training_history.to_csv(history_file, index=False)

        # Save confusion matrix
        cm_df = pd.DataFrame(test_metrics['confusion_matrix'],
                             index=['True_Non_Seizure', 'True_Seizure'],
                             columns=['Predicted_Non_Seizure', 'Predicted_Seizure'])
        cm_file = os.path.join(self.config.output_dir, 'confusion_matrix.csv')
        cm_df.to_csv(cm_file)

        self.logger.info(f"Results saved to {self.config.output_dir}")


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


def main():

    config = TransformerConfig(

        # Model parameters
        d_model=64,
        nhead=8,
        num_layers=3,
        cnn_channels=[32, 64, 128],
        dropout=0.1,

        # Training
        batch_size=8,
        num_epochs=50,
        learning_rate=1e-4,
        patience=10,

        # MPS
        num_workers=0,
        pin_memory=False,

        # Paths
        processed_data_dir="processed_data/transformer_sequences",
        output_dir="output/transformer_results",
        model_save_path="models/cnn_transformer_best.pth"
    )

    print(f"Data: {config.processed_data_dir}")
    print(f"Output: {config.output_dir}")
    print(f"Batch size: {config.batch_size} ")
    print(f"Max epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print()

    # Initialize trainer
    trainer = CNNTransformerTrainingPipeline(config)

    try:
        dataloaders = trainer.load_data()

        print("Building CNN-Transformer model...")
        trainer.build_model(dataloaders)

        print("Starting training...")
        print(f"Model saved to: {config.model_save_path}")
        print(f"Results saved to: {config.output_dir}")
        print(f"Trained on: {trainer.device_name}")
        print()

        best_val_metrics = trainer.train(dataloaders)

        print("Evaluating on test set...")
        test_metrics = trainer.evaluate(dataloaders['test'])

        print("Saving results...")
        trainer.save_results(test_metrics)

        print("Training completed successfully!")
        print("FINAL RESULTS:")
        print(f"Test F1 Score: {test_metrics['f1']:.3f}")
        print(f"Test AUC: {test_metrics['auc']:.3f}")
        print(f"Seizure F1: {test_metrics['seizure_f1']:.3f}")

        print(f"Model saved to: {config.model_save_path}")
        print(f"Results saved to: {config.output_dir}")
        print(f"Trained on: {trainer.device_name}")
        print()

    except FileNotFoundError as e:
        print(f"File not found: {str(e)}")
        print("Solution: Run transformer_preprocessing.py first!")
        print("This will convert your EDF files to transformer-ready sequences")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Out of memory error: {str(e)}")
            print("Solutions for Mac:")
            print("   1. Reduce batch_size in config (try 8 or 4)")
            print("   2. Reduce d_model (try 64 or 96)")
            print("   3. Reduce num_layers (try 3 or 2)")
            print("   4. Close other applications to free memory")
        else:
            print(f"Runtime error: {str(e)}")

    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

        print("\n Troubleshooting tips:")
        print("   1. Check if MPS is available: python -c 'import torch; print(torch.backends.mps.is_available())'")
        print("   2. Update PyTorch: pip install --upgrade torch torchvision")
        print("   3. Check system resources and close other apps")

if __name__ == '__main__':
    main()
