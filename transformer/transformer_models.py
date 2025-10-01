import math
from dataclasses import dataclass
from typing import List

import torch
from torch import nn

@dataclass
class ModelConfig:

    # Input dimensions
    input_dim: int = 23  # Number of EEG channels
    sequence_length: int = 256  # Sequence length (time steps)
    num_classes: int = 2  # Binary classification (seizure/non-seizure)

    # CNN parameters
    cnn_channels: List[int] = None  # [64, 128, 256] - feature maps per conv layer
    kernel_size: int = 3 # 3-point convolution window
    pool_size: int = 2 # 2-point max pooling

    # Transformer parameters
    d_model: int = 128  # Model dimension
    nhead: int = 8  # Number of attention heads
    num_layers: int = 4  # Number of transformer layers
    dim_feedforward: int = 512  # Feedforward dimension
    dropout: float = 0.1  # Dropout rate

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [64, 128, 256]


class PositionalEncoding(nn.Module):
    """
       Positional encoding for transformer to capture temporal information (patterns) .
       Adds positional information to input embeddings so the transformer knows the order of the sequence.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Create positional encoding matrix
        position_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        # Apply sine to even indices
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        position_encoding[:, 1::2] = torch.cos(position * div_term)

        position_encoding = position_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('position_encoding', position_encoding)


    def forward(self, x):
        """
        This forward method adds positional encodings to input embeddings during the forward pass
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Tensor with positional encoding added
        """

        return x + self.position_encoding[:x.size(1), :].squeeze(1) # Add positional encoding (in the registered buffer) to input


class CNNFeatureExtractor(nn.Module):
    """
       CNN module for extracting local temporal features from EEG signals.
       This captures short-term patterns like spikes, oscillations, etc.

       Input = raw EEG sequence (time-series, 23 channels × 256 time steps).
       CNN layers = extract local temporal features (spikes, oscillations, rhythms).
       Output = high-dimensional feature representation (output_channels) that goes into the next stage (e.g., Transformer).
    """

    def __init__(self, input_dim: int,
                 cnn_channels: List[int], # list of filters per conv layer
                 kernel_size: int = 3,
                 pool_size: int = 2, # max pooling size (downsampling)
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.cnn_channels = cnn_channels

        # Build CNN layers
        layers = []
        in_channels = 1 # Single input channel (raw EEG)

        # Build CNN layers in a loop
        for i, out_channels in enumerate(cnn_channels):

            # Convolutional layer
            layers.append(nn.Conv1d( # 1D conv for time-series data
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size= kernel_size,
                padding = kernel_size // 2 # Same padding
            ))

            layers.append(nn.BatchNorm1d(out_channels)) # Normalization
            layers.append(nn.ReLU()) # Non-linearity
            layers.append(nn.MaxPool1d(kernel_size=pool_size)) # Downsampling
            layers.append(nn.Dropout(dropout)) # Regularization

            in_channels = out_channels # next layer’s input channels = previous layer’s output channels.

        # Stack all layers into a sequential model
        self.cnn = nn.Sequential(*layers) # Unpack layers list, each layer becomes a separate argument, without * will treat the entire list as a single argument
        self.output_channels = cnn_channels[-1] # Final output channel
        self.pool_size = pool_size
        self.num_pools = len(cnn_channels)

    def forward(self, x):
        """
           Forward pass through CNN feature extractor
           Args:
               x: Input tensor of shape (batch_size, input_dim, sequence_length)
                  e.g., (32, 23, 256) for 32 samples, 23 EEG channels, 256 time steps
           Returns:
               Extracted features of shape (batch_size, output_channels, reduced_length)
        """

        # Reshape: (batch_size, 23, 256) -> (batch_size, 1, 23*256)
        batch_size, input_dim, seq_len = x.shape
        x = x.view(batch_size, 1, input_dim * seq_len)

        x = self.cnn(x)

        # Reshape back to (batch_size, seq_len, features) for transformer
        x = x.transpose(1, 2)  # (batch_size, features, seq_len) -> (batch_size, seq_len, features)

        return x

    def get_output_seq_len(self, input_seq_len: int) -> int:
        """Calculate output sequence length after pooling"""

        seq_len = input_seq_len

        for _ in range(self.num_pools):
            seq_len = seq_len // self.pool_size
        return seq_len

class TransformerEncoder(nn.Module):
    """
       Transformer encoder for capturing long-range temporal dependencies and relationships.
       Processes sequential data by capturing relationships between distant time points in a sequence simultaneously.
       For our project, captures global patterns like seizure evolution and spread.

       Key Components:
         - Multi-head self-attention: captures relationships between all time points.
         - Feedforward networks: process attention outputs and applies non-linearity.
         - Positional encoding: adds temporal order information to the input embeddings.
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first = True,
            norm_first = False
        )

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers,
            # norm = nn.LayerNorm(d_model)
        )

        self.d_model = d_model


    def forward(self, x):
        """
           Forward pass through CNN feature extractor
           Args:
               x: Input tensor of shape (batch_size, input_dim, sequence_length)
                  e.g., (32, 23, 256) for 32 samples, 23 EEG channels, 256 time steps
           Returns:
               Extracted features of shape (batch_size, output_channels, reduced_length)
        """

        # x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)

        return x


class CNNTransformer(nn.Module):
    """
        Hybrid CNN-Transformer architecture for EEG seizure detection.

        Architecture:
        1. CNN: Extracts local temporal features (spikes, oscillations) - local pattern recognition
        2. Projection: Maps CNN features to transformer dimension
        3. Transformer: Captures long-range dependencies (seizure evolution) - global temporal understanding
        4. Global pooling: Aggregates sequence information
        5. Classification head: Binary classification (seizure vs non-seizure)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # CNN feature extractor - this CNN component will extract local temporal features from the raw EEG signals.
        self.cnn_extractor = CNNFeatureExtractor(
            input_dim=config.input_dim,
            cnn_channels=config.cnn_channels,
            kernel_size=config.kernel_size,
            pool_size=config.pool_size,
            dropout=config.dropout
        )

        # Calculate reduced CNN output dimensions
        self.output_dimension = config.cnn_channels[-1] # final CNN output channels (256)
        self.output_dimension_seq_len = self.cnn_extractor.get_output_seq_len(input_seq_len=config.sequence_length)

        # Projection layer (CNN output dim (features) -> Transformer d_model (dimension) - input to transformer)
        self.input_projection = nn.Linear(self.output_dimension, config.d_model)

        # Transformer encoder
        self.transformer_encoder = TransformerEncoder(
            d_model = config.d_model,
            nhead = config.nhead,
            num_layers = config.num_layers,
            dim_feedforward = config.dim_feedforward,
            dropout = config.dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2), # Reduce dimension (128 -> 64)
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_classes) # Final output layer (64 -> 2 classes (seizure/non-seizure))
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize model weights using Xavier uniform initialization for linear layers.
        """

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight) # keeps gradients stable during training for linear/tanh activations, also prevents vanishing gradients
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu') # Optimized for ReLU activations (kills negative values) because we use ReLU after each conv layer
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def get_attention_weights(self):
        """
        Extract attention weights for visualization.
        Useful for understanding which time periods the model focuses on.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            attention_weights: List of attention weight matrices
        """

        pass

    def forward(self, x):
        """
        Forward pass through CNN-Transformer hybrid.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
               - batch_size: Number of sequences in batch
               - seq_len: Length of each sequence (256 for 1 second at 256Hz)
               - input_dim: Number of EEG channels (23)

        Returns:
            logits: Output tensor of shape (batch_size, num_classes) - raw unnormalized scores for each class
        """

        batch_size, seq_len, input_dim = x.shape # (32, 256, 23) - this is the input shape

        # CNN feature extraction
        cnn_features = self.cnn_extractor(x) # output = (batch_size, reduced_seq_len, cnn_output_dim) - (32, 32, 256)

        # Project CNN features to transformer dimension
        transformer_input = self.input_projection(cnn_features) # (batch_size, reduced_seq_len, d_model) - (32, 32, 128)
        transformer_input = self.layer_norm(transformer_input) # Normalize inputs to transformer

        # Transformer encoding
        transformer_output = self.transformer_encoder(transformer_input)

        # Global average pooling over sequence length
        pooled_output = torch.mean(transformer_output, dim=1)

        # Classification head
        logits = self.classifier(pooled_output)

        return logits

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count total trainable parameters in the model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def predict(self, x):
        """
            Get class probabilities for logits using softmax.
            Convert raw scores to probabilities for each class.
        """

        logits = self.forward(x)
        return torch.softmax(logits, dim = -1)


    def save_checkpoint(self, path: str):
        """Save model state to file"""
        torch.save(self.state_dict(), path)


def main():
    print("Testing CNN-Transformer Model")
    print("=" * 60)

    # Device setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Configuration
    config = ModelConfig(
        input_dim=23,  # 23 EEG channels
        sequence_length=256,  # 1 second at 256Hz
        d_model=128,  # Transformer dimension
        nhead=8,  # 8 attention heads
        num_layers=4,  # 4 transformer layers
        num_classes=2,
        cnn_channels=[64, 128, 256],
        dropout=0.1
    )

    print(f"Configuration:")
    print(f"  Input: {config.input_dim} channels × {config.sequence_length} time steps")
    print(f"  CNN channels: {config.cnn_channels}")
    print(f"  Transformer: d_model={config.d_model}, layers={config.num_layers}, heads={config.nhead}")
    print()

    # Create model
    model = CNNTransformer(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / 1024 ** 2
    print(f"Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: ~{model_size_mb:.1f} MB")
    print()

    # Test forward pass
    batch_size = 8
    dummy_eeg = torch.randn(batch_size, config.sequence_length, config.input_dim).to(device)
    dummy_labels = torch.randint(0, 2, (batch_size,)).to(device)

    print(f"Testing forward pass...")
    print(f"  Input shape: {dummy_eeg.shape}")
    print(f"  Labels shape: {dummy_labels.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_eeg)
        probabilities = model.predict(dummy_eeg)
        predictions = torch.argmax(probabilities, dim=1)

    print(f"  Output shape: {output.shape}")
    print(f"  Output logits sample: {output[0].cpu().numpy()}")
    print(f"  Probabilities shape: {probabilities.shape}")
    print(f"  Sample probabilities: {probabilities[0].cpu().numpy()}")
    print(f"  Predictions: {predictions.cpu().numpy()}")

    # Architecture breakdown
    print()
    print("=" * 60)
    print("Architecture Breakdown:")
    print("=" * 60)

    cnn_params = sum(p.numel() for p in model.cnn_extractor.parameters())
    transformer_params = sum(p.numel() for p in model.transformer_encoder.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    print(f"CNN Feature Extractor: {cnn_params:,} parameters ({cnn_params / total_params * 100:.1f}%)")
    print(f"Transformer Encoder: {transformer_params:,} parameters ({transformer_params / total_params * 100:.1f}%)")
    print(f"Classification Head: {classifier_params:,} parameters ({classifier_params / total_params * 100:.1f}%)")

    # Calculate reduced sequence length
    reduced_seq_len = model.cnn_output_seq_len
    print()
    print(f"Sequence Length After CNN: {config.sequence_length} → {reduced_seq_len}")
    print(f"  Reduction factor: {config.sequence_length / reduced_seq_len:.1f}x")
    print(f"  This means: CNN reduces computational cost for transformer!")

    print("\n" + "=" * 60)
    print("✓ Model test completed successfully!")


if __name__ == "__main__":
    main()