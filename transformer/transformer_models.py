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

        return x + self.position_encoding[:x.size(0), :] # Add positional encoding (in the registered buffer) to input


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

        x = x.transpose(1, 2) # Change shape to (batch_size, 1, input_dim, sequence_length) for Conv1d

        x = self.cnn(x) # Apply CNN layers

        x = x.transpose(1, 2)

        return x

    def get_output_seq_len(self, input_seq_len: int) -> int:
        """Calculate output sequence length after pooling"""

        seq_len = input_seq_len

        for _ in range(self.num_pools):
            seq_len = seq_len // self.pool_size
        return seq_len

class TransformerEncoder(nn.Module):
    """
       Transformer encoder for capturing long-range temporal dependencies.
       This captures global patterns like seizure evolution and spread.
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()

        # Positional encoding