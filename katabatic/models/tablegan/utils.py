"""
Utility functions and neural network architectures for TableGAN.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ConvGenerator(nn.Module):
    """
    Convolutional Generator for TableGAN.
    Uses transposed convolutions to upsample from noise to table format.
    """

    def __init__(
        self,
        noise_dim: int = 100,
        output_height: int = 8,
        output_width: int = 8,
        filters: Tuple[int, ...] = (256, 128, 64),
    ):
        super().__init__()

        self.noise_dim = noise_dim
        self.output_height = output_height
        self.output_width = output_width

        # Calculate the initial size after linear projection
        # Start with smaller initial size for small outputs
        if max(output_height, output_width) <= 4:
            self.init_size = 2
        else:
            self.init_size = 4

        self.init_channels = filters[0]

        # Linear projection from noise to initial feature map
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, self.init_channels * self.init_size * self.init_size),
            nn.BatchNorm1d(self.init_channels * self.init_size * self.init_size),
            nn.ReLU(inplace=True),
        )

        # Build convolutional layers
        layers = []
        in_channels = self.init_channels

        # Calculate how many upsampling layers we need
        current_size = self.init_size
        n_layers_needed = 0
        while current_size < max(output_height, output_width):
            current_size *= 2
            n_layers_needed += 1

        # Adjust filters if we need fewer layers
        n_layers_needed = max(1, n_layers_needed)
        filters_to_use = (
            filters[:n_layers_needed] if len(filters) >= n_layers_needed else filters
        )

        # Build upsampling layers
        for i in range(len(filters_to_use) - 1):
            out_channels = filters_to_use[i + 1]
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels

        # Final layer to get to 1 channel output
        if current_size >= max(output_height, output_width):
            # Need one more upsampling
            layers.append(
                nn.ConvTranspose2d(
                    in_channels, 1, kernel_size=4, stride=2, padding=1, bias=False
                )
            )
        else:
            # Just convert channels without upsampling
            layers.append(
                nn.Conv2d(
                    in_channels, 1, kernel_size=3, stride=1, padding=1, bias=False
                )
            )

        layers.append(nn.Tanh())  # Output in range [-1, 1]

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: Noise tensor of shape (batch_size, noise_dim)

        Returns:
            Generated table of shape (batch_size, 1, height, width)
        """
        # Project and reshape
        x = self.fc(z)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)

        # Upsample through conv layers
        x = self.conv_layers(x)

        # Crop or pad to exact output size if needed
        if x.size(2) != self.output_height or x.size(3) != self.output_width:
            x = nn.functional.adaptive_avg_pool2d(
                x, (self.output_height, self.output_width)
            )

        return x


class ConvDiscriminator(nn.Module):
    """
    Convolutional Discriminator for TableGAN.
    Uses standard convolutions to classify real vs fake tables.
    Adaptively handles different input sizes.
    """

    def __init__(
        self,
        input_height: int = 8,
        input_width: int = 8,
        filters: Tuple[int, ...] = (64, 128, 256),
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width

        # Determine appropriate kernel size and number of layers based on input size
        min_dim = min(input_height, input_width)

        if min_dim <= 4:
            # Very small input - use smaller kernels and fewer layers
            kernel_size = 3
            stride = 1
            padding = 1
            max_layers = 2
        elif min_dim <= 8:
            # Small input - use moderate settings
            kernel_size = 3
            stride = 2
            padding = 1
            max_layers = 2
        else:
            # Larger input - can use standard settings
            kernel_size = 4
            stride = 2
            padding = 1
            max_layers = 3

        # Build convolutional layers
        layers = []
        in_channels = 1  # Single channel input (table)

        # Limit the number of layers based on input size
        filters_to_use = filters[:max_layers]

        for i, out_channels in enumerate(filters_to_use):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(dropout),
                ]
            )
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the size after all convolutions
        # Use adaptive pooling to ensure we get a consistent size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        # Final classification layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 4, 1),  # 2x2 = 4 after adaptive pooling
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input table of shape (batch_size, 1, height, width)

        Returns:
            Discriminator output (logits) of shape (batch_size, 1)
        """
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        return x


class DataPreprocessor:
    """
    Preprocessor for tabular data.
    Handles both numerical and categorical features.
    """

    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.column_types: Dict[str, str] = {}
        self.columns: list = []
        self.min_vals: np.ndarray = None
        self.max_vals: np.ndarray = None

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit preprocessor and transform data.

        Args:
            df: Input dataframe

        Returns:
            Normalized numpy array
        """
        self.columns = df.columns.tolist()
        data_list = []

        for col in df.columns:
            # Detect column type
            if df[col].dtype == "object" or df[col].nunique() < 10:
                # Categorical
                self.column_types[col] = "categorical"
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder
                data_list.append(encoded.reshape(-1, 1))
            else:
                # Numerical
                self.column_types[col] = "numerical"
                scaler = StandardScaler()
                scaled = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
                data_list.append(scaled)

        data = np.hstack(data_list)

        # Store min/max for final normalization to [-1, 1]
        self.min_vals = data.min(axis=0)
        self.max_vals = data.max(axis=0)

        # Normalize to [-1, 1]
        data_range = self.max_vals - self.min_vals
        data_range[data_range == 0] = 1  # Avoid division by zero
        data_normalized = 2 * (data - self.min_vals) / data_range - 1

        return data_normalized.astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        """
        Inverse transform normalized data back to original format.

        Args:
            data: Normalized numpy array

        Returns:
            Dataframe with original column types
        """
        # Denormalize from [-1, 1]
        data_range = self.max_vals - self.min_vals
        data_denorm = (data + 1) / 2 * data_range + self.min_vals

        result = {}
        col_idx = 0

        for col in self.columns:
            if self.column_types[col] == "categorical":
                # Round and clip to valid range
                encoded = np.round(data_denorm[:, col_idx]).astype(int)
                n_classes = len(self.encoders[col].classes_)
                encoded = np.clip(encoded, 0, n_classes - 1)

                # Decode
                decoded = self.encoders[col].inverse_transform(encoded)
                result[col] = decoded
                col_idx += 1
            else:
                # Inverse scale
                scaled = data_denorm[:, col_idx : col_idx + 1]
                original = self.scalers[col].inverse_transform(scaled).ravel()
                result[col] = original
                col_idx += 1

        return pd.DataFrame(result)


def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, DataPreprocessor]:
    """
    Preprocess tabular data for TableGAN.

    Args:
        df: Input dataframe

    Returns:
        Tuple of (normalized data, preprocessor)
    """
    preprocessor = DataPreprocessor()
    data_normalized = preprocessor.fit_transform(df)
    return data_normalized, preprocessor


def postprocess_data(data: np.ndarray, preprocessor: DataPreprocessor) -> np.ndarray:
    """
    Postprocess normalized data back to original format.

    Args:
        data: Normalized numpy array
        preprocessor: Fitted preprocessor

    Returns:
        Numpy array with original column types
    """
    df = preprocessor.inverse_transform(data)
    return df.values


def sample_noise(batch_size: int, noise_dim: int, device: torch.device) -> torch.Tensor:
    """
    Sample random noise for generator input.

    Args:
        batch_size: Number of samples
        noise_dim: Dimension of noise vector
        device: Device to create tensor on

    Returns:
        Noise tensor of shape (batch_size, noise_dim)
    """
    return torch.randn(batch_size, noise_dim, device=device)
