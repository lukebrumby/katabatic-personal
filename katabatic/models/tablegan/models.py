"""
Fixed TableGAN implementation for the Katabatic framework.

Key fixes:
1. Improved dimension handling for small datasets
2. Better generator/discriminator architecture scaling
3. Fixed padding and reshaping logic
4. Added proper error handling
5. Improved training stability
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple
from pathlib import Path

from katabatic.models.base_model import Model
from katabatic.models.tablegan.utils import (
    ConvGenerator,
    ConvDiscriminator,
    preprocess_data,
    postprocess_data,
    sample_noise,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TableGAN(Model):
    """
    TableGAN: Convolutional GAN for tabular data synthesis.

    Treats tabular data as 2D images and applies convolutional neural networks.
    """

    def __init__(
        self,
        # Architecture hyperparameters
        noise_dim: int = 100,
        generator_filters: Tuple[int, ...] = (256, 128, 64),
        discriminator_filters: Tuple[int, ...] = (64, 128, 256),
        # Training hyperparameters
        epochs: int = 300,
        batch_size: int = 64,
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        # GAN training parameters
        n_critic: int = 5,
        lambda_gp: float = 10.0,
        use_gradient_penalty: bool = True,
        # Regularization
        dropout: float = 0.1,
        # Other
        random_state: int = 42,
        device: Optional[str] = None,
    ):
        super().__init__()

        # Hyperparameters
        self.noise_dim = noise_dim
        self.generator_filters = generator_filters
        self.discriminator_filters = discriminator_filters

        self.epochs = epochs
        self.batch_size = batch_size
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr

        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.use_gradient_penalty = use_gradient_penalty

        self.dropout = dropout
        self.random_state = random_state

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        # Models (will be initialized in train)
        self.generator = None
        self.discriminator = None

        # Data preprocessing info
        self.input_dim_ = None
        self.table_height_ = None
        self.table_width_ = None
        self.feature_names_ = None
        self.target_name_ = None
        self.preprocessor_ = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        """Return a list of required dependencies for this model."""
        return ["torch", "sklearn", "pandas", "numpy"]

    def train(self, dataset_dir: str, synthetic_dir: str = None, **kwargs):
        """
        Train TableGAN model following Katabatic framework.

        Args:
            dataset_dir: Directory containing x_train.csv and y_train.csv
            synthetic_dir: Directory to save synthetic data (optional)
            **kwargs: Additional arguments
        """
        logger.info("=" * 80)
        logger.info("Training TableGAN Model")
        logger.info("=" * 80)

        # Load training data
        x_train_path = os.path.join(dataset_dir, "x_train.csv")
        y_train_path = os.path.join(dataset_dir, "y_train.csv")

        X_train = pd.read_csv(x_train_path)
        logger.info(f"Loaded training data: {X_train.shape}")

        # Combine with y_train if exists
        if os.path.exists(y_train_path):
            y_train = pd.read_csv(y_train_path)
            self.target_name_ = y_train.columns[0]
            df_train = pd.concat([X_train, y_train], axis=1)
        else:
            df_train = X_train
            self.target_name_ = None

        self.feature_names_ = X_train.columns.tolist()

        # Preprocess data
        data_normalized, self.preprocessor_ = preprocess_data(df_train)
        self.input_dim_ = data_normalized.shape[1]

        # Compute table dimensions
        self.table_height_, self.table_width_ = self._compute_table_dimensions(
            self.input_dim_
        )
        logger.info(
            f"Table dimensions: {self.table_height_}x{self.table_width_} "
            f"(padded from {self.input_dim_} features)"
        )

        # Pad data if necessary
        target_size = self.table_height_ * self.table_width_
        if self.input_dim_ < target_size:
            padding = np.zeros((len(data_normalized), target_size - self.input_dim_))
            data_normalized = np.hstack([data_normalized, padding])

        # Reshape to 2D (batch, channel, height, width)
        data_2d = data_normalized.reshape(-1, 1, self.table_height_, self.table_width_)

        # Train the model
        self._fit(data_2d)

        # Generate synthetic data
        logger.info(f"\nGenerating {len(df_train)} synthetic samples...")
        synth_data = self.sample(len(df_train))

        # Determine synthetic directory
        if synthetic_dir is None:
            dataset_name = os.path.basename(os.path.normpath(dataset_dir))
            synthetic_dir = os.path.join("synthetic", dataset_name, "tablegan")

        os.makedirs(synthetic_dir, exist_ok=True)

        # Save synthetic data
        if self.target_name_ is not None:
            # Split back into X and y
            x_synth = pd.DataFrame(synth_data[:, :-1], columns=self.feature_names_)
            y_synth = pd.DataFrame(synth_data[:, -1:], columns=[self.target_name_])

            x_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
            y_synth.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)
        else:
            synth_df = pd.DataFrame(synth_data, columns=df_train.columns)
            synth_df.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)

        logger.info(f"\nSynthetic data saved to: {synthetic_dir}")
        logger.info("Training complete!")

        self.is_fitted = True
        return self

    def _compute_table_dimensions(self, n_features: int) -> Tuple[int, int]:
        """
        Compute table dimensions optimized for convolutional layers.
        Uses powers of 2 or multiples of 4 for better conv performance.
        """
        # Find smallest square that fits all features
        side = int(np.ceil(np.sqrt(n_features)))

        # Round up to appropriate size
        if side <= 4:
            side = 4
        elif side <= 8:
            side = 8
        elif side <= 16:
            side = 16
        elif side <= 32:
            side = 32
        else:
            # Round to next multiple of 4
            side = ((side + 3) // 4) * 4

        return side, side

    def _fit(self, data_2d: np.ndarray):
        """Internal fit method."""
        # Initialize models
        self.generator = ConvGenerator(
            noise_dim=self.noise_dim,
            output_height=self.table_height_,
            output_width=self.table_width_,
            filters=self.generator_filters,
        ).to(self.device)

        self.discriminator = ConvDiscriminator(
            input_height=self.table_height_,
            input_width=self.table_width_,
            filters=self.discriminator_filters,
            dropout=self.dropout,
        ).to(self.device)

        logger.info(
            f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}"
        )
        logger.info(
            f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}"
        )

        # Optimizers
        optimizer_g = optim.Adam(
            self.generator.parameters(), lr=self.generator_lr, betas=(0.5, 0.999)
        )
        optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.discriminator_lr,
            betas=(0.5, 0.999),
        )

        # Train GAN
        self._train_gan(data_2d, optimizer_g, optimizer_d)

    def _train_gan(self, data_2d: np.ndarray, optimizer_g, optimizer_d):
        """Train the GAN."""
        dataset = torch.tensor(data_2d, dtype=torch.float32)
        n_batches = (len(dataset) + self.batch_size - 1) // self.batch_size

        for epoch in range(self.epochs):
            self.generator.train()
            self.discriminator.train()

            d_loss_total = 0
            g_loss_total = 0

            # Shuffle data
            indices = torch.randperm(len(dataset))

            for i in range(n_batches):
                batch_idx = indices[i * self.batch_size : (i + 1) * self.batch_size]
                real_data = dataset[batch_idx].to(self.device)
                batch_len = len(real_data)

                # Train Discriminator
                for _ in range(self.n_critic):
                    optimizer_d.zero_grad()

                    # Real samples
                    d_real = self.discriminator(real_data)

                    # Fake samples
                    noise = sample_noise(batch_len, self.noise_dim, self.device)
                    fake_data = self.generator(noise).detach()
                    d_fake = self.discriminator(fake_data)

                    if self.use_gradient_penalty:
                        # WGAN-GP loss
                        d_loss = d_fake.mean() - d_real.mean()

                        # Gradient penalty
                        gp = self._gradient_penalty(real_data, fake_data)
                        d_loss = d_loss + self.lambda_gp * gp
                    else:
                        # Standard GAN loss
                        criterion = nn.BCEWithLogitsLoss()
                        real_labels = torch.ones(batch_len, 1, device=self.device)
                        fake_labels = torch.zeros(batch_len, 1, device=self.device)

                        d_loss_real = criterion(d_real, real_labels)
                        d_loss_fake = criterion(d_fake, fake_labels)
                        d_loss = d_loss_real + d_loss_fake

                    d_loss.backward()
                    optimizer_d.step()

                    d_loss_total += d_loss.item()

                # Train Generator
                optimizer_g.zero_grad()

                noise = sample_noise(batch_len, self.noise_dim, self.device)
                fake_data = self.generator(noise)
                d_fake = self.discriminator(fake_data)

                if self.use_gradient_penalty:
                    g_loss = -d_fake.mean()
                else:
                    criterion = nn.BCEWithLogitsLoss()
                    real_labels = torch.ones(batch_len, 1, device=self.device)
                    g_loss = criterion(d_fake, real_labels)

                g_loss.backward()
                optimizer_g.step()

                g_loss_total += g_loss.item()

            # Log progress
            if (epoch + 1) % 20 == 0 or epoch == 0:
                avg_d_loss = d_loss_total / (n_batches * self.n_critic)
                avg_g_loss = g_loss_total / n_batches
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs}: "
                    f"D Loss = {avg_d_loss:.6f}, G Loss = {avg_g_loss:.6f}"
                )

    def _gradient_penalty(
        self, real_data: torch.Tensor, fake_data: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real_data.size(0)

        # Random weight for interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        alpha = alpha.expand_as(real_data)

        # Interpolated samples
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)

        # Discriminator output
        d_interpolates = self.discriminator(interpolates)

        # Gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Flatten and compute penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def sample(self, n: int) -> np.ndarray:
        """
        Generate synthetic samples.

        Args:
            n: Number of samples to generate

        Returns:
            Synthetic data as numpy array
        """
        if self.generator is None or self.preprocessor_ is None:
            raise RuntimeError("Model must be trained before sampling")

        self.generator.eval()

        with torch.no_grad():
            all_samples = []
            remaining = n

            while remaining > 0:
                batch_size = min(self.batch_size, remaining)
                noise = sample_noise(batch_size, self.noise_dim, self.device)
                fake_data = self.generator(noise)

                # Reshape back to flat
                fake_data_flat = fake_data.view(batch_size, -1).cpu().numpy()

                # Remove padding
                fake_data_flat = fake_data_flat[:, : self.input_dim_]

                all_samples.append(fake_data_flat)
                remaining -= batch_size

            synthetic_data_normalized = np.vstack(all_samples)

        # Postprocess
        synthetic_data = postprocess_data(synthetic_data_normalized, self.preprocessor_)

        return synthetic_data

    def evaluate(self):
        """Evaluate is handled by the pipeline's TSTREvaluation."""
        pass
