"""
CopulaGAN Model Implementation for Katabatic Pipeline
Uses SDV library's CopulaGAN synthesizer

CopulaGAN combines Gaussian Copula transformations with CTGAN to better capture
multivariate correlations in tabular data.

Paper: The Synthetic Data Vault (Patki et al., 2016, IEEE DSAA)
CTGAN: Modeling Tabular Data using Conditional GAN (Xu et al., NeurIPS 2019)
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model as BaseModel
from .utils import detect_discrete_columns, save_metadata

warnings.filterwarnings("ignore")


class CopulaGANModel(BaseModel):
    """
    CopulaGAN: Combines copulas with GANs for better correlation modeling.

    Applies Gaussian Copula transformations to numerical columns before passing
    them to a CTGAN-style GAN, then inverse-transforms the output.

    Parameters match the SDV CopulaGANSynthesizer API.
    """

    def __init__(
        self,
        *,
        epochs: int = 300,
        batch_size: int = 500,
        embedding_dim: int = 128,
        generator_dim: tuple = (256, 256),
        discriminator_dim: tuple = (256, 256),
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        discriminator_steps: int = 1,
        log_frequency: bool = True,
        verbose: bool = False,
        pac: int = 10,
        cuda: bool = True,
        numerical_distributions: Optional[dict] = None,
        default_distribution: str = "beta",
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.pac = pac
        self.cuda = cuda
        self.numerical_distributions = numerical_distributions or {}
        self.default_distribution = default_distribution

        self.synthesizer = None
        self.discrete_columns: list = []
        self.column_names: Optional[list] = None
        self.n_train: int = 0

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["sdv", "copulas"]

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "CopulaGANModel":
        """Train the CopulaGAN model on data in data_dir, write synth output to synthetic_dir."""

        try:
            from sdv.single_table import CopulaGANSynthesizer
            from sdv.metadata import SingleTableMetadata
        except ImportError:
            raise ImportError(
                "SDV library not found. Install with: pip install sdv"
            )

        # Load training data
        train_full = os.path.join(data_dir, "train_full.csv")
        x_path = os.path.join(data_dir, "x_train.csv")
        y_path = os.path.join(data_dir, "y_train.csv")

        if os.path.exists(train_full):
            df = pd.read_csv(train_full)
        elif os.path.exists(x_path) and os.path.exists(y_path):
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path)
            if y.shape[1] != 1:
                raise ValueError("y_train.csv must have exactly one column.")
            df = pd.concat([X, y[y.columns[0]]], axis=1)
        else:
            raise FileNotFoundError(
                f"No training data found in {data_dir!r}. "
                "Expected train_full.csv or x_train.csv + y_train.csv."
            )

        self.column_names = df.columns.tolist()
        self.n_train = len(df)
        label_col = df.columns[-1]

        # Detect discrete columns
        self.discrete_columns = detect_discrete_columns(df)
        print(f"[CopulaGAN] Detected discrete columns: {self.discrete_columns}")

        # Build SDV metadata
        print("[CopulaGAN] Creating metadata...")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        # Resolve GPU availability
        try:
            import torch as _torch
            _gpu_available = _torch.cuda.is_available()
        except ImportError:
            _gpu_available = False

        print(f"[CopulaGAN] Initializing CopulaGAN with {self.epochs} epochs...")
        self.synthesizer = CopulaGANSynthesizer(
            metadata=metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            embedding_dim=self.embedding_dim,
            generator_dim=list(self.generator_dim),
            discriminator_dim=list(self.discriminator_dim),
            generator_lr=self.generator_lr,
            discriminator_lr=self.discriminator_lr,
            discriminator_steps=self.discriminator_steps,
            log_frequency=self.log_frequency,
            verbose=self.verbose,
            epochs=self.epochs,
            batch_size=self.batch_size,
            pac=self.pac,
            cuda=self.cuda and _gpu_available,
            numerical_distributions=self.numerical_distributions,
            default_distribution=self.default_distribution,
        )

        print(f"[CopulaGAN] Training on {len(df)} samples...")
        start_time = time.time()
        self.synthesizer.fit(df)
        elapsed = time.time() - start_time
        print(f"[CopulaGAN] Finished training in {elapsed:.2f} seconds.")

        self.is_fitted = True

        # Resolve output directory
        if not synthetic_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synthetic_dir = os.path.join("synthetic", dataset_name, "copulagan")
        os.makedirs(synthetic_dir, exist_ok=True)

        print(f"[CopulaGAN] Generating {len(df)} synthetic samples...")
        df_synth = self.sample(n=len(df))

        # Split into X and y
        x_synth = df_synth[df.columns[:-1]].copy()
        y_synth = df_synth[[label_col]].copy()

        # Align column names to real x_train.csv
        try:
            real_cols = pd.read_csv(x_path, nrows=0).columns.tolist()
            if len(real_cols) == x_synth.shape[1]:
                x_synth.columns = real_cols
                x_synth = x_synth.reindex(columns=real_cols)
        except Exception:
            pass

        # Write CSV outputs
        x_out = os.path.join(synthetic_dir, "x_synth.csv")
        y_out = os.path.join(synthetic_dir, "y_synth.csv")
        x_synth.to_csv(x_out, index=False)
        y_synth.to_csv(y_out, index=False, header=True)

        # Write metadata
        training_config = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "embedding_dim": self.embedding_dim,
            "generator_dim": list(self.generator_dim),
            "discriminator_dim": list(self.discriminator_dim),
            "default_distribution": self.default_distribution,
        }
        save_metadata(
            filepath=os.path.join(synthetic_dir, "metadata.json"),
            df=df,
            label_col=label_col,
            discrete_columns=self.discrete_columns,
            training_config=training_config,
        )

        print(
            f"[CopulaGAN] Synthetic data saved:\n  X -> {x_out}\n  y -> {y_out}"
        )
        return self

    def sample(
        self,
        n: Optional[int] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate synthetic samples. Returns a DataFrame."""
        if not self.is_fitted or self.synthesizer is None:
            raise RuntimeError("Call train() before sample().")

        n_samples = n if n is not None else (self.n_train or 1000)
        df_synth = self.synthesizer.sample(num_rows=n_samples)

        if self.column_names:
            available = [c for c in self.column_names if c in df_synth.columns]
            df_synth = df_synth[available]

        return df_synth

    def evaluate(self, *args, **kwargs) -> float:
        """Placeholder evaluate — TSTR is handled by TSTREvaluation."""
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return 0.0
