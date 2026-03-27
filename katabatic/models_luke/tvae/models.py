"""
TVAE Model Implementation for Katabatic Pipeline
Uses SDV library's TVAESynthesizer

TVAE (Tabular Variational Autoencoder) is a VAE-based generative model for tabular
data that learns a compressed latent representation and reconstructs realistic samples.

Paper: Modeling Tabular data using Conditional GAN (Xu et al., NeurIPS 2019)
GitHub: https://github.com/sdv-dev/CTGAN
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


class TVAEModel(BaseModel):
    """
    TVAE: Tabular Variational Autoencoder.

    Learns a latent representation of tabular data via a VAE encoder/decoder,
    handles mixed data types (continuous + categorical) using the same
    mode-specific normalisation as CTGAN.

    Parameters match the SDV TVAESynthesizer API.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        compress_dims: tuple = (128, 128),
        decompress_dims: tuple = (128, 128),
        l2scale: float = 1e-5,
        batch_size: int = 500,
        epochs: int = 300,
        loss_factor: int = 2,
        cuda: bool = True,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_factor = loss_factor
        self.cuda = cuda

        self.synthesizer = None
        self.discrete_columns: list = []
        self.column_names: Optional[list] = None
        self.n_train: int = 0

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["sdv"]

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "TVAEModel":
        """Train the TVAE model on data in data_dir, write synth output to synthetic_dir."""

        try:
            from sdv.single_table import TVAESynthesizer
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
        print(f"[TVAE] Detected discrete columns: {self.discrete_columns}")

        # Build SDV metadata
        print("[TVAE] Creating metadata...")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        # SDV auto-detects integer labels as 'numerical' (continuous), which causes
        # TVAE to generate out-of-range values after rounding. Force categorical so
        # only the original label values are ever generated.
        metadata.update_column(column_name=label_col, sdtype='categorical')

        # Resolve GPU availability
        try:
            import torch as _torch
            _gpu_available = _torch.cuda.is_available()
        except ImportError:
            _gpu_available = False

        print(f"[TVAE] Initializing TVAE with {self.epochs} epochs...")
        self.synthesizer = TVAESynthesizer(
            metadata=metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            embedding_dim=self.embedding_dim,
            compress_dims=list(self.compress_dims),
            decompress_dims=list(self.decompress_dims),
            l2scale=self.l2scale,
            batch_size=self.batch_size,
            epochs=self.epochs,
            loss_factor=self.loss_factor,
            cuda=self.cuda and _gpu_available,
        )

        print(f"[TVAE] Training on {len(df)} samples...")
        start_time = time.time()
        self.synthesizer.fit(df)
        elapsed = time.time() - start_time
        print(f"[TVAE] Finished training in {elapsed:.2f} seconds.")

        self.is_fitted = True

        # Resolve output directory
        if not synthetic_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synthetic_dir = os.path.join("synthetic", dataset_name, "tvae")
        os.makedirs(synthetic_dir, exist_ok=True)

        print(f"[TVAE] Generating {len(df)} synthetic samples...")
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
            "embedding_dim": self.embedding_dim,
            "compress_dims": list(self.compress_dims),
            "decompress_dims": list(self.decompress_dims),
            "l2scale": self.l2scale,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "loss_factor": self.loss_factor,
        }
        save_metadata(
            filepath=os.path.join(synthetic_dir, "metadata.json"),
            df=df,
            label_col=label_col,
            discrete_columns=self.discrete_columns,
            training_config=training_config,
        )

        print(
            f"[TVAE] Synthetic data saved:\n  X -> {x_out}\n  y -> {y_out}"
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
