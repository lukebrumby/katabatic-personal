"""
TVAE Model Implementation for Katabatic Pipeline
Uses CTGAN library's TVAE synthesizer
"""

from __future__ import annotations
from typing import Optional
import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch

from katabatic.models.base_model import Model as BaseModel

warnings.filterwarnings("ignore")


class TVAEModel(BaseModel):
    """
    TVAE: Tabular Variational AutoEncoder.

    Default parameters from CTGAN library:
    - epochs: 300
    - batch_size: 500
    - compress_dims: (128, 128)
    - decompress_dims: (128, 128)
    - embedding_dim: 128
    - l2scale: 1e-5
    - loss_factor: 2
    """

    def __init__(
        self,
        *,
        epochs: int = 300,
        batch_size: int = 500,
        compress_dims: tuple = (128, 128),
        decompress_dims: tuple = (128, 128),
        embedding_dim: int = 128,
        l2scale: float = 1e-5,
        loss_factor: int = 2,
        cuda: bool = True,
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.batch_size = batch_size
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.embedding_dim = embedding_dim
        self.l2scale = l2scale
        self.loss_factor = loss_factor
        self.cuda = cuda

        self.synthesizer = None
        self.discrete_columns = []
        self.column_names = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["ctgan", "torch"]

    def _detect_discrete_columns(self, df: pd.DataFrame) -> list:
        """Detect categorical/discrete columns."""
        discrete = []
        for col in df.columns:
            if df[col].dtype == "object" or df[col].nunique() < 20:
                discrete.append(col)
        return discrete

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "TVAEModel":
        """Train the TVAE model."""

        # Import here to avoid issues if ctgan not installed
        try:
            from ctgan import TVAE
        except ImportError:
            raise ImportError(
                "CTGAN library not found. Install with: pip install ctgan"
            )

        # Load training data
        train_full = os.path.join(data_dir, "train_full.csv")
        x_path = os.path.join(data_dir, "x_train.csv")
        y_path = os.path.join(data_dir, "y_train.csv")

        if os.path.exists(train_full):
            df = pd.read_csv(train_full)
        else:
            if not (os.path.exists(x_path) and os.path.exists(y_path)):
                raise FileNotFoundError(f"Could not find training data in {data_dir}.")
            X = pd.read_csv(x_path)
            y = pd.read_csv(y_path)
            if y.shape[1] != 1:
                raise ValueError("y_train.csv must have exactly one column.")
            y_col = y.columns[0]
            df = pd.concat([X, y[y_col]], axis=1)

        self.column_names = df.columns.tolist()

        # Detect discrete columns
        self.discrete_columns = self._detect_discrete_columns(df)
        print(f"[TVAE] Detected discrete columns: {self.discrete_columns}")

        # Initialize synthesizer
        print(f"[TVAE] Initializing TVAE with {self.epochs} epochs...")
        self.synthesizer = TVAE(
            epochs=self.epochs,
            batch_size=self.batch_size,
            compress_dims=list(self.compress_dims),
            decompress_dims=list(self.decompress_dims),
            embedding_dim=self.embedding_dim,
            l2scale=self.l2scale,
            loss_factor=self.loss_factor,
            cuda=self.cuda and torch.cuda.is_available(),
        )

        # Train
        print(f"[TVAE] Training on {len(df)} samples...")
        start_time = time.time()
        self.synthesizer.fit(df, discrete_columns=self.discrete_columns)
        end_time = time.time()
        print(f"[TVAE] Finished training in {end_time-start_time:.2f} seconds.")

        self.is_fitted = True

        # Generate and save synthetic data
        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "tvae")
        os.makedirs(synth_dir, exist_ok=True)

        print(f"[TVAE] Generating {len(df)} synthetic samples...")
        df_synth = self.sample(n=len(df))

        # Split into X and y
        label = df.columns[-1]
        x_synth = df_synth[df.columns[:-1]].copy()
        y_synth = df_synth[[label]].copy()

        # Align column names with real data
        real_x_train_path = os.path.join(data_dir, "x_train.csv")
        try:
            real_cols = pd.read_csv(real_x_train_path, nrows=0).columns.tolist()
            if len(real_cols) == x_synth.shape[1]:
                x_synth.columns = real_cols
                x_synth = x_synth.reindex(columns=real_cols)
        except Exception:
            pass

        # Save synthetic data
        x_path_out = os.path.join(synth_dir, "x_synth.csv")
        y_path_out = os.path.join(synth_dir, "y_synth.csv")
        x_synth.to_csv(x_path_out, index=False)
        y_synth.to_csv(y_path_out, index=False, header=True)

        # Save metadata
        meta = {
            "schema": {
                "columns": df.columns.tolist(),
                "label": label,
                "dtypes": {c: str(df[c].dtype) for c in df.columns},
                "discrete_columns": self.discrete_columns,
            },
            "training": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "embedding_dim": self.embedding_dim,
            },
        }
        with open(os.path.join(synth_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[TVAE] Synthetic data saved:\n  X -> {x_path_out}\n  y -> {y_path_out}")
        return self

    def evaluate(self, *args, **kwargs) -> float:
        """Evaluate the model (placeholder for TSTR evaluation)."""
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return 0.0

    def sample(
        self,
        n: Optional[int] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate synthetic samples."""
        if not self.is_fitted or self.synthesizer is None:
            raise RuntimeError("Call train() before sample().")

        n_samples = n if n is not None else 1000

        # Generate samples
        df_synth = self.synthesizer.sample(n_samples)

        # Ensure column order matches original
        if self.column_names:
            df_synth = df_synth[self.column_names]

        return df_synth
