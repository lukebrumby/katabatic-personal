"""
CTAB-GAN Model Implementation for Katabatic Pipeline
Based on: https://github.com/Team-TUD/CTAB-GAN
"""

from __future__ import annotations
from typing import Any, Optional, Dict, List, Tuple
import os
import json
import warnings
import time

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model as BaseModel

warnings.filterwarnings("ignore")


class CTABGANModel(BaseModel):
    """
    CTAB-GAN: Conditional Tabular GAN with improved mixed-type handling.

    Default parameters match the GitHub repository (Experiment_Script_Adult.ipynb):
    - epochs: 150
    - batch_size: 500
    - class_dim: (256, 256, 256, 256)
    - random_dim: 100
    - num_channels: 64
    - l2scale: 1e-5
    """

    def __init__(
        self,
        *,
        epochs: int = 150,
        batch_size: int = 500,
        class_dim: Tuple[int, ...] = (256, 256, 256, 256),
        random_dim: int = 100,
        num_channels: int = 64,
        l2scale: float = 1e-5,
        test_ratio: float = 0.20,
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.batch_size = batch_size
        self.class_dim = class_dim
        self.random_dim = random_dim
        self.num_channels = num_channels
        self.l2scale = l2scale
        self.test_ratio = test_ratio

        self.synthesizer = None
        self.data_prep = None
        self.raw_df = None
        self.categorical_columns = []
        self.log_columns = []
        self.mixed_columns = {}
        self.integer_columns = []
        self.problem_type = {}

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["torch", "sklearn", "tqdm"]

    def _detect_column_types(self, df: pd.DataFrame) -> None:
        """Auto-detect column types from the data."""
        self.categorical_columns = []
        self.integer_columns = []
        self.mixed_columns = {}

        for col in df.columns[:-1]:  # Exclude last column (target)
            dtype = df[col].dtype
            n_unique = df[col].nunique()

            # Categorical: object type or low cardinality
            if dtype == "object" or (dtype in ["int64", "int32"] and n_unique < 20):
                self.categorical_columns.append(col)

            # Integer columns
            elif dtype in ["int64", "int32"]:
                self.integer_columns.append(col)

            # Mixed columns: continuous with many zeros
            elif dtype in ["float64", "float32"]:
                zero_ratio = (df[col] == 0).sum() / len(df)
                if zero_ratio > 0.3:
                    self.mixed_columns[col] = [0.0]
                    if col not in self.integer_columns:
                        self.integer_columns.append(col)

        # Add target column to categoricals if appropriate
        target_col = df.columns[-1]
        if df[target_col].dtype == "object" or df[target_col].nunique() < 20:
            if target_col not in self.categorical_columns:
                self.categorical_columns.append(target_col)

        print(f"[CTAB-GAN] Detected column types:")
        print(f"  Categorical: {self.categorical_columns}")
        print(f"  Integer: {self.integer_columns}")
        print(f"  Mixed: {list(self.mixed_columns.keys())}")

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "CTABGANModel":
        """Train the CTAB-GAN model."""

        from .ctabgan_synthesizer import CTABGANSynthesizer
        from .data_preparation import DataPrep

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

        self.raw_df = df.copy()
        self._detect_column_types(df)

        # Set problem type
        target_col = df.columns[-1]
        if target_col in self.categorical_columns:
            self.problem_type = {"Classification": target_col}

        # Initialize synthesizer
        self.synthesizer = CTABGANSynthesizer(
            class_dim=self.class_dim,
            random_dim=self.random_dim,
            num_channels=self.num_channels,
            l2scale=self.l2scale,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )

        # Prepare data
        print(f"[CTAB-GAN] Preparing data with {len(df)} samples...")
        self.data_prep = DataPrep(
            df=self.raw_df,
            categorical_columns=self.categorical_columns,
            log_columns=self.log_columns,
            mixed_columns=self.mixed_columns,
            integer_columns=self.integer_columns,
            problem_type=self.problem_type,
            test_ratio=self.test_ratio,
        )

        # Fit the synthesizer
        print(f"[CTAB-GAN] Training for {self.epochs} epochs...")
        start_time = time.time()
        self.synthesizer.fit(
            train_data=self.data_prep.df,
            categorical=self.data_prep.column_types["categorical"],
            mixed=self.data_prep.column_types["mixed"],
            type=self.problem_type,
        )
        end_time = time.time()
        print(f"[CTAB-GAN] Finished training in {end_time-start_time:.2f} seconds.")

        self.is_fitted = True

        # Generate and save synthetic data
        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "ctabgan")
        os.makedirs(synth_dir, exist_ok=True)

        print(f"[CTAB-GAN] Generating {len(df)} synthetic samples...")
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
                "categorical_columns": self.categorical_columns,
                "integer_columns": self.integer_columns,
                "mixed_columns": list(self.mixed_columns.keys()),
            },
            "training": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "random_dim": self.random_dim,
                "num_channels": self.num_channels,
            },
        }
        with open(os.path.join(synth_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(
            f"[CTAB-GAN] Synthetic data saved:\n  X -> {x_path_out}\n  y -> {y_path_out}"
        )
        return self

    def evaluate(self, *args, **kwargs) -> float:
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

        n_samples = n if n is not None else len(self.raw_df)
        sample = self.synthesizer.sample(n_samples)
        sample_df = self.data_prep.inverse_prep(sample)

        return sample_df
