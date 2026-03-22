"""
Gaussian Copula Model Implementation for Katabatic Pipeline
Uses SDV's GaussianCopulaSynthesizer (statistical, no neural networks)
"""

from __future__ import annotations
from typing import Optional
import os
import time
import warnings

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model as BaseModel
from katabatic.models_luke.gaussian_copula.utils import save_metadata

warnings.filterwarnings("ignore")


class GaussianCopulaModel(BaseModel):
    """
    Gaussian Copula: Statistical method using copula theory.

    No neural networks - pure statistical modeling of correlations.
    Very fast and interpretable.
    """

    def __init__(
        self,
        *,
        default_distribution: str = "beta",
        numerical_distributions: dict = None,
        enforce_min_max_values: bool = True,
        enforce_rounding: bool = True,
    ) -> None:
        super().__init__()

        self.default_distribution = default_distribution
        self.numerical_distributions = numerical_distributions or {}
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding

        self.synthesizer = None
        self.column_names = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["sdv", "copulas"]

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "GaussianCopulaModel":
        """Train the Gaussian Copula model."""

        # Import here
        try:
            from sdv.single_table import GaussianCopulaSynthesizer
            from sdv.metadata import SingleTableMetadata
        except ImportError:
            raise ImportError("SDV library not found. Install with: pip install sdv")

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

        # Create SDV metadata
        print("[GaussianCopula] Creating metadata...")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        # Initialize synthesizer
        print(f"[GaussianCopula] Initializing Gaussian Copula...")
        self.synthesizer = GaussianCopulaSynthesizer(
            metadata=metadata,
            enforce_min_max_values=self.enforce_min_max_values,
            enforce_rounding=self.enforce_rounding,
            default_distribution=self.default_distribution,
            numerical_distributions=self.numerical_distributions,
        )

        # Fit (very fast - no iterative training)
        print(f"[GaussianCopula] Fitting to {len(df)} samples...")
        start_time = time.time()
        self.synthesizer.fit(df)
        end_time = time.time()
        print(
            f"[GaussianCopula] Finished fitting in {end_time-start_time:.2f} seconds."
        )

        self.is_fitted = True

        # Generate and save synthetic data
        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "gaussian_copula")
        os.makedirs(synth_dir, exist_ok=True)

        print(f"[GaussianCopula] Generating {len(df)} synthetic samples...")
        df_synth = self.sample(n=len(df))

        # Split into X and y
        label = df.columns[-1]
        x_synth = df_synth[df.columns[:-1]].copy()
        y_synth = df_synth[[label]].copy()

        # Align column names
        real_x_train_path = os.path.join(data_dir, "x_train.csv")
        try:
            real_cols = pd.read_csv(real_x_train_path, nrows=0).columns.tolist()
            if len(real_cols) == x_synth.shape[1]:
                x_synth.columns = real_cols
                x_synth = x_synth.reindex(columns=real_cols)
        except Exception:
            pass

        # Save
        x_path_out = os.path.join(synth_dir, "x_synth.csv")
        y_path_out = os.path.join(synth_dir, "y_synth.csv")
        x_synth.to_csv(x_path_out, index=False)
        y_synth.to_csv(y_path_out, index=False, header=True)

        # Save metadata
        save_metadata(
            os.path.join(synth_dir, "metadata.json"),
            df,
            self.default_distribution,
        )

        print(
            f"[GaussianCopula] Synthetic data saved:\n  X -> {x_path_out}\n  y -> {y_path_out}"
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

        n_samples = n if n is not None else 1000

        # Generate samples (very fast)
        df_synth = self.synthesizer.sample(num_rows=n_samples)

        # Ensure column order
        if self.column_names:
            df_synth = df_synth[self.column_names]

        return df_synth
