"""
ForestFlow Model Implementation for Katabatic Pipeline
Uses ForestDiffusion library with flow-matching for fast tabular data generation
"""

from __future__ import annotations
from typing import Optional
import os
import json
import time
import warnings

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model as BaseModel

warnings.filterwarnings("ignore")


class ForestFlowModel(BaseModel):
    """
    ForestFlow: Flow-based XGBoost diffusion for tabular data.

    Fast and efficient model using XGBoost-based flow matching.
    Default parameters based on GitHub repo:
    - n_t: 50 (number of noise levels/sampling steps)
    - duplicate_K: 100 (number of noise per sample or epochs when using n_batch)
    - n_batch: 1 (data iterator batches for memory efficiency)
    - diffusion_type: 'flow' (flow-matching, faster than 'vp' diffusion)
    - n_jobs: -1 (use all CPUs)
    """

    def __init__(
        self,
        *,
        n_t: int = 50,
        duplicate_K: int = 100,
        n_batch: int = 1,
        diffusion_type: str = "flow",
        n_jobs: int = -1,
        max_depth: int = 7,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        super().__init__()

        self.n_t = n_t
        self.duplicate_K = duplicate_K
        self.n_batch = n_batch
        self.diffusion_type = diffusion_type
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.forest_model = None
        self.column_names = None
        self.cat_indexes = None
        self.int_indexes = None
        self.bin_indexes = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["ForestDiffusion", "xgboost", "scikit-learn"]

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "ForestFlowModel":
        """Train the ForestFlow model."""

        # Import here to avoid issues if not installed
        try:
            from ForestDiffusion import ForestDiffusionModel
        except ImportError:
            raise ImportError(
                "ForestDiffusion not found. Install with: pip install ForestDiffusion"
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

        # Separate X and y
        X_train = df.iloc[:, :-1].values
        y_train = df.iloc[:, -1].values

        # Detect categorical, integer, and binary columns
        self.cat_indexes = []
        self.int_indexes = []
        self.bin_indexes = []

        for i, col in enumerate(df.columns[:-1]):
            unique_vals = df[col].nunique()
            if unique_vals == 2:
                self.bin_indexes.append(i)
            elif df[col].dtype in ["int64", "int32"] and unique_vals < 20:
                self.int_indexes.append(i)
            elif df[col].dtype == "object" or (
                df[col].dtype in ["int64", "int32"] and unique_vals < 50
            ):
                self.cat_indexes.append(i)

        # Combine X and y for ForestDiffusion
        Xy = np.column_stack([X_train, y_train])

        print(
            f"[ForestFlow] Initializing with n_t={self.n_t}, duplicate_K={self.duplicate_K}..."
        )
        print(
            f"[ForestFlow] Using diffusion_type='{self.diffusion_type}' with n_batch={self.n_batch}"
        )
        print(
            f"[ForestFlow] Detected {len(self.cat_indexes)} categorical, {len(self.int_indexes)} integer, {len(self.bin_indexes)} binary columns"
        )

        start_time = time.time()

        # Initialize ForestFlow model
        self.forest_model = ForestDiffusionModel(
            Xy,
            n_t=self.n_t,
            duplicate_K=self.duplicate_K,
            n_batch=self.n_batch,
            bin_indexes=self.bin_indexes,
            cat_indexes=self.cat_indexes + [len(df.columns) - 1],  # Add label column
            int_indexes=self.int_indexes,
            diffusion_type=self.diffusion_type,
            n_jobs=self.n_jobs,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            seed=self.random_state,
        )

        end_time = time.time()
        print(f"[ForestFlow] Model trained in {end_time-start_time:.2f} seconds.")

        self.is_fitted = True

        # Save synthetic data
        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "forestflow")
        os.makedirs(synth_dir, exist_ok=True)

        # Generate synthetic samples
        print(f"[ForestFlow] Generating {len(X_train)} synthetic samples...")
        start_gen = time.time()
        Xy_synth = self.forest_model.generate(batch_size=len(X_train))
        end_gen = time.time()
        print(f"[ForestFlow] Generated samples in {end_gen-start_gen:.2f} seconds.")

        # Convert to DataFrame
        df_synth = pd.DataFrame(Xy_synth, columns=self.column_names)

        # Split into X and y
        label = df.columns[-1]
        x_synth = df_synth[df.columns[:-1]].copy()
        y_synth = df_synth[[label]].copy()

        # Save
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
            },
            "training": {
                "n_t": self.n_t,
                "duplicate_K": self.duplicate_K,
                "n_batch": self.n_batch,
                "diffusion_type": self.diffusion_type,
                "n_jobs": self.n_jobs,
                "n_samples": len(X_train),
                "cat_indexes": self.cat_indexes,
                "int_indexes": self.int_indexes,
                "bin_indexes": self.bin_indexes,
            },
        }
        with open(os.path.join(synth_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(
            f"[ForestFlow] Synthetic data saved:\n  X -> {x_path_out}\n  y -> {y_path_out}"
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
        if not self.is_fitted or self.forest_model is None:
            raise RuntimeError("Call train() before sample().")

        # Default to same size as training data if not specified
        if n is None:
            n = 1000

        Xy_synth = self.forest_model.generate(batch_size=n)
        df_synth = pd.DataFrame(Xy_synth, columns=self.column_names)

        return df_synth
