"""
ForestDiffusion Model Implementation for Katabatic Pipeline
Based on: https://github.com/SamsungSAILMontreal/ForestDiffusion
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


class ForestDiffusionModel(BaseModel):
    """
    ForestDiffusion: Combines tree-based models with diffusion processes.

    Default parameters from ForestDiffusion paper:
    - n_t: 50 (number of diffusion timesteps)
    - model: 'xgboost' (tree model type)
    - diffusion_type: 'vp' (variance preserving)
    - max_depth: 7 (tree depth)
    - n_estimators: 100 (number of trees)
    - duplicate_K: 100 (data duplication for training)
    """

    def __init__(
        self,
        *,
        n_t: int = 50,
        model: str = "xgboost",
        diffusion_type: str = "vp",
        max_depth: int = 7,
        n_estimators: int = 100,
        eta: float = 0.3,
        duplicate_K: int = 100,
        num_leaves: int = 31,
        eps: float = 1e-3,
        beta_min: float = 0.1,
        beta_max: float = 8.0,
        n_jobs: int = -1,
        n_batch: int = 1,
        gpu_hist: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.n_t = n_t
        self.model = model
        self.diffusion_type = diffusion_type
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.eta = eta
        self.duplicate_K = duplicate_K
        self.num_leaves = num_leaves
        self.eps = eps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_jobs = n_jobs
        self.n_batch = n_batch
        self.gpu_hist = gpu_hist
        self.seed = seed

        self.forest_model = None
        self.bin_indexes = []
        self.cat_indexes = []
        self.int_indexes = []
        self.column_names = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["forest-diffusion", "xgboost", "scikit-learn"]

    def _detect_column_types(self, df: pd.DataFrame):
        """Detect binary, categorical, and integer columns."""
        self.bin_indexes = []
        self.cat_indexes = []
        self.int_indexes = []

        for idx, col in enumerate(df.columns[:-1]):  # Exclude target
            n_unique = df[col].nunique()
            dtype = df[col].dtype

            if dtype == "object":
                if n_unique == 2:
                    self.bin_indexes.append(idx)
                else:
                    self.cat_indexes.append(idx)
            elif n_unique == 2:
                self.bin_indexes.append(idx)
            elif n_unique < 20 and dtype in ["int64", "int32"]:
                self.cat_indexes.append(idx)
            elif dtype in ["int64", "int32"]:
                self.int_indexes.append(idx)

        print(f"[ForestDiffusion] Detected column types:")
        print(f"  Binary: {self.bin_indexes}")
        print(f"  Categorical: {self.cat_indexes}")
        print(f"  Integer: {self.int_indexes}")

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "ForestDiffusionModel":
        """Train the ForestDiffusion model."""

        # Import here to avoid issues if not installed
        try:
            from ForestDiffusion import ForestDiffusionModel as FDModel
        except ImportError:
            raise ImportError(
                "ForestDiffusion library not found. Install with: pip install forest-diffusion"
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

        # Detect column types
        self._detect_column_types(df)

        # Convert to numpy and handle categorical encoding
        data = df.values.copy()

        # Factorize categorical columns (ForestDiffusion requires numeric)
        self.cat_mappings = {}
        for idx in self.cat_indexes + self.bin_indexes:
            if df.iloc[:, idx].dtype == "object":
                codes, uniques = pd.factorize(df.iloc[:, idx])
                data[:, idx] = codes
                self.cat_mappings[idx] = uniques

        # Separate X and y
        X_data = data[:, :-1]
        y_data = data[:, -1]

        # Check if target is categorical
        target_is_cat = (
            df.iloc[:, -1].dtype == "object" or df.iloc[:, -1].nunique() < 20
        )

        print(f"[ForestDiffusion] Initializing with {self.n_t} timesteps...")
        print(
            f"[ForestDiffusion] Using {self.model} with duplicate_K={self.duplicate_K}"
        )

        start_time = time.time()

        # Initialize ForestDiffusion
        if target_is_cat:
            # Use label conditioning
            self.forest_model = FDModel(
                X=X_data,
                label_y=y_data,
                n_t=self.n_t,
                model=self.model,
                diffusion_type=self.diffusion_type,
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                eta=self.eta,
                num_leaves=self.num_leaves,
                duplicate_K=self.duplicate_K,
                bin_indexes=self.bin_indexes,
                cat_indexes=self.cat_indexes,
                int_indexes=self.int_indexes,
                eps=self.eps,
                beta_min=self.beta_min,
                beta_max=self.beta_max,
                n_jobs=self.n_jobs,
                n_batch=self.n_batch,
                gpu_hist=self.gpu_hist,
                seed=self.seed,
            )
        else:
            # No label conditioning
            self.forest_model = FDModel(
                X=data,
                n_t=self.n_t,
                model=self.model,
                diffusion_type=self.diffusion_type,
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                eta=self.eta,
                num_leaves=self.num_leaves,
                duplicate_K=self.duplicate_K,
                bin_indexes=(
                    self.bin_indexes + [len(df.columns) - 1]
                    if df.iloc[:, -1].nunique() == 2
                    else self.bin_indexes
                ),
                cat_indexes=(
                    self.cat_indexes + [len(df.columns) - 1]
                    if target_is_cat and df.iloc[:, -1].nunique() > 2
                    else self.cat_indexes
                ),
                int_indexes=(
                    self.int_indexes + [len(df.columns) - 1]
                    if not target_is_cat and df.iloc[:, -1].nunique() > 2
                    else self.int_indexes
                ),
                eps=self.eps,
                beta_min=self.beta_min,
                beta_max=self.beta_max,
                n_jobs=self.n_jobs,
                n_batch=self.n_batch,
                gpu_hist=self.gpu_hist,
                seed=self.seed,
            )

        end_time = time.time()
        print(
            f"[ForestDiffusion] Finished training in {end_time-start_time:.2f} seconds."
        )

        self.is_fitted = True

        # Generate and save synthetic data
        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "forestdiffusion")
        os.makedirs(synth_dir, exist_ok=True)

        print(f"[ForestDiffusion] Generating {len(df)} synthetic samples...")
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
                "bin_indexes": self.bin_indexes,
                "cat_indexes": self.cat_indexes,
                "int_indexes": self.int_indexes,
            },
            "training": {
                "n_t": self.n_t,
                "model": self.model,
                "diffusion_type": self.diffusion_type,
                "n_estimators": self.n_estimators,
                "duplicate_K": self.duplicate_K,
            },
        }
        with open(os.path.join(synth_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(
            f"[ForestDiffusion] Synthetic data saved:\n  X -> {x_path_out}\n  y -> {y_path_out}"
        )
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
        if not self.is_fitted or self.forest_model is None:
            raise RuntimeError("Call train() before sample().")

        n_samples = n if n is not None else 1000

        # Generate samples
        samples = self.forest_model.generate(batch_size=n_samples, n_t=self.n_t)

        # Convert back to DataFrame
        df_synth = pd.DataFrame(samples, columns=self.column_names)

        # Reverse factorization for categorical columns
        for idx, uniques in self.cat_mappings.items():
            col_name = self.column_names[idx]
            df_synth[col_name] = df_synth[col_name].round().astype(int)
            df_synth[col_name] = df_synth[col_name].clip(0, len(uniques) - 1)
            df_synth[col_name] = uniques[df_synth[col_name].values]

        return df_synth
