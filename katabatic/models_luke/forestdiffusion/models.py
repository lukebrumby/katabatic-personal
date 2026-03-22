"""
ForestDiffusion Model Implementation for Katabatic Pipeline
Based on: https://github.com/SamsungSAILMontreal/ForestDiffusion
Paper: Jolicoeur-Martineau et al., AISTATS 2024 (arXiv:2309.09968)
"""

from __future__ import annotations
from typing import Optional, List
import os
import json
import time
import warnings

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model as BaseModel
from .utils import detect_column_types, save_metadata

warnings.filterwarnings("ignore")


class ForestDiffusionModel(BaseModel):
    """
    ForestDiffusion: tabular data generation via diffusion and flow-based XGBoost models.

    Default parameters follow the ForestDiffusion paper (AISTATS 2024):
    - diffusion_type: 'flow' (conditional flow matching, recommended default)
    - n_t: 50 (diffusion timesteps)
    - model: 'xgboost'
    - duplicate_K: 100
    - seed: 666
    """

    def __init__(
        self,
        *,
        n_t: int = 50,
        model: str = "xgboost",
        diffusion_type: str = "flow",
        max_depth: int = 7,
        n_estimators: int = 100,
        eta: float = 0.3,
        tree_method: str = "hist",
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        subsample: float = 1.0,
        num_leaves: int = 31,
        duplicate_K: int = 100,
        bin_indexes: Optional[List[int]] = None,
        cat_indexes: Optional[List[int]] = None,
        int_indexes: Optional[List[int]] = None,
        remove_miss: bool = False,
        p_in_one: bool = True,
        true_min_max_values: Optional[List[List[float]]] = None,
        eps: float = 1e-3,
        beta_min: float = 0.1,
        beta_max: float = 8.0,
        n_z: int = 10,
        n_jobs: int = -1,
        n_batch: int = 1,
        gpu_hist: bool = False,
        seed: int = 666,
    ) -> None:
        super().__init__()

        self.n_t = n_t
        self.model = model
        self.diffusion_type = diffusion_type
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.eta = eta
        self.tree_method = tree_method
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.num_leaves = num_leaves
        self.duplicate_K = duplicate_K
        self.bin_indexes = bin_indexes if bin_indexes is not None else []
        self.cat_indexes = cat_indexes if cat_indexes is not None else []
        self.int_indexes = int_indexes if int_indexes is not None else []
        self.remove_miss = remove_miss
        self.p_in_one = p_in_one
        self.true_min_max_values = true_min_max_values
        self.eps = eps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_z = n_z
        self.n_jobs = n_jobs
        self.n_batch = n_batch
        self.gpu_hist = gpu_hist
        self.seed = seed

        self.forest_model = None
        self.column_names = None
        self.cat_mappings = {}
        self._target_is_cat = False

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["ForestDiffusion", "xgboost"]

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "ForestDiffusionModel":
        """Train the ForestDiffusion model."""

        try:
            from ForestDiffusion import ForestDiffusionModel as FDModel
        except ImportError:
            raise ImportError(
                "ForestDiffusion library not found. Install with: pip install ForestDiffusion"
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

        # Detect binary, categorical, and integer columns (features only)
        bin_idxs, cat_idxs, int_idxs, cat_mappings = detect_column_types(df)
        self.bin_indexes = bin_idxs
        self.cat_indexes = cat_idxs
        self.int_indexes = int_idxs
        self.cat_mappings = cat_mappings

        print(f"[ForestDiffusion] Detected column types:")
        print(f"  Binary: {self.bin_indexes}")
        print(f"  Categorical: {self.cat_indexes}")
        print(f"  Integer: {self.int_indexes}")

        # Convert to numpy; factorize object-typed columns
        data = df.values.copy().astype(float, errors="ignore")
        for idx, uniques in cat_mappings.items():
            codes = pd.Categorical(df.iloc[:, idx], categories=uniques).codes
            data[:, idx] = codes

        # Separate X and y
        X_data = data[:, :-1].astype(float)
        y_data = data[:, -1]

        self._target_is_cat = (
            df.iloc[:, -1].dtype == "object" or df.iloc[:, -1].nunique() < 20
        )

        print(f"[ForestDiffusion] Initializing with {self.n_t} timesteps, "
              f"diffusion_type='{self.diffusion_type}', model='{self.model}'...")
        print(f"[ForestDiffusion] duplicate_K={self.duplicate_K}, n_rows={len(df)}")

        start_time = time.time()

        if self._target_is_cat:
            self.forest_model = FDModel(
                X=X_data,
                label_y=y_data,
                n_t=self.n_t,
                model=self.model,
                diffusion_type=self.diffusion_type,
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                eta=self.eta,
                tree_method=self.tree_method,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                subsample=self.subsample,
                num_leaves=self.num_leaves,
                duplicate_K=self.duplicate_K,
                bin_indexes=self.bin_indexes,
                cat_indexes=self.cat_indexes,
                int_indexes=self.int_indexes,
                remove_miss=self.remove_miss,
                p_in_one=self.p_in_one,
                true_min_max_values=self.true_min_max_values,
                eps=self.eps,
                beta_min=self.beta_min,
                beta_max=self.beta_max,
                n_z=self.n_z,
                n_jobs=self.n_jobs,
                n_batch=self.n_batch,
                gpu_hist=self.gpu_hist,
                seed=self.seed,
            )
        else:
            # Treat target as continuous — pass full data array
            n_cols = len(df.columns)
            extra_bin = [n_cols - 1] if df.iloc[:, -1].nunique() == 2 else []
            extra_cat = (
                [n_cols - 1]
                if self._target_is_cat and df.iloc[:, -1].nunique() > 2
                else []
            )
            self.forest_model = FDModel(
                X=data.astype(float),
                n_t=self.n_t,
                model=self.model,
                diffusion_type=self.diffusion_type,
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                eta=self.eta,
                tree_method=self.tree_method,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                subsample=self.subsample,
                num_leaves=self.num_leaves,
                duplicate_K=self.duplicate_K,
                bin_indexes=self.bin_indexes + extra_bin,
                cat_indexes=self.cat_indexes + extra_cat,
                int_indexes=self.int_indexes,
                remove_miss=self.remove_miss,
                p_in_one=self.p_in_one,
                true_min_max_values=self.true_min_max_values,
                eps=self.eps,
                beta_min=self.beta_min,
                beta_max=self.beta_max,
                n_z=self.n_z,
                n_jobs=self.n_jobs,
                n_batch=self.n_batch,
                gpu_hist=self.gpu_hist,
                seed=self.seed,
            )

        elapsed = time.time() - start_time
        print(f"[ForestDiffusion] Training finished in {elapsed:.2f}s.")

        self.is_fitted = True

        # Resolve synthetic_dir
        if not synthetic_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synthetic_dir = os.path.join("synthetic", dataset_name, "forestdiffusion")
        os.makedirs(synthetic_dir, exist_ok=True)

        print(f"[ForestDiffusion] Generating {len(df)} synthetic samples...")
        df_synth = self.sample(n=len(df))

        label = df.columns[-1]
        x_synth = df_synth[df.columns[:-1]].copy()
        y_synth = df_synth[[label]].copy()

        # Align column names to real x_train.csv
        try:
            real_cols = pd.read_csv(x_path, nrows=0).columns.tolist()
            if len(real_cols) == x_synth.shape[1]:
                x_synth.columns = real_cols
                x_synth = x_synth.reindex(columns=real_cols)
        except Exception:
            pass

        x_path_out = os.path.join(synthetic_dir, "x_synth.csv")
        y_path_out = os.path.join(synthetic_dir, "y_synth.csv")
        x_synth.to_csv(x_path_out, index=False)
        y_synth.to_csv(y_path_out, index=False, header=True)

        save_metadata(
            synthetic_dir=synthetic_dir,
            df=df,
            label=label,
            bin_indexes=self.bin_indexes,
            cat_indexes=self.cat_indexes,
            int_indexes=self.int_indexes,
            params={
                "n_t": self.n_t,
                "model": self.model,
                "diffusion_type": self.diffusion_type,
                "n_estimators": self.n_estimators,
                "duplicate_K": self.duplicate_K,
                "seed": self.seed,
            },
        )

        print(
            f"[ForestDiffusion] Synthetic data saved:\n"
            f"  X -> {x_path_out}\n  y -> {y_path_out}"
        )
        return self

    def sample(
        self,
        n: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate synthetic samples from the trained ForestDiffusion model."""
        if not self.is_fitted or self.forest_model is None:
            raise RuntimeError("Call train() before sample().")

        n_samples = n if n is not None else 1000

        samples = self.forest_model.generate(batch_size=n_samples)

        df_synth = pd.DataFrame(samples, columns=self.column_names)

        # Reverse factorization for object-typed categorical columns
        for idx, uniques in self.cat_mappings.items():
            col_name = self.column_names[idx]
            codes = df_synth[col_name].round().astype(int).clip(0, len(uniques) - 1)
            df_synth[col_name] = uniques[codes.values]

        return df_synth

    def evaluate(self, *args, **kwargs) -> float:
        """Evaluate the model (stub — TSTR evaluation handled by TSTREvaluation)."""
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return 0.0
