"""
SMOTE Model Implementation for Katabatic Pipeline
Uses imbalanced-learn's SMOTE for synthetic oversampling
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


class SMOTEModel(BaseModel):
    """
    SMOTE: Synthetic Minority Over-sampling Technique.

    Simple k-nearest neighbors interpolation for data augmentation.
    Default parameters:
    - k_neighbors: 5 (number of neighbors)
    - sampling_strategy: 'auto' (balance classes)
    """

    def __init__(
        self,
        *,
        k_neighbors: int = 5,
        sampling_strategy: str = "auto",
        random_state: int = 42,
    ) -> None:
        super().__init__()

        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

        self.smote = None
        self.column_names = None
        self.X_train = None
        self.y_train = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["imbalanced-learn", "scikit-learn"]

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "SMOTEModel":
        """Train (fit) the SMOTE model."""

        # Import here to avoid issues if not installed
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise ImportError(
                "imbalanced-learn not found. Install with: pip install imbalanced-learn"
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

        # Store for sampling
        self.X_train = X_train
        self.y_train = y_train

        # Check minimum class size and adjust k_neighbors if needed
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        min_class_size = class_counts.min()

        # SMOTE needs k_neighbors < min_class_size
        adjusted_k = self.k_neighbors
        if min_class_size <= self.k_neighbors:
            adjusted_k = max(1, min_class_size - 1)
            print(f"[SMOTE] Warning: Smallest class has {min_class_size} samples.")
            print(
                f"[SMOTE] Adjusting k_neighbors from {self.k_neighbors} to {adjusted_k}"
            )

        # Initialize SMOTE
        print(f"[SMOTE] Initializing with k_neighbors={adjusted_k}...")
        self.smote = SMOTE(
            k_neighbors=adjusted_k,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )

        # SMOTE doesn't "train" - it just stores the data
        print(
            f"[SMOTE] Ready to generate samples from {len(X_train)} training samples..."
        )

        start_time = time.time()

        # Generate samples
        X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)

        end_time = time.time()
        print(f"[SMOTE] Generated samples in {end_time-start_time:.2f} seconds.")

        # Calculate how many synthetic samples were created
        n_synthetic = len(X_resampled) - len(X_train)

        self.is_fitted = True

        # Save synthetic data
        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "smote")
        os.makedirs(synth_dir, exist_ok=True)

        print(f"[SMOTE] Generated {n_synthetic} new synthetic samples...")
        print(
            f"[SMOTE] Returning {len(X_train)} total samples (original size with balanced classes)..."
        )

        # Sample to match original size but with balanced classes
        if len(X_resampled) > len(X_train):
            indices = np.random.choice(len(X_resampled), len(X_train), replace=False)
            X_final = X_resampled[indices]
            y_final = y_resampled[indices]
        else:
            X_final = X_resampled
            y_final = y_resampled

        # Convert to DataFrame
        df_synth = pd.DataFrame(
            np.column_stack([X_final, y_final]), columns=self.column_names
        )

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
                "k_neighbors": adjusted_k,
                "sampling_strategy": self.sampling_strategy,
                "n_original": len(X_train),
                "n_synthetic": n_synthetic,  # ✅ FIXED: Use the variable we calculated
                "n_returned": len(X_final),
            },
        }
        with open(os.path.join(synth_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(
            f"[SMOTE] Synthetic data saved:\n  X -> {x_path_out}\n  y -> {y_path_out}"
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
        if not self.is_fitted or self.smote is None:
            raise RuntimeError("Call train() before sample().")

        # SMOTE generates by resampling
        X_resampled, y_resampled = self.smote.fit_resample(self.X_train, self.y_train)

        # Extract only synthetic samples
        n_original = len(self.X_train)
        X_synth = X_resampled[n_original:]
        y_synth = y_resampled[n_original:]

        # If n specified, sample that many
        if n is not None and n < len(X_synth):
            indices = np.random.choice(len(X_synth), n, replace=False)
            X_synth = X_synth[indices]
            y_synth = y_synth[indices]

        # Convert to DataFrame
        df_synth = pd.DataFrame(
            np.column_stack([X_synth, y_synth]), columns=self.column_names
        )

        return df_synth
