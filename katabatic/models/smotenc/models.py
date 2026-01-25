"""
SMOTE-NC/N Model Implementation for Katabatic Pipeline
Uses imbalanced-learn's SMOTE-NC for mixed data or SMOTE-N for categorical data
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


class SMOTENCModel(BaseModel):
    def __init__(
        self,
        *,
        k_neighbors: int = 5,
        sampling_strategy: str = "auto",
        categorical_features: Optional[list] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.categorical_features = categorical_features
        self.random_state = random_state

        self.smote_engine = None  # Will hold either SMOTE-NC or SMOTE-N
        self.column_names = None
        self.X_train = None
        self.y_train = None
        self.categorical_indices = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["imbalanced-learn", "scikit-learn"]

    def _detect_categorical_features(self, df: pd.DataFrame) -> list:
        feature_df = df.iloc[:, :-1]
        categorical_indices = []
        for idx, col in enumerate(feature_df.columns):
            dtype = feature_df[col].dtype
            # Stricter detection: only object/category or non-numeric types
            if dtype in ["object", "category"] or not np.issubdtype(dtype, np.number):
                categorical_indices.append(idx)
        return categorical_indices

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "SMOTENCModel":
        try:
            from imblearn.over_sampling import SMOTENC, SMOTEN
        except ImportError:
            raise ImportError("pip install imbalanced-learn")

        # Data Loading logic
        train_full = os.path.join(data_dir, "train_full.csv")
        if os.path.exists(train_full):
            df = pd.read_csv(train_full)
        else:
            X = pd.read_csv(os.path.join(data_dir, "x_train.csv"))
            y = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
            df = pd.concat([X, y], axis=1)

        self.column_names = df.columns.tolist()
        self.X_train = df.iloc[:, :-1].values
        self.y_train = df.iloc[:, -1].values

        # Categorical detection
        if self.categorical_features is None:
            self.categorical_indices = self._detect_categorical_features(df)
        else:
            self.categorical_indices = self.categorical_features

        # Check class sizes for k_neighbors adjustment
        unique_classes, class_counts = np.unique(self.y_train, return_counts=True)
        min_class_size = class_counts.min()
        adjusted_k = self.k_neighbors
        if min_class_size <= self.k_neighbors:
            adjusted_k = max(1, min_class_size - 1)

        # ENGINE SELECTION: Fix for the "no numerical features" error
        n_features = self.X_train.shape[1]
        if len(self.categorical_indices) == n_features:
            print("[SMOTE] Purely categorical data detected. Using SMOTE-N.")
            self.smote_engine = SMOTEN(
                k_neighbors=adjusted_k,
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )
        else:
            print(
                f"[SMOTE] Mixed data detected ({len(self.categorical_indices)} categorical). Using SMOTE-NC."
            )
            self.smote_engine = SMOTENC(
                categorical_features=self.categorical_indices,
                k_neighbors=adjusted_k,
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )

        start_time = time.time()
        X_resampled, y_resampled = self.smote_engine.fit_resample(
            self.X_train, self.y_train
        )

        # Post-processing and saving (omitted for brevity, keep your original logic)
        self.is_fitted = True
        # ... (keep your existing CSV saving and metadata logic here) ...
        return self

    def sample(self, n: Optional[int] = None, *args, **kwargs) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call train() before sample().")
        X_resampled, y_resampled = self.smote_engine.fit_resample(
            self.X_train, self.y_train
        )
        n_original = len(self.X_train)
        X_synth, y_synth = X_resampled[n_original:], y_resampled[n_original:]

        if n is not None and n < len(X_synth):
            indices = np.random.choice(len(X_synth), n, replace=False)
            X_synth, y_synth = X_synth[indices], y_synth[indices]

        return pd.DataFrame(
            np.column_stack([X_synth, y_synth]), columns=self.column_names
        )
