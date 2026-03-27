"""
Utility helpers for TableGAN.

Covers data loading, min-max normalisation to [-1, 1], and metadata persistence.
No heavy dependencies (numpy/pandas/json only) — safe to import at module level.
"""

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class MinMaxNormalizer:
    """Normalise features to [-1, 1] using per-column min/max (tableGAN convention).

    Parameters
    ----------
    None

    Attributes
    ----------
    min_vals : np.ndarray
        Per-column minimum observed during fit.
    max_vals : np.ndarray
        Per-column maximum observed during fit.
    """

    def __init__(self) -> None:
        self.min_vals: np.ndarray | None = None
        self.max_vals: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxNormalizer":
        self.min_vals = X.min(axis=0)
        self.max_vals = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        range_ = self.max_vals - self.min_vals
        range_[range_ == 0] = 1.0  # avoid division by zero for constant columns
        return 2.0 * (X - self.min_vals) / range_ - 1.0

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        range_ = self.max_vals - self.min_vals
        range_[range_ == 0] = 1.0
        return (X_norm + 1.0) / 2.0 * range_ + self.min_vals


def load_training_data(
    data_dir: str,
) -> Tuple[pd.DataFrame, List[str], str]:
    """Load training data from a Katabatic data directory.

    Tries ``train_full.csv`` first, falls back to ``x_train.csv`` +
    ``y_train.csv``.  Assumes the label is the last column.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the split CSV files.

    Returns
    -------
    df : pd.DataFrame
        Combined feature + label DataFrame.
    x_columns : list[str]
        Feature column names (all columns except the last).
    label_col : str
        Name of the target/label column.

    Raises
    ------
    FileNotFoundError
        If neither ``train_full.csv`` nor the x/y pair is present.
    """
    train_full = os.path.join(data_dir, "train_full.csv")
    x_train = os.path.join(data_dir, "x_train.csv")
    y_train = os.path.join(data_dir, "y_train.csv")

    if os.path.exists(train_full):
        df = pd.read_csv(train_full)
        label_col = df.columns[-1]
        x_columns = df.columns[:-1].tolist()
    elif os.path.exists(x_train) and os.path.exists(y_train):
        x_df = pd.read_csv(x_train)
        y_df = pd.read_csv(y_train)
        label_col = y_df.columns[0]
        df = pd.concat([x_df, y_df], axis=1)
        x_columns = x_df.columns.tolist()
    else:
        raise FileNotFoundError(
            f"No training data found in '{data_dir}'. "
            "Expected 'train_full.csv' or both 'x_train.csv' and 'y_train.csv'."
        )

    return df, x_columns, label_col


def save_metadata(
    filepath: str,
    x_columns: List[str],
    label_col: str,
    n_classes: int,
    n_features: int,
    training_config: Dict[str, Any],
) -> None:
    """Persist training metadata to a JSON file.

    Parameters
    ----------
    filepath : str
        Destination path for the JSON file.
    x_columns : list[str]
        Feature column names.
    label_col : str
        Label column name.
    n_classes : int
        Number of distinct target classes.
    n_features : int
        Number of feature columns.
    training_config : dict
        Hyperparameter snapshot.
    """
    metadata: Dict[str, Any] = {
        "model_type": "TableGAN",
        "x_columns": x_columns,
        "label_col": label_col,
        "n_classes": n_classes,
        "n_features": n_features,
        "training_config": training_config,
    }
    with open(filepath, "w") as fh:
        json.dump(metadata, fh, indent=2)
