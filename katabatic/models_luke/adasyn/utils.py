"""
Helper utilities for the ADASYN model implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def adjust_n_neighbors(y_train: np.ndarray, n_neighbors: int) -> int:
    """Return adjusted n_neighbors that won't exceed smallest class size - 1."""
    _, class_counts = np.unique(y_train, return_counts=True)
    min_class_size = int(class_counts.min())
    if min_class_size <= n_neighbors:
        return max(1, min_class_size - 1)
    return n_neighbors


def build_resampled_df(
    X: np.ndarray,
    y: np.ndarray,
    column_names: list[str],
) -> pd.DataFrame:
    """Combine resampled X and y arrays into a DataFrame with original column names."""
    return pd.DataFrame(np.column_stack([X, y]), columns=column_names)
