"""
Utility helpers for the CopulaGAN model implementation.
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd


def detect_discrete_columns(df: pd.DataFrame, nunique_threshold: int = 20) -> list:
    """Return column names that are categorical or have few unique values.

    A column is considered discrete if its dtype is object/string or if its
    number of unique values is <= nunique_threshold.
    """
    discrete = []
    for col in df.columns:
        if df[col].dtype == "object" or df[col].nunique() <= nunique_threshold:
            discrete.append(col)
    return discrete


def save_metadata(
    filepath: str,
    df: pd.DataFrame,
    label_col: str,
    discrete_columns: list,
    training_config: dict,
    extra: Optional[dict] = None,
) -> None:
    """Write model metadata to a JSON file.

    Args:
        filepath: Destination path for metadata.json.
        df: The training DataFrame (used for schema inference).
        label_col: Name of the label/target column.
        discrete_columns: List of columns detected as discrete.
        training_config: Dict of training hyperparameters.
        extra: Optional additional fields merged into the top-level dict.
    """
    meta = {
        "schema": {
            "columns": df.columns.tolist(),
            "label": label_col,
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "discrete_columns": discrete_columns,
            "n_rows": len(df),
        },
        "training": training_config,
    }
    if extra:
        meta.update(extra)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
