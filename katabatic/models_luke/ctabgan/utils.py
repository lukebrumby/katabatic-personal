"""Utility helpers for CTAB-GAN — column type detection and metadata I/O."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import pandas as pd


def detect_column_types(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str], Dict[str, List[float]]]:
    """Auto-detect categorical, integer, and mixed columns from a DataFrame.

    The last column is treated as the target and is always evaluated for
    categorical classification independently of its dtype.

    Returns:
        categorical_columns: column names treated as categorical (label-encoded).
        integer_columns: column names whose generated values are rounded to int.
        mixed_columns: maps column name -> list of modal/special values (e.g. [0.0]).
    """
    categorical_columns: List[str] = []
    integer_columns: List[str] = []
    mixed_columns: Dict[str, List[float]] = {}

    for col in df.columns[:-1]:
        dtype = df[col].dtype
        n_unique = df[col].nunique()

        if dtype == "object" or (dtype in ["int64", "int32"] and n_unique < 20):
            categorical_columns.append(col)
        elif dtype in ["int64", "int32"]:
            integer_columns.append(col)
        elif dtype in ["float64", "float32"]:
            zero_ratio = (df[col] == 0).sum() / len(df)
            if zero_ratio > 0.3:
                mixed_columns[col] = [0.0]
                if col not in integer_columns:
                    integer_columns.append(col)

    target_col = df.columns[-1]
    if df[target_col].dtype == "object" or df[target_col].nunique() < 20:
        if target_col not in categorical_columns:
            categorical_columns.append(target_col)

    return categorical_columns, integer_columns, mixed_columns


def save_metadata(path: str, meta: dict) -> None:
    """Write a metadata dict to a JSON file, creating parent dirs as needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
