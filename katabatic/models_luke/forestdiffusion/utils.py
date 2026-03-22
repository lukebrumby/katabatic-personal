"""
Utility helpers for ForestDiffusion Katabatic wrapper.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def detect_column_types(
    df: pd.DataFrame,
) -> Tuple[List[int], List[int], List[int], Dict[int, np.ndarray]]:
    """
    Detect binary, categorical, and integer column indices from a DataFrame.

    Only inspects feature columns (all columns except the last, which is the label).

    Returns
    -------
    bin_indexes : list of int
        Indices of binary columns (exactly 2 unique values).
    cat_indexes : list of int
        Indices of multi-category categorical columns (object dtype or int with < 20 unique).
    int_indexes : list of int
        Indices of integer-valued continuous columns (int dtype, >= 20 unique values).
    cat_mappings : dict of {int: np.ndarray}
        For each object-dtype column (binary or categorical), maps column index to
        the unique categories array used for factorization.
    """
    bin_indexes: List[int] = []
    cat_indexes: List[int] = []
    int_indexes: List[int] = []
    cat_mappings: Dict[int, np.ndarray] = {}

    feature_cols = df.columns[:-1]  # exclude label (last column)

    for idx, col in enumerate(feature_cols):
        n_unique = df[col].nunique()
        dtype = df[col].dtype

        if dtype == "object":
            uniques = df[col].unique()
            cat_mappings[idx] = uniques
            if n_unique == 2:
                bin_indexes.append(idx)
            else:
                cat_indexes.append(idx)
        elif n_unique == 2:
            bin_indexes.append(idx)
        elif n_unique < 20 and dtype in ("int64", "int32"):
            cat_indexes.append(idx)
        elif dtype in ("int64", "int32"):
            int_indexes.append(idx)

    return bin_indexes, cat_indexes, int_indexes, cat_mappings


def save_metadata(
    synthetic_dir: str,
    df: pd.DataFrame,
    label: str,
    bin_indexes: List[int],
    cat_indexes: List[int],
    int_indexes: List[int],
    params: dict,
) -> None:
    """
    Save model and schema metadata to metadata.json in synthetic_dir.

    Parameters
    ----------
    synthetic_dir : str
        Directory where metadata.json will be written.
    df : pd.DataFrame
        Training DataFrame (used to record schema).
    label : str
        Name of the label column.
    bin_indexes, cat_indexes, int_indexes : list of int
        Column type indices detected by detect_column_types().
    params : dict
        Training hyperparameters to record (e.g. n_t, model, diffusion_type).
    """
    meta = {
        "schema": {
            "columns": df.columns.tolist(),
            "label": label,
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "bin_indexes": bin_indexes,
            "cat_indexes": cat_indexes,
            "int_indexes": int_indexes,
        },
        "training": params,
    }
    path = os.path.join(synthetic_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
