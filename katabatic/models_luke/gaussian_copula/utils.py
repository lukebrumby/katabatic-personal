"""
Utility helpers for the Gaussian Copula model.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import pandas as pd


def save_metadata(
    filepath: str,
    df: pd.DataFrame,
    default_distribution: str,
    numerical_distributions: Optional[dict] = None,
    random_state: Optional[int] = None,
) -> None:
    """Save training metadata to a JSON file."""
    metadata = {
        "n_samples": len(df),
        "columns": df.columns.tolist(),
        "default_distribution": default_distribution,
        "numerical_distributions": numerical_distributions or {},
        "random_state": random_state,
        "generated_at": datetime.utcnow().isoformat(),
    }
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)


def build_sdv_metadata(df: pd.DataFrame):
    """Create and auto-detect an SDV SingleTableMetadata object from a DataFrame."""
    try:
        from sdv.metadata import SingleTableMetadata
    except ImportError:
        raise ImportError("SDV library not found. Install with: pip install sdv")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    return metadata


def align_columns(x_synth: pd.DataFrame, x_train_path: str) -> pd.DataFrame:
    """Reindex synthetic feature DataFrame to match the column order of x_train.csv."""
    try:
        real_cols = pd.read_csv(x_train_path, nrows=0).columns.tolist()
        if len(real_cols) == x_synth.shape[1]:
            x_synth = x_synth.copy()
            x_synth.columns = real_cols
            return x_synth.reindex(columns=real_cols)
    except Exception:
        pass
    return x_synth
