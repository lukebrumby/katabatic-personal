"""Utility helpers for the Gaussian Copula model."""

import json

import pandas as pd


def save_metadata(filepath: str, df: pd.DataFrame, default_distribution: str) -> None:
    """Save synthesis metadata to a JSON file at filepath."""
    label = df.columns[-1]
    meta = {
        "schema": {
            "columns": df.columns.tolist(),
            "label": label,
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
        },
        "training": {
            "default_distribution": default_distribution,
            "method": "Gaussian Copula (statistical)",
        },
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
