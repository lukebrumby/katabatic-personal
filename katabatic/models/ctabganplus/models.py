from __future__ import annotations

from typing import Any, Optional, Dict, List
import os
import json
import importlib
import warnings

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model as BaseModel


def _try_import(module: str):
    try:
        return importlib.import_module(module)
    except Exception:
        return None


class CTABGANModel(BaseModel):
    """
    CTAB-GAN-Plus implementation for Katabatic.

    Conditional Tabular GAN with improved conditioning and image-based architecture.
    Uses Bayesian Gaussian Mixture Models for continuous features and conditional vectors.
    """

    def __init__(
        self,
        *,
        epochs: int = 150,
        batch_size: int = 500,
        random_dim: int = 100,
        num_channels: int = 64,
        class_dim: tuple = (256, 256, 256, 256),
        l2scale: float = 1e-5,
        categorical: list = None,
        log: list = None,
        mixed: dict = None,
        general: list = None,
        non_categorical: list = None,
        integer: list = None,
        problem_type: dict = None,
        seed: int = 42,
    ) -> None:
        """
        Initialize CTAB-GAN-Plus model.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            random_dim: Dimension of random noise vector
            num_channels: Number of channels in convolutional layers
            class_dim: Hidden dimensions for classifier network
            l2scale: L2 regularization scale
            categorical: List of categorical column indices/names
            log: List of log-transformed column indices/names
            mixed: Dictionary of mixed-type columns with modal values
            general: List of general columns (no GMM)
            non_categorical: List of non-categorical columns to round
            integer: List of integer columns
            problem_type: Dict with problem type and target column
            seed: Random seed
        """
        super().__init__()

        self.cfg = {
            "epochs": epochs,
            "batch_size": batch_size,
            "random_dim": random_dim,
            "num_channels": num_channels,
            "class_dim": list(class_dim),
            "l2scale": l2scale,
            "seed": seed,
        }

        # Data preparation parameters
        self.categorical = categorical or []
        self.log = log or []
        self.mixed = mixed or {}
        self.general = general or []
        self.non_categorical = non_categorical or []
        self.integer = integer or []
        self.problem_type = problem_type or {}

        # Model components (initialized during training)
        self.synthesizer = None
        self.data_prep = None
        self.transformer = None
        self._trained_columns = None

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["torch", "sklearn"]

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "CTABGANModel":
        """
        Train CTAB-GAN-Plus model.

        Args:
            data_dir: Directory containing training data
            synthetic_dir: Directory to save synthetic data (optional)
        """
        # Import required modules
        torch = _try_import("torch")
        if torch is None:
            raise ImportError(
                "torch is required for CTAB-GAN-Plus. Install with: pip install torch"
            )

        # Import CTAB-GAN components
        from .ctabgan_synthesizer import CTABGANSynthesizer
        from .data_preparation import DataPrep

        # Set random seed
        np.random.seed(self.cfg["seed"])
        torch.manual_seed(self.cfg["seed"])

        # Load data
        train_full_path = os.path.join(data_dir, "train_full.csv")
        x_train_path = os.path.join(data_dir, "x_train.csv")
        y_train_path = os.path.join(data_dir, "y_train.csv")

        if os.path.exists(train_full_path):
            raw_df = pd.read_csv(train_full_path)
        elif os.path.exists(x_train_path) and os.path.exists(y_train_path):
            X = pd.read_csv(x_train_path)
            y = pd.read_csv(y_train_path)
            if y.shape[1] != 1:
                raise ValueError(
                    "y_train.csv must have exactly one column (the target)."
                )
            raw_df = pd.concat([X, y], axis=1)
        else:
            raise FileNotFoundError(
                f"Could not find training data in {data_dir}. "
                f"Expected train_full.csv or x_train.csv/y_train.csv."
            )

        self._trained_columns = raw_df.columns.tolist()

        # Prepare column indices if names were provided
        categorical_indices = self._get_column_indices(raw_df, self.categorical)
        log_indices = self._get_column_indices(raw_df, self.log)
        general_indices = self._get_column_indices(raw_df, self.general)
        non_categorical_indices = self._get_column_indices(raw_df, self.non_categorical)
        integer_indices = self._get_column_indices(raw_df, self.integer)

        # Convert mixed dict keys to indices
        mixed_dict = {}
        for key, value in self.mixed.items():
            idx = self._get_column_index(raw_df, key)
            mixed_dict[idx] = value

        # Convert problem_type to use column index
        problem_dict = {}
        if self.problem_type:
            prob_type = list(self.problem_type.keys())[0]
            target_col = list(self.problem_type.values())[0]
            problem_dict[prob_type] = target_col

        # Data preparation
        print(f"[CTAB-GAN-Plus] Preparing data...")
        self.data_prep = DataPrep(
            raw_df=raw_df,
            categorical=categorical_indices,
            log=log_indices,
            mixed=mixed_dict,
            general=general_indices,
            non_categorical=non_categorical_indices,
            integer=integer_indices,
            type=problem_dict,
            test_ratio=0.20,  # Fixed test ratio for data prep
        )

        train_data = self.data_prep.df

        # Initialize synthesizer
        print(f"[CTAB-GAN-Plus] Initializing synthesizer...")
        self.synthesizer = CTABGANSynthesizer(
            class_dim=tuple(self.cfg["class_dim"]),
            random_dim=self.cfg["random_dim"],
            num_channels=self.cfg["num_channels"],
            l2scale=self.cfg["l2scale"],
            batch_size=self.cfg["batch_size"],
            epochs=self.cfg["epochs"],
        )

        # Fit synthesizer
        print(f"[CTAB-GAN-Plus] Training GAN (epochs={self.cfg['epochs']})...")
        self.synthesizer.fit(
            train_data=train_data,
            categorical=self.data_prep.column_types["categorical"],
            mixed=self.data_prep.column_types["mixed"],
            general=self.data_prep.column_types["general"],
            non_categorical=self.data_prep.column_types["non_categorical"],
            type=problem_dict,
        )

        self.is_fitted = True

        # Generate and save synthetic data
        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "ctabgan")
        os.makedirs(synth_dir, exist_ok=True)

        print(f"[CTAB-GAN-Plus] Generating synthetic data...")
        df_synth = self.sample(n=len(train_data))

        # Split into features and target
        if problem_dict:
            target_col = list(problem_dict.values())[0]
            label_col = (
                df_synth.columns[-1]
                if target_col == df_synth.columns[-1]
                else target_col
            )
            x_synth = df_synth.drop(columns=[label_col])
            y_synth = df_synth[[label_col]]
        else:
            # No target specified, use last column as default
            label_col = df_synth.columns[-1]
            x_synth = df_synth.iloc[:, :-1]
            y_synth = df_synth[[label_col]]

        # Save synthetic data
        x_path_out = os.path.join(synth_dir, "x_synth.csv")
        y_path_out = os.path.join(synth_dir, "y_synth.csv")
        x_synth.to_csv(x_path_out, index=False)
        y_synth.to_csv(y_path_out, index=False)

        # Save metadata
        meta = {
            "schema": {
                "columns": df_synth.columns.tolist(),
                "label": label_col,
                "dtypes": {c: str(df_synth[c].dtype) for c in df_synth.columns},
                "categorical_columns": [
                    df_synth.columns[i] for i in categorical_indices
                ],
            },
            "training": self.cfg,
            "data_preparation": {
                "categorical": categorical_indices,
                "log": log_indices,
                "mixed": mixed_dict,
                "general": general_indices,
                "non_categorical": non_categorical_indices,
                "integer": integer_indices,
                "problem_type": problem_dict,
            },
        }

        with open(os.path.join(synth_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[CTAB-GAN-Plus] Training complete!")
        print(f"[CTAB-GAN-Plus] Synthetic data saved:")
        print(f"  X -> {x_path_out}")
        print(f"  y -> {y_path_out}")

        return self

    def evaluate(self, *args, **kwargs) -> float:
        """Evaluate model performance."""
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return 0.0

    def sample(
        self,
        n: Optional[int] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate synthetic samples.

        Args:
            n: Number of samples to generate

        Returns:
            DataFrame of synthetic samples
        """
        if not self.is_fitted or self.synthesizer is None:
            raise RuntimeError("Call train() before sample().")

        n_samples = n if n is not None else 1000

        # Generate synthetic data using CTAB-GAN
        synthetic_data = self.synthesizer.sample(n_samples)

        # Inverse transform to original scale
        df_synth = self.data_prep.inverse_prep(synthetic_data)

        # Ensure column order matches training data
        if self._trained_columns is not None:
            df_synth = df_synth[self._trained_columns]

        return df_synth

    def _get_column_indices(self, df: pd.DataFrame, columns: list) -> list:
        """Convert column names to indices."""
        indices = []
        for col in columns:
            idx = self._get_column_index(df, col)
            if idx is not None:
                indices.append(idx)
        return indices

    def _get_column_index(self, df: pd.DataFrame, column) -> Optional[int]:
        """Get column index from name or return index if already int."""
        if isinstance(column, int):
            return column
        elif isinstance(column, str):
            try:
                return df.columns.get_loc(column)
            except KeyError:
                warnings.warn(f"Column '{column}' not found in dataframe")
                return None
        return None
