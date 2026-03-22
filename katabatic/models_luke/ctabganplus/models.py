from __future__ import annotations

from typing import Optional
import os

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model as BaseModel
from .utils import (
    load_training_data,
    resolve_column_index,
    resolve_column_indices,
    infer_problem_type,
    save_metadata,
)


class CTABGANModel(BaseModel):
    """
    CTAB-GAN+ implementation for Katabatic.

    Conditional Tabular GAN with image-based architecture, Wasserstein loss with
    gradient penalty, and downstream classifier loss for improved ML utility.
    Reference: Zhao et al. (2023), https://arxiv.org/abs/2204.00401
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

        self.categorical = categorical or []
        self.log = log or []
        self.mixed = mixed or {}
        self.general = general or []
        self.non_categorical = non_categorical or []
        self.integer = integer or []
        self.problem_type = problem_type or {}

        self.synthesizer = None
        self.data_prep = None
        self._trained_columns = None
        self._label_col = None
        self._feature_cols = None

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
        import importlib
        torch = importlib.import_module("torch")

        from .utils import CTABGANSynthesizer, DataPrep

        np.random.seed(self.cfg["seed"])
        torch.manual_seed(self.cfg["seed"])

        # Load data — derives label_col from y_train.csv when available
        raw_df, label_col = load_training_data(data_dir)

        # Override with user-specified problem_type if provided
        if self.problem_type:
            problem_dict = dict(self.problem_type)
            label_col = list(problem_dict.values())[0]
        else:
            prob_type = infer_problem_type(raw_df, label_col)
            problem_dict = {prob_type: label_col}

        self._label_col = label_col
        self._trained_columns = raw_df.columns.tolist()
        self._feature_cols = [c for c in self._trained_columns if c != label_col]

        # Resolve column specs to integer indices
        # Auto-include label_col as categorical for classification problems
        user_categorical = list(self.categorical)
        prob_type_key = list(problem_dict.keys())[0]
        if prob_type_key == "Classification" and label_col not in user_categorical:
            user_categorical = user_categorical + [label_col]

        categorical_indices = resolve_column_indices(raw_df, user_categorical)
        log_indices = resolve_column_indices(raw_df, self.log)
        general_indices = resolve_column_indices(raw_df, self.general)
        non_categorical_indices = resolve_column_indices(raw_df, self.non_categorical)
        integer_indices = resolve_column_indices(raw_df, self.integer)

        # Convert mixed dict keys to indices
        mixed_dict = {}
        for key, value in self.mixed.items():
            idx = resolve_column_index(raw_df, key)
            if idx is not None:
                mixed_dict[idx] = value

        # DataPrep: pass type={} to skip internal train-test split.
        # The pipeline already provides the training subset via train_full.csv.
        # Column type annotations are passed separately.
        print("[CTAB-GAN+] Preparing data...")
        self.data_prep = DataPrep(
            raw_df=raw_df,
            categorical=user_categorical,
            log=self.log,
            mixed=self.mixed,
            general=self.general,
            non_categorical=self.non_categorical,
            integer=self.integer,
            type={},
            test_ratio=0.0,
        )

        train_data = self.data_prep.df

        # Resolve column indices on the prepared df (column order may differ)
        categorical_indices_prep = resolve_column_indices(train_data, user_categorical)
        log_indices_prep = resolve_column_indices(train_data, self.log)
        general_indices_prep = resolve_column_indices(train_data, self.general)
        non_categorical_indices_prep = resolve_column_indices(train_data, self.non_categorical)

        mixed_dict_prep = {}
        for key, value in self.mixed.items():
            idx = resolve_column_index(train_data, key)
            if idx is not None:
                mixed_dict_prep[idx] = value

        print(f"[CTAB-GAN+] Initializing synthesizer...")
        self.synthesizer = CTABGANSynthesizer(
            class_dim=tuple(self.cfg["class_dim"]),
            random_dim=self.cfg["random_dim"],
            num_channels=self.cfg["num_channels"],
            l2scale=self.cfg["l2scale"],
            batch_size=self.cfg["batch_size"],
            epochs=self.cfg["epochs"],
        )

        print(f"[CTAB-GAN+] Training GAN (epochs={self.cfg['epochs']})...")
        self.synthesizer.fit(
            train_data=train_data,
            categorical=self.data_prep.column_types["categorical"],
            mixed=self.data_prep.column_types["mixed"],
            general=self.data_prep.column_types["general"],
            non_categorical=self.data_prep.column_types["non_categorical"],
            type=problem_dict,
        )

        self.is_fitted = True

        # Set synthetic output directory
        synth_dir = synthetic_dir
        if not synth_dir:
            dataset_name = os.path.basename(os.path.normpath(data_dir)) or "dataset"
            synth_dir = os.path.join("synthetic", dataset_name, "ctabganplus")
        os.makedirs(synth_dir, exist_ok=True)

        print("[CTAB-GAN+] Generating synthetic data...")
        df_synth = self.sample(n=len(train_data))

        # Split features and label
        if label_col in df_synth.columns:
            x_synth = df_synth.drop(columns=[label_col])
            y_synth = df_synth[[label_col]]
        else:
            x_synth = df_synth.iloc[:, :-1]
            y_synth = df_synth.iloc[:, -1:]

        # Align x_synth columns to x_train.csv (or feature_cols if no x_train.csv)
        x_train_path = os.path.join(data_dir, "x_train.csv")
        if os.path.exists(x_train_path):
            x_train_cols = pd.read_csv(x_train_path, nrows=0).columns.tolist()
            x_synth = x_synth.reindex(columns=x_train_cols)
        else:
            x_synth = x_synth.reindex(columns=self._feature_cols)

        x_path_out = os.path.join(synth_dir, "x_synth.csv")
        y_path_out = os.path.join(synth_dir, "y_synth.csv")
        x_synth.to_csv(x_path_out, index=False)
        y_synth.to_csv(y_path_out, index=False)

        save_metadata(
            synth_dir=synth_dir,
            columns=df_synth.columns.tolist(),
            label_col=label_col,
            dtypes={c: str(df_synth[c].dtype) for c in df_synth.columns},
            categorical_indices=categorical_indices,
            training_config=self.cfg,
            data_prep_config={
                "categorical": categorical_indices,
                "log": log_indices,
                "mixed": {str(k): v for k, v in mixed_dict.items()},
                "general": general_indices,
                "non_categorical": non_categorical_indices,
                "integer": integer_indices,
                "problem_type": problem_dict,
            },
        )

        print("[CTAB-GAN+] Training complete!")
        print(f"  X -> {x_path_out}")
        print(f"  y -> {y_path_out}")

        return self

    def sample(
        self,
        n: Optional[int] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        if not self.is_fitted or self.synthesizer is None:
            raise RuntimeError("Call train() before sample().")

        n_samples = n if n is not None else 1000

        synthetic_data = self.synthesizer.sample(n_samples)
        df_synth = self.data_prep.inverse_prep(synthetic_data)

        if self._trained_columns is not None:
            available = [c for c in self._trained_columns if c in df_synth.columns]
            df_synth = df_synth[available]

        return df_synth

    def evaluate(self, *args, **kwargs) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return 0.0
