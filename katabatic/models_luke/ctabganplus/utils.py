"""CTAB-GAN+ utilities.

Consolidates:
  - Katabatic pipeline helpers (load_training_data, resolve_*, infer_*, save_metadata)
  - DataPrep        : raw-data preprocessing and inverse transform
  - DataTransformer : VGM / OHE column encoding (ported from transformer.py)
  - ImageTransformer: 2-D image reshape helper (ported from transformer.py)
  - Cond / Sampler  : conditional vector and data samplers
  - CTABGANSynthesizer : full GAN training loop (ported from ctabgan_synthesizer.py)

Ported from: https://github.com/Team-TUD/CTAB-GAN-Plus
Paper: Zhao et al. (2023) "CTAB-GAN+: Enhancing Tabular Data Synthesis"
       Frontiers in Big Data. https://arxiv.org/abs/2204.00401

Modifications vs. original:
  - DataPrep accepts string column names (converts to integer indices internally)
  - All torch / torch.nn imports are lazy (inside methods only)
  - tqdm removed; plain print used for epoch logging
  - CTABGANSynthesizer.sample() returns pd.DataFrame matching self.transformer columns
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.mixture import BayesianGaussianMixture


# ── Katabatic helpers ──────────────────────────────────────────────────────────

def load_training_data(data_dir: str):
    """Load training data from data_dir.

    Tries train_full.csv first; falls back to x_train.csv + y_train.csv.
    Returns (raw_df, label_col).
    """
    train_full = os.path.join(data_dir, "train_full.csv")
    x_train = os.path.join(data_dir, "x_train.csv")
    y_train = os.path.join(data_dir, "y_train.csv")

    if os.path.exists(train_full):
        df = pd.read_csv(train_full)
        label_col = (
            pd.read_csv(y_train, nrows=0).columns[0]
            if os.path.exists(y_train)
            else df.columns[-1]
        )
        return df, label_col

    if os.path.exists(x_train) and os.path.exists(y_train):
        X = pd.read_csv(x_train)
        y = pd.read_csv(y_train)
        if y.shape[1] != 1:
            raise ValueError("y_train.csv must have exactly one column.")
        label_col = y.columns[0]
        return pd.concat([X, y], axis=1), label_col

    raise FileNotFoundError(
        f"No training data found in {data_dir!r}. "
        "Expected train_full.csv or both x_train.csv and y_train.csv."
    )


def resolve_column_index(df: pd.DataFrame, col: Union[int, str]) -> Optional[int]:
    """Return integer column index. Returns None and warns if name not found."""
    if isinstance(col, int):
        return col
    if isinstance(col, str):
        try:
            return df.columns.get_loc(col)
        except KeyError:
            warnings.warn(f"Column {col!r} not found in dataframe; skipping.")
            return None
    return None


def resolve_column_indices(df: pd.DataFrame, cols: list) -> list:
    """Resolve a list of column names/indices to integer positions."""
    result = []
    for c in cols:
        idx = resolve_column_index(df, c)
        if idx is not None:
            result.append(idx)
    return result


def infer_problem_type(df: pd.DataFrame, label_col: str) -> str:
    """Return 'Classification' or 'Regression' based on label column."""
    col = df[label_col]
    if col.dtype == object or col.nunique() <= 20:
        return "Classification"
    return "Regression"


def save_metadata(
    synth_dir: str,
    columns: List[str],
    label_col: str,
    dtypes: Dict[str, str],
    categorical_indices: List[int],
    training_config: Dict[str, Any],
    data_prep_config: Dict[str, Any],
) -> None:
    """Write metadata.json to synth_dir."""
    meta = {
        "schema": {
            "columns": columns,
            "label": label_col,
            "dtypes": dtypes,
            "categorical_columns": [
                columns[i] for i in categorical_indices if i < len(columns)
            ],
        },
        "training": training_config,
        "data_preparation": data_prep_config,
    }
    with open(os.path.join(synth_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ── DataPrep ───────────────────────────────────────────────────────────────────

class DataPrep(object):
    """Preprocesses raw tabular data for CTABGANSynthesizer.

    Accepts column specifications as either string names OR integer indices.
    Internally converts everything to integer indices to match the DataTransformer
    interface used by CTABGANSynthesizer.

    Ported from:
      https://github.com/Team-TUD/CTAB-GAN-Plus/blob/main/model/pipeline/data_preparation.py
    """

    def __init__(
        self,
        raw_df: pd.DataFrame,
        categorical: list,
        log: list,
        mixed: dict,
        general: list,
        non_categorical: list,
        integer: list,
        type: dict,
        test_ratio: float,
    ):
        # ── Convert string column names to integer indices ──────────────────
        cols = list(raw_df.columns)

        def _to_idx(col):
            if isinstance(col, str):
                try:
                    return cols.index(col)
                except ValueError:
                    warnings.warn(f"Column {col!r} not found; skipping.")
                    return None
            return col

        categorical = [i for i in (_to_idx(c) for c in categorical) if i is not None]
        log = [i for i in (_to_idx(c) for c in log) if i is not None]
        general = [i for i in (_to_idx(c) for c in general) if i is not None]
        non_categorical = [i for i in (_to_idx(c) for c in non_categorical) if i is not None]
        integer = [i for i in (_to_idx(c) for c in integer) if i is not None]
        mixed = {
            _to_idx(k): v
            for k, v in mixed.items()
            if _to_idx(k) is not None
        }

        # ── Store specs ─────────────────────────────────────────────────────
        self.categorical_columns = categorical
        self.log_columns = log
        self.mixed_columns = mixed
        self.general_columns = general
        self.non_categorical_columns = non_categorical
        self.integer_columns = integer
        self.column_types = dict()
        self.column_types["categorical"] = []
        self.column_types["mixed"] = {}
        self.column_types["general"] = []
        self.column_types["non_categorical"] = []
        self.lower_bounds = {}
        self.label_encoder_list = []

        # ── Optional train/test split (skip when type={}) ───────────────────
        if type:
            problem = list(type.keys())[0]
            target_col = list(type.values())[0]
            if problem:
                y_real = raw_df[target_col]
                X_real = raw_df.drop(columns=[target_col])
                if problem == "Classification":
                    X_tr, _, y_tr, _ = model_selection.train_test_split(
                        X_real, y_real,
                        test_size=test_ratio, stratify=y_real, random_state=42,
                    )
                else:
                    X_tr, _, y_tr, _ = model_selection.train_test_split(
                        X_real, y_real, test_size=test_ratio, random_state=42,
                    )
                X_tr[target_col] = y_tr
                self.df = X_tr
            else:
                self.df = raw_df
        else:
            self.df = raw_df

        self.df = self.df.replace(r" ", np.nan)
        self.df = self.df.fillna("empty")

        # ── Handle missing values in numeric columns ─────────────────────────
        all_columns = set(self.df.columns)
        irrelevant_missing = set(self.df.columns[i] for i in self.categorical_columns)
        relevant_missing = list(all_columns - irrelevant_missing)

        for col_name in relevant_missing:
            col_idx = cols.index(col_name) if col_name in cols else None
            if col_idx is None:
                continue
            if col_idx in self.log_columns:
                if "empty" in list(self.df[col_name].values):
                    self.df[col_name] = self.df[col_name].apply(
                        lambda x: -9999999 if x == "empty" else x
                    )
                    self.mixed_columns[col_idx] = [-9999999]
            elif col_idx in self.mixed_columns:
                if "empty" in list(self.df[col_name].values):
                    self.df[col_name] = self.df[col_name].apply(
                        lambda x: -9999999 if x == "empty" else x
                    )
                    self.mixed_columns[col_idx].append(-9999999)
            else:
                if "empty" in list(self.df[col_name].values):
                    self.df[col_name] = self.df[col_name].apply(
                        lambda x: -9999999 if x == "empty" else x
                    )
                    self.mixed_columns[col_idx] = [-9999999]

        # ── Log-transform ────────────────────────────────────────────────────
        if self.log_columns:
            for log_col_idx in self.log_columns:
                col_name = self.df.columns[log_col_idx]
                valid_mask = self.df[col_name] != -9999999
                eps = 1
                lower = np.min(self.df[col_name][valid_mask].values)
                self.lower_bounds[col_name] = lower
                if lower > 0:
                    self.df[col_name] = self.df[col_name].apply(
                        lambda x: np.log(x) if x != -9999999 else -9999999
                    )
                elif lower == 0:
                    self.df[col_name] = self.df[col_name].apply(
                        lambda x: np.log(x + eps) if x != -9999999 else -9999999
                    )
                else:
                    self.df[col_name] = self.df[col_name].apply(
                        lambda x: np.log(x - lower + eps) if x != -9999999 else -9999999
                    )

        # ── Label-encode + track column types ────────────────────────────────
        for column_index, column in enumerate(self.df.columns):
            if column_index in self.categorical_columns:
                le = preprocessing.LabelEncoder()
                self.df[column] = self.df[column].astype(str)
                le.fit(self.df[column])
                entry = {"column": column, "label_encoder": le}
                self.df[column] = le.transform(self.df[column])
                self.label_encoder_list.append(entry)
                self.column_types["categorical"].append(column_index)

                if column_index in self.general_columns:
                    self.column_types["general"].append(column_index)
                if column_index in self.non_categorical_columns:
                    self.column_types["non_categorical"].append(column_index)

            elif column_index in self.mixed_columns:
                self.column_types["mixed"][column_index] = self.mixed_columns[column_index]

            elif column_index in self.general_columns:
                self.column_types["general"].append(column_index)

        super().__init__()

    def inverse_prep(self, data, eps=1):
        """Inverse-transform synthesized data back to original scale.

        Args:
            data: Numpy array (n, num_cols) OR DataFrame with columns matching self.df.
            eps:  Epsilon for inverse log-transform.

        Returns:
            DataFrame in original feature space.
        """
        df_sample = pd.DataFrame(data, columns=self.df.columns)

        # Inverse label-encode
        for entry in self.label_encoder_list:
            col = entry["column"]
            le = entry["label_encoder"]
            df_sample[col] = df_sample[col].astype(int)
            df_sample[col] = le.inverse_transform(df_sample[col])

        # Inverse log-transform
        if self.log_columns:
            for col in df_sample.columns:
                if col in self.lower_bounds:
                    lower = self.lower_bounds[col]
                    if lower > 0:
                        df_sample[col] = df_sample[col].apply(np.exp)
                    elif lower == 0:
                        df_sample[col] = df_sample[col].apply(
                            lambda x: np.ceil(np.exp(x) - eps)
                            if (np.exp(x) - eps) < 0
                            else (np.exp(x) - eps)
                        )
                    else:
                        df_sample[col] = df_sample[col].apply(
                            lambda x: np.exp(x) - eps + lower
                        )

        # Round integer columns
        if self.integer_columns:
            for col_idx in self.integer_columns:
                col = self.df.columns[col_idx]
                df_sample[col] = np.round(df_sample[col].values)
                df_sample[col] = df_sample[col].astype(int)

        df_sample.replace(-9999999, np.nan, inplace=True)
        df_sample.replace("empty", np.nan, inplace=True)

        return df_sample


# ── DataTransformer ────────────────────────────────────────────────────────────

class DataTransformer:
    """VGM + OHE encoding for CTABGANSynthesizer.

    Uses integer column indices throughout (produced by DataPrep).

    Ported from:
      https://github.com/Team-TUD/CTAB-GAN-Plus/blob/main/model/synthesizer/ctabgan_synthesizer.py
      (transformer.py in the Katabatic working copy)
    """

    def __init__(
        self,
        train_data: pd.DataFrame = pd.DataFrame(),
        categorical_list: list = (),
        mixed_dict: dict = None,
        general_list: list = (),
        non_categorical_list: list = (),
        n_clusters: int = 10,
        eps: float = 0.005,
    ):
        self.meta = None
        self.n_clusters = n_clusters
        self.eps = eps
        self.train_data = train_data
        self.categorical_columns = list(categorical_list)
        self.mixed_columns = mixed_dict if mixed_dict is not None else {}
        self.general_columns = list(general_list)
        self.non_categorical_columns = list(non_categorical_list)

    def get_metadata(self):
        meta = []
        for index in range(self.train_data.shape[1]):
            column = self.train_data.iloc[:, index]
            if index in self.categorical_columns:
                if index in self.non_categorical_columns:
                    meta.append({
                        "name": index,
                        "type": "continuous",
                        "min": column.min(),
                        "max": column.max(),
                    })
                else:
                    mapper = column.value_counts().index.tolist()
                    meta.append({
                        "name": index,
                        "type": "categorical",
                        "size": len(mapper),
                        "i2s": mapper,
                    })
            elif index in self.mixed_columns:
                meta.append({
                    "name": index,
                    "type": "mixed",
                    "min": column.min(),
                    "max": column.max(),
                    "modal": self.mixed_columns[index],
                })
            else:
                meta.append({
                    "name": index,
                    "type": "continuous",
                    "min": column.min(),
                    "max": column.max(),
                })
        return meta

    def fit(self):
        data = self.train_data.values
        self.meta = self.get_metadata()
        model = []
        self.ordering = []
        self.output_info = []
        self.output_dim = 0
        self.components = []
        self.filter_arr = []

        for id_, info in enumerate(self.meta):
            if info["type"] == "continuous":
                if id_ not in self.general_columns:
                    gm = BayesianGaussianMixture(
                        n_components=self.n_clusters,
                        weight_concentration_prior_type="dirichlet_process",
                        weight_concentration_prior=0.001,
                        max_iter=100, n_init=1, random_state=42,
                    )
                    gm.fit(data[:, id_].reshape([-1, 1]))
                    mode_freq = (
                        pd.Series(gm.predict(data[:, id_].reshape([-1, 1])))
                        .value_counts()
                        .index
                    )
                    model.append(gm)
                    old_comp = gm.weights_ > self.eps
                    comp = [
                        (i in mode_freq) and old_comp[i]
                        for i in range(self.n_clusters)
                    ]
                    self.components.append(comp)
                    self.output_info += [(1, "tanh", "no_g"), (int(np.sum(comp)), "softmax")]
                    self.output_dim += 1 + int(np.sum(comp))
                else:
                    model.append(None)
                    self.components.append(None)
                    self.output_info += [(1, "tanh", "yes_g")]
                    self.output_dim += 1

            elif info["type"] == "mixed":
                gm1 = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    max_iter=100, n_init=1, random_state=42,
                )
                gm2 = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    max_iter=100, n_init=1, random_state=42,
                )
                gm1.fit(data[:, id_].reshape([-1, 1]))
                filter_arr = [val not in info["modal"] for val in data[:, id_]]
                filtered = data[:, id_][filter_arr]
                if len(filtered) > 0:
                    gm2.fit(filtered.reshape([-1, 1]))
                    mode_freq = (
                        pd.Series(gm2.predict(filtered.reshape([-1, 1])))
                        .value_counts()
                        .index
                    )
                    old_comp = gm2.weights_ > self.eps
                    comp = [
                        (i in mode_freq) and old_comp[i]
                        for i in range(self.n_clusters)
                    ]
                else:
                    comp = [False] * self.n_clusters
                self.filter_arr.append(filter_arr)
                model.append((gm1, gm2))
                self.components.append(comp)
                self.output_info += [
                    (1, "tanh", "no_g"),
                    (int(np.sum(comp)) + len(info["modal"]), "softmax"),
                ]
                self.output_dim += 1 + int(np.sum(comp)) + len(info["modal"])

            else:  # categorical
                model.append(None)
                self.components.append(None)
                self.output_info += [(info["size"], "softmax")]
                self.output_dim += info["size"]

        self.model = model

    def transform(self, data, ispositive=False, positive_list=None):
        values = []
        mixed_counter = 0
        for id_, info in enumerate(self.meta):
            current = data[:, id_]
            if info["type"] == "continuous":
                if id_ not in self.general_columns:
                    current = current.reshape([-1, 1])
                    means = self.model[id_].means_.reshape((1, self.n_clusters))
                    stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))
                    if ispositive and id_ in (positive_list or []):
                        features = np.abs(current - means) / (4 * stds)
                    else:
                        features = (current - means) / (4 * stds)

                    probs = self.model[id_].predict_proba(current.reshape([-1, 1]))
                    n_opts = int(sum(self.components[id_]))
                    features = features[:, self.components[id_]]
                    probs = probs[:, self.components[id_]]

                    opt_sel = np.zeros(len(data), dtype="int")
                    for i in range(len(data)):
                        pp = probs[i] + 1e-6
                        pp /= pp.sum()
                        opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

                    idx = np.arange(len(features))
                    features = features[idx, opt_sel].reshape([-1, 1])
                    features = np.clip(features, -0.99, 0.99)
                    probs_onehot = np.zeros_like(probs)
                    probs_onehot[np.arange(len(probs)), opt_sel] = 1

                    re_ordered = np.zeros_like(probs_onehot)
                    col_sums = probs_onehot.sum(axis=0)
                    largest_indices = np.argsort(-col_sums)[: probs_onehot.shape[1]]
                    self.ordering.append(largest_indices)
                    for oid, val in enumerate(largest_indices):
                        re_ordered[:, oid] = probs_onehot[:, val]

                    values += [features, re_ordered]
                else:
                    self.ordering.append(None)
                    if id_ in self.non_categorical_columns:
                        info["min"] = -1e-3
                        info["max"] = info["max"] + 1e-3
                    current = (current - info["min"]) / (info["max"] - info["min"])
                    current = current * 2 - 1
                    current = current.reshape([-1, 1])
                    values.append(current)

            elif info["type"] == "mixed":
                means_0 = self.model[id_][0].means_.reshape([-1])
                stds_0 = np.sqrt(self.model[id_][0].covariances_).reshape([-1])

                zero_std_list, means_needed, stds_needed = [], [], []
                for mode in info["modal"]:
                    if mode != -9999999:
                        dist = [abs(mode - v) for v in means_0.flatten()]
                        zero_std_list.append(int(np.argmin(dist)))
                for idx in zero_std_list:
                    means_needed.append(means_0[idx])
                    stds_needed.append(stds_0[idx])

                mode_vals = [
                    np.abs(i - j) / (4 * k)
                    for i, j, k in zip(info["modal"], means_needed, stds_needed)
                    if i != -9999999
                ]
                if -9999999 in info["modal"]:
                    mode_vals.append(0)

                current = current.reshape([-1, 1])
                filter_arr = self.filter_arr[mixed_counter]
                current_f = current[filter_arr]

                means = self.model[id_][1].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_][1].covariances_).reshape((1, self.n_clusters))
                if len(current_f) > 0:
                    if ispositive and id_ in (positive_list or []):
                        features = np.abs(current_f - means) / (4 * stds)
                    else:
                        features = (current_f - means) / (4 * stds)
                    probs = self.model[id_][1].predict_proba(current_f.reshape([-1, 1]))
                    n_opts = int(sum(self.components[id_]))
                    features = features[:, self.components[id_]]
                    probs = probs[:, self.components[id_]]
                    opt_sel = np.zeros(len(current_f), dtype="int")
                    for i in range(len(current_f)):
                        pp = probs[i] + 1e-6
                        pp /= pp.sum()
                        opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)
                    idx = np.arange(len(features))
                    features = features[idx, opt_sel].reshape([-1, 1])
                    features = np.clip(features, -0.99, 0.99)
                    probs_onehot = np.zeros_like(probs)
                    probs_onehot[np.arange(len(probs)), opt_sel] = 1
                else:
                    n_comp = int(sum(self.components[id_]))
                    features = np.zeros((0, 1))
                    probs_onehot = np.zeros((0, n_comp))

                extra_bits = np.zeros([len(current_f), len(info["modal"])])
                temp_probs_onehot = np.concatenate([extra_bits, probs_onehot], axis=1)
                n_comp_total = probs_onehot.shape[1] if len(probs_onehot) > 0 else 0
                final = np.zeros(
                    [len(data), 1 + n_comp_total + len(info["modal"])]
                )
                features_cursor = 0
                for idx, val in enumerate(data[:, id_]):
                    if val in info["modal"]:
                        cat = list(info["modal"]).index(val)
                        final[idx, 0] = mode_vals[cat]
                        final[idx, cat + 1] = 1
                    else:
                        if features_cursor < len(features):
                            final[idx, 0] = features[features_cursor]
                            final[idx, 1 + len(info["modal"]):] = (
                                temp_probs_onehot[features_cursor][len(info["modal"]):]
                            )
                            features_cursor += 1

                just_onehot = final[:, 1:]
                re_ordered_jhot = np.zeros_like(just_onehot)
                n = just_onehot.shape[1]
                col_sums = just_onehot.sum(axis=0)
                largest_indices = np.argsort(-col_sums)[:n]
                self.ordering.append(largest_indices)
                for oid, val in enumerate(largest_indices):
                    re_ordered_jhot[:, oid] = just_onehot[:, val]
                final_features = final[:, 0].reshape([-1, 1])
                values += [final_features, re_ordered_jhot]
                mixed_counter += 1

            else:  # categorical
                self.ordering.append(None)
                col_t = np.zeros([len(data), info["size"]])
                idx = list(map(info["i2s"].index, current))
                col_t[np.arange(len(data)), idx] = 1
                values.append(col_t)

        return np.concatenate(values, axis=1)

    def inverse_transform(self, data):
        data_t = np.zeros([len(data), len(self.meta)])
        invalid_ids = []
        st = 0

        for id_, info in enumerate(self.meta):
            if info["type"] == "continuous":
                if id_ not in self.general_columns:
                    u = data[:, st]
                    n_comp = int(np.sum(self.components[id_]))
                    v = data[:, st + 1 : st + 1 + n_comp]
                    order = self.ordering[id_]
                    v_re = np.zeros_like(v)
                    for oid, val in enumerate(order):
                        v_re[:, val] = v[:, oid]
                    v = v_re
                    u = np.clip(u, -1, 1)
                    v_t = np.ones((len(data), self.n_clusters)) * -100
                    v_t[:, self.components[id_]] = v
                    v = v_t
                    st += 1 + n_comp
                    means = self.model[id_].means_.reshape([-1])
                    stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                    p_argmax = np.argmax(v, axis=1)
                    tmp = u * 4 * stds[p_argmax] + means[p_argmax]
                    for idx, val in enumerate(tmp):
                        if val < info["min"] or val > info["max"]:
                            invalid_ids.append(idx)
                    if id_ in self.non_categorical_columns:
                        tmp = np.round(tmp)
                    data_t[:, id_] = tmp
                else:
                    u = data[:, st]
                    u = np.clip((u + 1) / 2, 0, 1)
                    u = u * (info["max"] - info["min"]) + info["min"]
                    if id_ in self.non_categorical_columns:
                        data_t[:, id_] = np.round(u)
                    else:
                        data_t[:, id_] = u
                    st += 1

            elif info["type"] == "mixed":
                u = data[:, st]
                n_comp = int(np.sum(self.components[id_]))
                full_v = data[:, st + 1 : st + 1 + len(info["modal"]) + n_comp]
                order = self.ordering[id_]
                fv_re = np.zeros_like(full_v)
                for oid, val in enumerate(order):
                    fv_re[:, val] = full_v[:, oid]
                full_v = fv_re
                mixed_v = full_v[:, : len(info["modal"])]
                v = full_v[:, -n_comp:] if n_comp > 0 else np.zeros((len(data), 0))
                u = np.clip(u, -1, 1)
                v_t = np.ones((len(data), self.n_clusters)) * -100
                if n_comp > 0:
                    v_t[:, self.components[id_]] = v
                v = np.concatenate([mixed_v, v_t], axis=1)
                st += 1 + n_comp + len(info["modal"])
                means = self.model[id_][1].means_.reshape([-1])
                stds = np.sqrt(self.model[id_][1].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)
                result = np.zeros(len(data))
                for idx in range(len(data)):
                    if p_argmax[idx] < len(info["modal"]):
                        result[idx] = info["modal"][p_argmax[idx]]
                    else:
                        ci = p_argmax[idx] - len(info["modal"])
                        result[idx] = u[idx] * 4 * stds[ci] + means[ci]
                for idx, val in enumerate(result):
                    if val < info["min"] or val > info["max"]:
                        invalid_ids.append(idx)
                data_t[:, id_] = result

            else:  # categorical
                current = data[:, st : st + info["size"]]
                st += info["size"]
                idx = np.argmax(current, axis=1)
                data_t[:, id_] = list(map(info["i2s"].__getitem__, idx))

        invalid_ids = np.unique(np.array(invalid_ids))
        valid_ids = list(set(np.arange(len(data))) - set(invalid_ids))
        return data_t[valid_ids], len(invalid_ids)


# ── ImageTransformer ───────────────────────────────────────────────────────────

class ImageTransformer:
    """Reshapes 1-D encoded rows to/from 2-D image tensors.

    Ported from transformer.py in the Katabatic working copy.
    torch must be available at call time (imported lazily in CTABGANSynthesizer).
    """

    def __init__(self, side: int):
        self.height = side

    def transform(self, data):
        import importlib
        torch = importlib.import_module("torch")
        if self.height * self.height > data.shape[1]:
            padding = torch.zeros(
                (data.shape[0], self.height * self.height - data.shape[1]),
                device=data.device,
            )
            data = torch.cat([data, padding], dim=1)
        return data.view(-1, 1, self.height, self.height)

    def inverse_transform(self, data):
        return data.view(-1, self.height * self.height)


# ── Cond / Sampler helpers ─────────────────────────────────────────────────────

def _random_choice_prob_index_sampling(probs, col_idx):
    return np.array([
        np.random.choice(np.arange(len(probs[i])), p=probs[i])
        for i in col_idx
    ]).reshape(col_idx.shape)


def _random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)


def _maximum_interval(output_info):
    return max(item[0] for item in output_info)


class Cond(object):
    """Conditional vector sampler for categorical columns."""

    def __init__(self, data, output_info):
        self.model = []
        st = 0
        for item in output_info:
            if item[1] == "tanh":
                st += item[0]
                continue
            elif item[1] == "softmax":
                ed = st + item[0]
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                st = ed

        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        st = 0
        counter = len(self.model)
        self.p = np.zeros((counter, _maximum_interval(output_info)))
        self.p_sampling = []

        st = 0
        for item in output_info:
            if item[1] == "tanh":
                st += item[0]
                continue
            elif item[1] == "softmax":
                ed = st + item[0]
                tmp = np.sum(data[:, st:ed], axis=0)
                tmp_s = tmp.copy()
                tmp = np.log(tmp + 1)
                tmp /= np.sum(tmp)
                tmp_s /= np.sum(tmp_s)
                self.p_sampling.append(tmp_s)
                self.p[self.n_col, : item[0]] = tmp
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                st = ed

        self.interval = np.asarray(self.interval)

    def sample_train(self, batch):
        if self.n_col == 0:
            return None
        idx = np.random.choice(np.arange(self.n_col), batch)
        vec = np.zeros((batch, self.n_opt), dtype="float32")
        mask = np.zeros((batch, self.n_col), dtype="float32")
        mask[np.arange(batch), idx] = 1
        opt1prime = _random_choice_prob_index(self.p[idx])
        for i in range(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
        return vec, mask, idx, opt1prime

    def sample(self, batch):
        if self.n_col == 0:
            return None
        idx = np.random.choice(np.arange(self.n_col), batch)
        vec = np.zeros((batch, self.n_opt), dtype="float32")
        opt1prime = _random_choice_prob_index_sampling(self.p_sampling, idx)
        for i in range(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
        return vec


class Sampler(object):
    """Samples real rows conditioned on a chosen discrete category."""

    def __init__(self, data, output_info):
        self.data = data
        self.model = []
        self.n = len(data)
        st = 0
        for item in output_info:
            if item[1] == "tanh":
                st += item[0]
                continue
            elif item[1] == "softmax":
                ed = st + item[0]
                tmp = [np.nonzero(data[:, st + j])[0] for j in range(item[0])]
                self.model.append(tmp)
                st = ed

    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = [np.random.choice(self.model[c][o]) for c, o in zip(col, opt)]
        return self.data[idx]


def _get_st_ed(target_col_index, output_info):
    st, c, tc = 0, 0, 0
    for item in output_info:
        if c == target_col_index:
            break
        if item[1] == "tanh":
            st += item[0]
            if item[2] == "yes_g":
                c += 1
        elif item[1] == "softmax":
            st += item[0]
            c += 1
        tc += 1
    ed = st + output_info[tc][0]
    return (st, ed)


def _apply_activate(data, output_info):
    """Apply per-span activations (tanh / gumbel_softmax) to raw generator output."""
    import importlib
    torch = importlib.import_module("torch")
    F = importlib.import_module("torch.nn.functional")
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == "tanh":
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == "softmax":
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
    return torch.cat(data_t, dim=1)


# ── CTABGANSynthesizer ─────────────────────────────────────────────────────────

class CTABGANSynthesizer:
    """Full CTAB-GAN+ synthesizer.

    Conditional tabular GAN with:
      - Image-based 2-D data representation
      - WGAN with SLERP gradient penalty
      - Downstream classifier loss
      - Conditional vector generation

    Ported from: https://github.com/Team-TUD/CTAB-GAN-Plus
    """

    def __init__(
        self,
        class_dim: tuple = (256, 256, 256, 256),
        random_dim: int = 100,
        num_channels: int = 64,
        l2scale: float = 1e-5,
        batch_size: int = 500,
        epochs: int = 150,
    ):
        self.random_dim = random_dim
        self.class_dim = class_dim
        self.num_channels = num_channels
        self.dside: Optional[int] = None
        self.gside: Optional[int] = None
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs

        # Set device lazily; updated in fit()
        self._device_str: str = "cpu"

    def fit(
        self,
        train_data: pd.DataFrame = pd.DataFrame(),
        categorical: list = (),
        mixed: dict = None,
        general: list = (),
        non_categorical: list = (),
        type: dict = None,
    ) -> None:
        import importlib
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        F = importlib.import_module("torch.nn.functional")
        optim_mod = importlib.import_module("torch.optim")

        if mixed is None:
            mixed = {}
        if type is None:
            type = {}

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._device_str = str(device)

        # ── Inner classes requiring nn.Module ────────────────────────────────

        class_dim_local = self.class_dim
        num_channels_local = self.num_channels

        class Classifier(nn.Module):
            def __init__(self_, input_dim, dis_dims, st_ed):
                super().__init__()
                dim = input_dim - (st_ed[1] - st_ed[0])
                self_.str_end = st_ed
                seq = []
                for item in list(dis_dims):
                    seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
                    dim = item
                span = st_ed[1] - st_ed[0]
                if span == 1:
                    seq += [nn.Linear(dim, 1)]
                elif span == 2:
                    seq += [nn.Linear(dim, 1), nn.Sigmoid()]
                else:
                    seq += [nn.Linear(dim, span)]
                self_.seq = nn.Sequential(*seq)

            def forward(self_, input_):
                span = self_.str_end[1] - self_.str_end[0]
                if span == 1:
                    label = input_[:, self_.str_end[0] : self_.str_end[1]]
                else:
                    label = torch.argmax(
                        input_[:, self_.str_end[0] : self_.str_end[1]], dim=-1
                    )
                new_inp = torch.cat(
                    (input_[:, : self_.str_end[0]], input_[:, self_.str_end[1] :]), dim=1
                )
                if span in (1, 2):
                    return self_.seq(new_inp).view(-1), label
                return self_.seq(new_inp), label

        def determine_layers_disc(side, num_channels):
            assert 4 <= side <= 128, f"Discriminator side {side} out of range [4, 128]"
            layer_dims = [(1, side), (num_channels, side // 2)]
            while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
                layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

            ln_shapes, nc, ns = [], num_channels, side / 2
            for _ in range(len(layer_dims) - 1):
                ln_shapes.append([int(nc), int(ns), int(ns)])
                nc *= 2
                ns /= 2

            layers = []
            for prev, curr, ln in zip(layer_dims, layer_dims[1:], ln_shapes):
                layers += [
                    nn.Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
                    nn.LayerNorm(ln),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            layers += [
                nn.Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0),
                nn.ReLU(True),
            ]
            return layers

        def determine_layers_gen(side, random_dim, num_channels):
            assert 4 <= side <= 128, f"Generator side {side} out of range [4, 128]"
            layer_dims = [(1, side), (num_channels, side // 2)]
            while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
                layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

            nc = num_channels * (2 ** (len(layer_dims) - 2))
            ns = int(side / (2 ** (len(layer_dims) - 1)))
            ln_shapes = []
            for _ in range(len(layer_dims) - 1):
                ln_shapes.append([int(nc), int(ns), int(ns)])
                nc //= 2
                ns *= 2

            layers = [
                nn.ConvTranspose2d(
                    random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0,
                    output_padding=0, bias=False,
                )
            ]
            for prev, curr, ln in zip(
                reversed(layer_dims), reversed(layer_dims[:-1]), ln_shapes
            ):
                layers += [
                    nn.LayerNorm(ln),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(prev[0], curr[0], 4, 2, 1,
                                       output_padding=0, bias=True),
                ]
            return layers

        class Generator(nn.Module):
            def __init__(self_, side, layers):
                super().__init__()
                self_.side = side
                self_.seq = nn.Sequential(*layers)

            def forward(self_, input_):
                return self_.seq(input_)

        class Discriminator(nn.Module):
            def __init__(self_, side, layers):
                super().__init__()
                self_.side = side
                info_idx = len(layers) - 2
                self_.seq = nn.Sequential(*layers)
                self_.seq_info = nn.Sequential(*layers[:info_idx])

            def forward(self_, input_):
                return self_.seq(input_), self_.seq_info(input_)

        def slerp(val, low, high):
            low_norm = low / torch.norm(low, dim=1, keepdim=True)
            high_norm = high / torch.norm(high, dim=1, keepdim=True)
            omega = torch.acos(
                (low_norm * high_norm).sum(1).clamp(-1 + 1e-6, 1 - 1e-6)
            ).view(val.size(0), 1)
            so = torch.sin(omega)
            return (torch.sin((1.0 - val) * omega) / so) * low + \
                   (torch.sin(val * omega) / so) * high

        def calc_gradient_penalty_slerp(netD, real_data, fake_data, transformer,
                                         device_, lambda_=10):
            batchsize = real_data.shape[0]
            alpha = torch.rand(batchsize, 1, device=device_)
            interpolates = slerp(alpha, real_data, fake_data).to(device_)
            interpolates = transformer.transform(interpolates)
            interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
            disc_interpolates, _ = netD(interpolates)
            gradients = torch.autograd.grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones(disc_interpolates.size(), device=device_),
                create_graph=True, retain_graph=True, only_inputs=True,
            )[0]
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
            return gradient_penalty

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        def cond_loss(data, output_info, c, m):
            loss = []
            st, st_c = 0, 0
            for item in output_info:
                if item[1] == "tanh":
                    st += item[0]
                    continue
                elif item[1] == "softmax":
                    ed, ed_c = st + item[0], st_c + item[0]
                    tmp = F.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction="none",
                    )
                    loss.append(tmp)
                    st, st_c = ed, ed_c
            loss = torch.stack(loss, dim=1)
            return (loss * m).sum() / data.size(0)

        # ── Encode data ──────────────────────────────────────────────────────
        problem_type = None
        target_index = None
        if type:
            problem_type = list(type.keys())[0]
            if problem_type:
                target_index = train_data.columns.get_loc(type[problem_type])

        self.transformer = DataTransformer(
            train_data=train_data,
            categorical_list=list(categorical),
            mixed_dict=dict(mixed),
            general_list=list(general),
            non_categorical_list=list(non_categorical),
        )
        self.transformer.fit()
        train_encoded = self.transformer.transform(train_data.values)
        data_sampler = Sampler(train_encoded, self.transformer.output_info)
        data_dim = self.transformer.output_dim
        self.cond_generator = Cond(train_encoded, self.transformer.output_info)

        sides = [4, 8, 16, 24, 32, 64]
        col_size_d = data_dim + self.cond_generator.n_opt
        for s in sides:
            if s * s >= col_size_d:
                self.dside = s
                break
        if self.dside is None:
            self.dside = 64

        col_size_g = data_dim
        for s in sides:
            if s * s >= col_size_g:
                self.gside = s
                break
        if self.gside is None:
            self.gside = 64

        # ── Build networks ───────────────────────────────────────────────────
        layers_G = determine_layers_gen(
            self.gside, self.random_dim + self.cond_generator.n_opt, self.num_channels
        )
        layers_D = determine_layers_disc(self.dside, self.num_channels)

        self.generator = Generator(self.gside, layers_G).to(device)
        discriminator = Discriminator(self.dside, layers_D).to(device)

        optimizer_params = dict(
            lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale
        )
        optimizerG = optim_mod.Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = optim_mod.Adam(discriminator.parameters(), **optimizer_params)

        st_ed = None
        classifier = None
        optimizerC = None
        if target_index is not None:
            st_ed = _get_st_ed(target_index, self.transformer.output_info)
            classifier = Classifier(data_dim, self.class_dim, st_ed).to(device)
            optimizerC = optim_mod.Adam(classifier.parameters(), **optimizer_params)

        self.generator.apply(weights_init)
        discriminator.apply(weights_init)

        self.Gtransformer = ImageTransformer(self.gside)
        self.Dtransformer = ImageTransformer(self.dside)

        steps_per_epoch = max(1, len(train_encoded) // self.batch_size)
        ci = 5  # discriminator steps per generator step

        for epoch in range(self.epochs):
            for _ in range(steps_per_epoch):

                # ── discriminator loop ──────────────────────────────────────
                for _ in range(ci):
                    noisez = torch.randn(self.batch_size, self.random_dim, device=device)
                    condvec = self.cond_generator.sample_train(self.batch_size)
                    c, m, col, opt = condvec
                    c = torch.from_numpy(c).to(device)
                    m = torch.from_numpy(m).to(device)
                    noisez = torch.cat([noisez, c], dim=1)
                    noisez = noisez.view(
                        self.batch_size,
                        self.random_dim + self.cond_generator.n_opt, 1, 1,
                    )
                    perm = np.random.permutation(self.batch_size)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c_perm = c[perm]
                    real = torch.from_numpy(real.astype("float32")).to(device)

                    fake = self.generator(noisez)
                    faket = self.Gtransformer.inverse_transform(fake)
                    fakeact = _apply_activate(faket, self.transformer.output_info)

                    fake_cat = self.Dtransformer.transform(torch.cat([fakeact, c], dim=1))
                    real_cat = self.Dtransformer.transform(torch.cat([real, c_perm], dim=1))

                    optimizerD.zero_grad()
                    d_real, _ = discriminator(real_cat)
                    (-torch.mean(d_real)).backward()
                    d_fake, _ = discriminator(fake_cat)
                    torch.mean(d_fake).backward()
                    pen = calc_gradient_penalty_slerp(
                        discriminator,
                        torch.cat([real, c_perm], dim=1),
                        torch.cat([fakeact, c], dim=1),
                        self.Dtransformer, device,
                    )
                    pen.backward()
                    optimizerD.step()

                # ── generator step ──────────────────────────────────────────
                noisez = torch.randn(self.batch_size, self.random_dim, device=device)
                condvec = self.cond_generator.sample_train(self.batch_size)
                c, m, col, opt = condvec
                c = torch.from_numpy(c).to(device)
                m = torch.from_numpy(m).to(device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez = noisez.view(
                    self.batch_size,
                    self.random_dim + self.cond_generator.n_opt, 1, 1,
                )

                optimizerG.zero_grad()
                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = _apply_activate(faket, self.transformer.output_info)
                fake_cat_d = self.Dtransformer.transform(torch.cat([fakeact, c], dim=1))
                y_fake, info_fake = discriminator(fake_cat_d)
                cross_entropy = cond_loss(faket, self.transformer.output_info, c, m)
                _, info_real = discriminator(real_cat)

                g = -torch.mean(y_fake) + cross_entropy
                g.backward(retain_graph=True)
                loss_info = (
                    torch.norm(
                        torch.mean(info_fake.view(self.batch_size, -1), dim=0) -
                        torch.mean(info_real.view(self.batch_size, -1), dim=0), 1
                    ) +
                    torch.norm(
                        torch.std(info_fake.view(self.batch_size, -1), dim=0) -
                        torch.std(info_real.view(self.batch_size, -1), dim=0), 1
                    )
                )
                loss_info.backward()
                optimizerG.step()

                # ── classifier step ─────────────────────────────────────────
                if problem_type and classifier is not None:
                    fake = self.generator(noisez)
                    faket = self.Gtransformer.inverse_transform(fake)
                    fakeact = _apply_activate(faket, self.transformer.output_info)

                    real_pre, real_label = classifier(real)
                    fake_pre, fake_label = classifier(fakeact)

                    span = st_ed[1] - st_ed[0]
                    if span == 1:
                        c_loss = nn.SmoothL1Loss()
                        real_label = real_label.type_as(real_pre).reshape(real_pre.size())
                        fake_label = fake_label.type_as(fake_pre).reshape(fake_pre.size())
                    elif span == 2:
                        c_loss = nn.BCELoss()
                        real_label = real_label.type_as(real_pre)
                        fake_label = fake_label.type_as(fake_pre)
                    else:
                        c_loss = nn.CrossEntropyLoss()

                    loss_cc = c_loss(real_pre, real_label)
                    loss_cg = c_loss(fake_pre, fake_label)

                    optimizerG.zero_grad()
                    loss_cg.backward()
                    optimizerG.step()

                    optimizerC.zero_grad()
                    loss_cc.backward()
                    optimizerC.step()

            if (epoch + 1) % 10 == 0:
                print(f"  [CTAB-GAN+] Epoch {epoch + 1}/{self.epochs}")

    def sample(self, n: int) -> np.ndarray:
        """Generate n synthetic rows.

        Returns:
            Numpy array of shape (n, num_original_columns).
        """
        import importlib
        torch = importlib.import_module("torch")

        self.generator.eval()
        device = torch.device(self._device_str)
        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1

        data = []
        for _ in range(steps):
            noisez = torch.randn(self.batch_size, self.random_dim, device=device)
            condvec = self.cond_generator.sample(self.batch_size)
            if condvec is None:
                pass
            else:
                c = torch.from_numpy(condvec).to(device)
                noisez = torch.cat([noisez, c], dim=1)
            noisez = noisez.view(
                self.batch_size,
                self.random_dim + self.cond_generator.n_opt, 1, 1,
            )
            fake = self.generator(noisez)
            faket = self.Gtransformer.inverse_transform(fake)
            fakeact = _apply_activate(faket, output_info)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        result, resample = self.transformer.inverse_transform(data)

        # Resample to fill invalid rows
        max_iter = 20
        itr = 0
        while len(result) < n and itr < max_iter:
            itr += 1
            steps_left = resample // self.batch_size + 1
            data_resample = []
            for _ in range(steps_left):
                noisez = torch.randn(self.batch_size, self.random_dim, device=device)
                condvec = self.cond_generator.sample(self.batch_size)
                if condvec is not None:
                    c = torch.from_numpy(condvec).to(device)
                    noisez = torch.cat([noisez, c], dim=1)
                noisez = noisez.view(
                    self.batch_size,
                    self.random_dim + self.cond_generator.n_opt, 1, 1,
                )
                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = _apply_activate(faket, output_info)
                data_resample.append(fakeact.detach().cpu().numpy())
            data_resample = np.concatenate(data_resample, axis=0)
            res, resample = self.transformer.inverse_transform(data_resample)
            result = np.concatenate([result, res], axis=0)

        return result[:n]
