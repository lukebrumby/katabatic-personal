"""
Utility helpers for CTAB-GAN.

Contains:
- detect_column_types  — heuristic column-type detection for models.py
- save_metadata        — JSON metadata writer
- DataTransformer      — mode-specific normalisation + one-hot encoding
- ImageTransformer     — tabular <-> square-image reshaping
- DataPrep             — label-encoding / log-transform / missing-value handling
- Condvec              — conditional vector sampler
- Sampler              — training-by-sampling helper
- cond_loss            — conditional cross-entropy loss
- get_st_ed            — target-column index helper
- Classifier           — auxiliary classification head
- Discriminator        — convolutional discriminator
- Generator            — convolutional generator
- determine_layers_*   — DCGAN layer builders
- apply_activate       — post-generation activation
- weights_init         — DCGAN weight initialiser
- CTABGANSynthesizer   — core GAN training and sampling loop

torch and sklearn are imported at module level here.  models.py imports this
module lazily (inside train()) so that `import CTABGANModel` does NOT load
torch until the model is actually trained.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (
    Dropout,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential,
    Conv2d,
    ConvTranspose2d,
    BatchNorm2d,
    Sigmoid,
    init,
    BCELoss,
    CrossEntropyLoss,
)
from sklearn.mixture import BayesianGaussianMixture
from sklearn import preprocessing, model_selection
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Column type detection & metadata helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DataTransformer  (mode-specific normalisation + one-hot encoding)
# ---------------------------------------------------------------------------

class DataTransformer:
    """
    Transformer responsible for processing data to train the CTABGANSynthesizer model.

    Applies mode-specific normalisation to continuous/mixed columns via
    BayesianGaussianMixture and one-hot encoding to categorical columns.
    """

    def __init__(
        self,
        train_data=pd.DataFrame,
        categorical_list=[],
        mixed_dict={},
        n_clusters=10,
        eps=0.005,
    ):
        self.meta = None
        self.train_data = train_data
        self.categorical_columns = categorical_list
        self.mixed_columns = mixed_dict
        self.n_clusters = n_clusters
        self.eps = eps
        self.ordering = []
        self.output_info = []
        self.output_dim = 0
        self.components = []
        self.filter_arr = []
        self.meta = self.get_metadata()

    def get_metadata(self):
        meta = []
        for index in range(self.train_data.shape[1]):
            column = self.train_data.iloc[:, index]
            if index in self.categorical_columns:
                mapper = column.value_counts().index.tolist()
                meta.append({
                    "name": index,
                    "type": "categorical",
                    "size": len(mapper),
                    "i2s": mapper,
                })
            elif index in self.mixed_columns.keys():
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
        model = []

        for id_, info in enumerate(self.meta):
            if info["type"] == "continuous":
                gm = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    n_init=1,
                    random_state=42,
                )
                gm.fit(data[:, id_].reshape([-1, 1]))
                model.append(gm)
                old_comp = gm.weights_ > self.eps
                mode_freq = (
                    pd.Series(gm.predict(data[:, id_].reshape([-1, 1])))
                    .value_counts()
                    .keys()
                )
                comp = []
                for i in range(self.n_clusters):
                    if (i in mode_freq) & old_comp[i]:
                        comp.append(True)
                    else:
                        comp.append(False)
                self.components.append(comp)
                self.output_info += [(1, "tanh"), (np.sum(comp), "softmax")]
                self.output_dim += 1 + np.sum(comp)

            elif info["type"] == "mixed":
                gm1 = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    n_init=1,
                    random_state=42,
                )
                gm2 = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    n_init=1,
                    random_state=42,
                )
                gm1.fit(data[:, id_].reshape([-1, 1]))
                filter_arr = []
                for element in data[:, id_]:
                    if element not in info["modal"]:
                        filter_arr.append(True)
                    else:
                        filter_arr.append(False)
                self.filter_arr.append(filter_arr)
                gm2.fit(data[:, id_][filter_arr].reshape([-1, 1]))
                model.append((gm1, gm2))
                old_comp = gm2.weights_ > self.eps
                mode_freq = (
                    pd.Series(gm2.predict(data[:, id_][filter_arr].reshape([-1, 1])))
                    .value_counts()
                    .keys()
                )
                comp = []
                for i in range(self.n_clusters):
                    if (i in mode_freq) & old_comp[i]:
                        comp.append(True)
                    else:
                        comp.append(False)
                self.components.append(comp)
                self.output_info += [
                    (1, "tanh"),
                    (np.sum(comp) + len(info["modal"]), "softmax"),
                ]
                self.output_dim += 1 + np.sum(comp) + len(info["modal"])

            else:
                model.append(None)
                self.components.append(None)
                self.output_info += [(info["size"], "softmax")]
                self.output_dim += info["size"]

        self.model = model

    def transform(self, data):
        values = []
        mixed_counter = 0

        for id_, info in enumerate(self.meta):
            current = data[:, id_]
            if info["type"] == "continuous":
                current = current.reshape([-1, 1])
                means = self.model[id_].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))
                features = np.empty(shape=(len(current), self.n_clusters))
                features = (current - means) / (4 * stds)
                n_opts = sum(self.components[id_])
                opt_sel = np.zeros(len(data), dtype="int")
                probs = self.model[id_].predict_proba(current.reshape([-1, 1]))
                probs = probs[:, self.components[id_]]
                for i in range(len(data)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)
                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1
                idx = np.arange((len(features)))
                features = features[:, self.components[id_]]
                features = features[idx, opt_sel].reshape([-1, 1])
                features = np.clip(features, -0.99, 0.99)
                re_ordered_phot = np.zeros_like(probs_onehot)
                col_sums = probs_onehot.sum(axis=0)
                n = probs_onehot.shape[1]
                largest_indices = np.argsort(-1 * col_sums)[:n]
                for id, val in enumerate(largest_indices):
                    re_ordered_phot[:, id] = probs_onehot[:, val]
                self.ordering.append(largest_indices)
                values += [features, re_ordered_phot]

            elif info["type"] == "mixed":
                means_0 = self.model[id_][0].means_.reshape([-1])
                stds_0 = np.sqrt(self.model[id_][0].covariances_).reshape([-1])
                zero_std_list = []
                means_needed = []
                stds_needed = []
                for mode in info["modal"]:
                    if mode != -9999999:
                        dist = []
                        for idx, val in enumerate(list(means_0.flatten())):
                            dist.append(abs(mode - val))
                        index_min = np.argmin(np.array(dist))
                        zero_std_list.append(index_min)
                    else:
                        continue
                mode_vals = []
                for idx in zero_std_list:
                    means_needed.append(means_0[idx])
                    stds_needed.append(stds_0[idx])
                for i, j, k in zip(info["modal"], means_needed, stds_needed):
                    this_val = np.clip(((i - j) / (4 * k)), -0.99, 0.99)
                    mode_vals.append(this_val)
                if -9999999 in info["modal"]:
                    mode_vals.append(0)
                current = current.reshape([-1, 1])
                filter_arr = self.filter_arr[mixed_counter]
                current = current[filter_arr]
                means = self.model[id_][1].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_][1].covariances_).reshape((1, self.n_clusters))
                features = np.empty(shape=(len(current), self.n_clusters))
                features = (current - means) / (4 * stds)
                n_opts = sum(self.components[id_])
                probs = self.model[id_][1].predict_proba(current.reshape([-1, 1]))
                probs = probs[:, self.components[id_]]
                opt_sel = np.zeros(len(current), dtype="int")
                for i in range(len(current)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)
                idx = np.arange((len(features)))
                features = features[:, self.components[id_]]
                features = features[idx, opt_sel].reshape([-1, 1])
                features = np.clip(features, -0.99, 0.99)
                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1
                extra_bits = np.zeros([len(current), len(info["modal"])])
                temp_probs_onehot = np.concatenate([extra_bits, probs_onehot], axis=1)
                final = np.zeros(
                    [len(data), 1 + probs_onehot.shape[1] + len(info["modal"])]
                )
                features_curser = 0
                for idx, val in enumerate(data[:, id_]):
                    if val in info["modal"]:
                        category_ = list(map(info["modal"].index, [val]))[0]
                        final[idx, 0] = mode_vals[category_]
                        final[idx, (category_ + 1)] = 1
                    else:
                        final[idx, 0] = features[features_curser]
                        final[idx, (1 + len(info["modal"])):] = temp_probs_onehot[
                            features_curser
                        ][len(info["modal"]):]
                        features_curser = features_curser + 1
                just_onehot = final[:, 1:]
                re_ordered_jhot = np.zeros_like(just_onehot)
                n = just_onehot.shape[1]
                col_sums = just_onehot.sum(axis=0)
                largest_indices = np.argsort(-1 * col_sums)[:n]
                for id, val in enumerate(largest_indices):
                    re_ordered_jhot[:, id] = just_onehot[:, val]
                final_features = final[:, 0].reshape([-1, 1])
                self.ordering.append(largest_indices)
                values += [final_features, re_ordered_jhot]
                mixed_counter = mixed_counter + 1

            else:
                self.ordering.append(None)
                col_t = np.zeros([len(data), info["size"]])
                idx = list(map(info["i2s"].index, current))
                col_t[np.arange(len(data)), idx] = 1
                values.append(col_t)

        return np.concatenate(values, axis=1)

    def inverse_transform(self, data):
        data_t = np.zeros([len(data), len(self.meta)])
        st = 0

        for id_, info in enumerate(self.meta):
            if info["type"] == "continuous":
                u = data[:, st]
                u = np.clip(u, -1, 1)
                v = data[:, st + 1: st + 1 + np.sum(self.components[id_])]
                order = self.ordering[id_]
                v_re_ordered = np.zeros_like(v)
                for id, val in enumerate(order):
                    v_re_ordered[:, val] = v[:, id]
                v = v_re_ordered
                v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                v_t[:, self.components[id_]] = v
                v = v_t
                means = self.model[id_].means_.reshape([-1])
                stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]
                tmp = u * 4 * std_t + mean_t
                data_t[:, id_] = tmp
                st += 1 + np.sum(self.components[id_])

            elif info["type"] == "mixed":
                u = data[:, st]
                u = np.clip(u, -1, 1)
                full_v = data[
                    :,
                    (st + 1): (st + 1) + len(info["modal"]) + np.sum(self.components[id_]),
                ]
                order = self.ordering[id_]
                full_v_re_ordered = np.zeros_like(full_v)
                for id, val in enumerate(order):
                    full_v_re_ordered[:, val] = full_v[:, id]
                full_v = full_v_re_ordered
                mixed_v = full_v[:, :len(info["modal"])]
                v = full_v[:, -np.sum(self.components[id_]):]
                v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                v_t[:, self.components[id_]] = v
                v = np.concatenate([mixed_v, v_t], axis=1)
                p_argmax = np.argmax(v, axis=1)
                means = self.model[id_][1].means_.reshape([-1])
                stds = np.sqrt(self.model[id_][1].covariances_).reshape([-1])
                result = np.zeros_like(u)
                for idx in range(len(data)):
                    if p_argmax[idx] < len(info["modal"]):
                        argmax_value = p_argmax[idx]
                        result[idx] = float(
                            list(map(info["modal"].__getitem__, [argmax_value]))[0]
                        )
                    else:
                        std_t = stds[(p_argmax[idx] - len(info["modal"]))]
                        mean_t = means[(p_argmax[idx] - len(info["modal"]))]
                        result[idx] = u[idx] * 4 * std_t + mean_t
                data_t[:, id_] = result
                st += 1 + np.sum(self.components[id_]) + len(info["modal"])

            else:
                current = data[:, st: st + info["size"]]
                idx = np.argmax(current, axis=1)
                data_t[:, id_] = list(map(info["i2s"].__getitem__, idx))
                st += info["size"]

        return data_t


# ---------------------------------------------------------------------------
# ImageTransformer  (tabular <-> square-image reshaping)
# ---------------------------------------------------------------------------

class ImageTransformer:
    """Converts tabular data rows to square images and vice versa."""

    def __init__(self, side):
        self.height = side

    def transform(self, data):
        if self.height * self.height > len(data[0]):
            padding = torch.zeros(
                (len(data), self.height * self.height - len(data[0]))
            ).to(data.device)
            data = torch.cat([data, padding], axis=1)
        return data.view(-1, 1, self.height, self.height)

    def inverse_transform(self, data):
        data = data.view(-1, self.height * self.height)
        return data


# ---------------------------------------------------------------------------
# DataPrep  (label-encoding, log-transforms, missing-value handling)
# ---------------------------------------------------------------------------

class DataPrep(object):
    """Pre-processing and post-processing for CTAB-GAN training data."""

    def __init__(
        self,
        df: pd.DataFrame,
        categorical_columns: list,
        log_columns: list,
        mixed_columns: dict,
        integer_columns: list,
        problem_type: dict,
        test_ratio: float,
    ):
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.column_types = dict()
        self.column_types["categorical"] = []
        self.column_types["mixed"] = {}
        self.lower_bounds = {}
        self.label_encoder_list = []

        if problem_type:
            target_col = list(problem_type.values())[0]
            y_real = df[target_col]
            X_real = df.drop(columns=[target_col])
            X_train_real, _, y_train_real, _ = model_selection.train_test_split(
                X_real, y_real, test_size=test_ratio, stratify=y_real, random_state=42
            )
            X_train_real[target_col] = y_train_real
            self.df = X_train_real
        else:
            self.df = df.copy()

        self.df = self.df.replace(r" ", np.nan)
        self.df = self.df.fillna("empty")

        all_columns = set(self.df.columns)
        irrelevant_missing_columns = set(self.categorical_columns)
        relevant_missing_columns = list(all_columns - irrelevant_missing_columns)

        for i in relevant_missing_columns:
            if i in list(self.mixed_columns.keys()):
                if "empty" in list(self.df[i].values):
                    self.df[i] = self.df[i].apply(
                        lambda x: -9999999 if x == "empty" else x
                    )
                    self.mixed_columns[i].append(-9999999)
            else:
                if "empty" in list(self.df[i].values):
                    self.df[i] = self.df[i].apply(
                        lambda x: -9999999 if x == "empty" else x
                    )
                    self.mixed_columns[i] = [-9999999]

        if self.log_columns:
            for log_column in self.log_columns:
                eps = 1
                lower = np.min(
                    self.df.loc[self.df[log_column] != -9999999][log_column].values
                )
                self.lower_bounds[log_column] = lower
                if lower > 0:
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x) if x != -9999999 else -9999999
                    )
                elif lower == 0:
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x + eps) if x != -9999999 else -9999999
                    )
                else:
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x - lower + eps) if x != -9999999 else -9999999
                    )

        for column_index, column in enumerate(self.df.columns):
            if column in self.categorical_columns:
                label_encoder = preprocessing.LabelEncoder()
                self.df[column] = self.df[column].astype(str)
                label_encoder.fit(self.df[column])
                current_label_encoder = dict()
                current_label_encoder["column"] = column
                current_label_encoder["label_encoder"] = label_encoder
                transformed_column = label_encoder.transform(self.df[column])
                self.df[column] = transformed_column
                self.label_encoder_list.append(current_label_encoder)
                self.column_types["categorical"].append(column_index)
            elif column in self.mixed_columns:
                self.column_types["mixed"][column_index] = self.mixed_columns[column]

        super().__init__()

    def inverse_prep(self, data, eps=1):
        """Inverse transform generated data to original format."""
        df_sample = pd.DataFrame(data, columns=self.df.columns)

        for i in range(len(self.label_encoder_list)):
            le = self.label_encoder_list[i]["label_encoder"]
            df_sample[self.label_encoder_list[i]["column"]] = df_sample[
                self.label_encoder_list[i]["column"]
            ].astype(int)
            df_sample[self.label_encoder_list[i]["column"]] = le.inverse_transform(
                df_sample[self.label_encoder_list[i]["column"]]
            )

        if self.log_columns:
            for i in df_sample:
                if i in self.log_columns:
                    lower_bound = self.lower_bounds[i]
                    if lower_bound > 0:
                        df_sample[i] = df_sample[i].apply(
                            lambda x: np.exp(x) if x != -9999999 else -9999999
                        )
                    elif lower_bound == 0:
                        df_sample[i] = df_sample[i].apply(
                            lambda x: (
                                np.ceil(np.exp(x) - eps)
                                if ((x != -9999999) & ((np.exp(x) - eps) < 0))
                                else (np.exp(x) - eps if x != -9999999 else -9999999)
                            )
                        )
                    else:
                        df_sample[i] = df_sample[i].apply(
                            lambda x: (
                                np.exp(x) - eps + lower_bound
                                if x != -9999999
                                else -9999999
                            )
                        )

        if self.integer_columns:
            for column in self.integer_columns:
                df_sample[column] = np.round(df_sample[column].values)
                df_sample[column] = df_sample[column].astype(int)

        df_sample.replace(-9999999, np.nan, inplace=True)
        df_sample.replace("empty", np.nan, inplace=True)

        return df_sample


# ---------------------------------------------------------------------------
# CTABGANSynthesizer helpers
# ---------------------------------------------------------------------------

def random_choice_prob_index_sampling(probs, col_idx):
    """Sample a specific category within a chosen one-hot-encoding."""
    option_list = []
    for i in col_idx:
        pp = probs[i] + 1e-6
        pp = pp / sum(pp)
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
    return np.array(option_list).reshape(col_idx.shape)


class Condvec(object):
    """Samples conditional vectors for the generator."""

    def __init__(self, data, output_info):
        self.model = []
        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        self.p_log_sampling = []
        self.p_sampling = []

        st = 0
        for item in output_info:
            if item[1] == "tanh":
                st += item[0]
                continue
            elif item[1] == "softmax":
                ed = st + item[0]
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                self.interval.append((self.n_opt, item[0]))
                self.n_col += 1
                self.n_opt += item[0]
                freq = np.sum(data[:, st:ed], axis=0)
                log_freq = np.log(freq + 1)
                log_pmf = log_freq / np.sum(log_freq)
                self.p_log_sampling.append(log_pmf)
                pmf = freq / np.sum(freq)
                self.p_sampling.append(pmf)
                st = ed

        self.interval = np.asarray(self.interval)

    def sample_train(self, batch):
        if self.n_col == 0:
            return None
        vec = np.zeros((batch, self.n_opt), dtype="float32")
        idx = np.random.choice(np.arange(self.n_col), batch)
        mask = np.zeros((batch, self.n_col), dtype="float32")
        mask[np.arange(batch), idx] = 1
        opt1prime = random_choice_prob_index_sampling(self.p_log_sampling, idx)
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
        return vec, mask, idx, opt1prime

    def sample(self, batch):
        if self.n_col == 0:
            return None
        vec = np.zeros((batch, self.n_opt), dtype="float32")
        idx = np.random.choice(np.arange(self.n_col), batch)
        opt1prime = random_choice_prob_index_sampling(self.p_sampling, idx)
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
        return vec


def cond_loss(data, output_info, c, m):
    """Conditional cross-entropy loss for the generator."""
    tmp_loss = []
    st = 0
    st_c = 0
    for item in output_info:
        if item[1] == "tanh":
            st += item[0]
            continue
        elif item[1] == "softmax":
            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(
                data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction="none"
            )
            tmp_loss.append(tmp)
            st = ed
            st_c = ed_c
    tmp_loss = torch.stack(tmp_loss, dim=1)
    loss = (tmp_loss * m).sum() / data.size()[0]
    return loss


class Sampler(object):
    """Samples real data rows according to a conditional vector."""

    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
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
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st + j])[0])
                self.model.append(tmp)
                st = ed

    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        return self.data[idx]


def get_st_ed(target_col_index, output_info):
    """Starting and ending positions of the target column in transformed data."""
    st = 0
    c = 0
    tc = 0
    for item in output_info:
        if c == target_col_index:
            break
        if item[1] == "tanh":
            st += item[0]
        elif item[1] == "softmax":
            st += item[0]
            c += 1
        tc += 1
    ed = st + output_info[tc][0]
    return (st, ed)


class Classifier(Module):
    """Auxiliary classifier head used to preserve label fidelity during generation."""

    def __init__(self, input_dim, class_dims, st_ed):
        super(Classifier, self).__init__()
        self.dim = input_dim - (st_ed[1] - st_ed[0])
        self.str_end = st_ed
        seq = []
        tmp_dim = self.dim
        for item in list(class_dims):
            seq += [Linear(tmp_dim, item), LeakyReLU(0.2), Dropout(0.5)]
            tmp_dim = item
        if (st_ed[1] - st_ed[0]) == 2:
            seq += [Linear(tmp_dim, 1), Sigmoid()]
        else:
            seq += [Linear(tmp_dim, (st_ed[1] - st_ed[0]))]
        self.seq = Sequential(*seq)

    def forward(self, input):
        label = torch.argmax(input[:, self.str_end[0]: self.str_end[1]], axis=-1)
        new_imp = torch.cat(
            (input[:, :self.str_end[0]], input[:, self.str_end[1]:]), 1
        )
        if (self.str_end[1] - self.str_end[0]) == 2:
            return self.seq(new_imp).view(-1), label
        else:
            return self.seq(new_imp), label


class Discriminator(Module):
    """Convolutional discriminator network."""

    def __init__(self, layers):
        super(Discriminator, self).__init__()
        self.seq = Sequential(*layers)
        self.seq_info = Sequential(*layers[:len(layers) - 2])

    def forward(self, input):
        return (self.seq(input)), self.seq_info(input)


class Generator(Module):
    """Convolutional generator network."""

    def __init__(self, layers):
        super(Generator, self).__init__()
        self.seq = Sequential(*layers)

    def forward(self, input):
        return self.seq(input)


def determine_layers_disc(side, num_channels):
    """Build DCGAN-style discriminator layers."""
    layer_dims = [(1, side), (num_channels, side // 2)]
    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))
    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True),
        ]
    layers_D += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0), Sigmoid()]
    return layers_D


def determine_layers_gen(side, random_dim, num_channels):
    """Build DCGAN-style generator layers."""
    layer_dims = [(1, side), (num_channels, side // 2)]
    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))
    layers_G = [
        ConvTranspose2d(
            random_dim,
            layer_dims[-1][0],
            layer_dims[-1][1],
            1,
            0,
            output_padding=0,
            bias=False,
        )
    ]
    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [
            BatchNorm2d(prev[0]),
            ReLU(True),
            ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True),
        ]
    return layers_G


def apply_activate(data, output_info):
    """Apply final activation (tanh for numeric, gumbel-softmax for categorical)."""
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
    act_data = torch.cat(data_t, dim=1)
    return act_data


def weights_init(model):
    """Initialise Conv and BatchNorm weights using normal distribution."""
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        init.normal_(model.weight.data, 1.0, 0.02)
        init.constant_(model.bias.data, 0)


# ---------------------------------------------------------------------------
# CTABGANSynthesizer  (core GAN training and sampling loop)
# ---------------------------------------------------------------------------

class CTABGANSynthesizer:
    """
    Core CTAB-GAN model: trains a convolutional conditional GAN on tabular data
    and generates synthetic samples via inverse transformation.
    """

    def __init__(
        self,
        class_dim=(256, 256, 256, 256),
        random_dim=100,
        num_channels=64,
        l2scale=1e-5,
        batch_size=500,
        epochs=1,
    ):
        self.random_dim = random_dim
        self.class_dim = class_dim
        self.num_channels = num_channels
        self.dside = None
        self.gside = None
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = None

    def fit(self, train_data=pd.DataFrame, categorical=[], mixed={}, type={}):
        problem_type = None
        target_index = None

        if type:
            problem_type = list(type.keys())[0]
            if problem_type:
                target_index = train_data.columns.get_loc(type[problem_type])

        self.transformer = DataTransformer(
            train_data=train_data, categorical_list=categorical, mixed_dict=mixed
        )
        self.transformer.fit()
        train_data = self.transformer.transform(train_data.values)
        data_dim = self.transformer.output_dim

        data_sampler = Sampler(train_data, self.transformer.output_info)
        self.cond_generator = Condvec(train_data, self.transformer.output_info)

        sides = [4, 8, 16, 24, 32]
        col_size_d = data_dim + self.cond_generator.n_opt
        for i in sides:
            if i * i >= col_size_d:
                self.dside = i
                break

        sides = [4, 8, 16, 24, 32]
        col_size_g = data_dim
        for i in sides:
            if i * i >= col_size_g:
                self.gside = i
                break

        layers_G = determine_layers_gen(
            self.gside, self.random_dim + self.cond_generator.n_opt, self.num_channels
        )
        layers_D = determine_layers_disc(self.dside, self.num_channels)
        self.generator = Generator(layers_G).to(self.device)
        discriminator = Discriminator(layers_D).to(self.device)

        optimizer_params = dict(
            lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale
        )
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)

        st_ed = None
        classifier = None
        optimizerC = None
        if target_index is not None:
            st_ed = get_st_ed(target_index, self.transformer.output_info)
            classifier = Classifier(data_dim, self.class_dim, st_ed).to(self.device)
            optimizerC = optim.Adam(classifier.parameters(), **optimizer_params)

        self.generator.apply(weights_init)
        discriminator.apply(weights_init)

        self.Gtransformer = ImageTransformer(self.gside)
        self.Dtransformer = ImageTransformer(self.dside)

        steps_per_epoch = max(1, len(train_data) // self.batch_size)
        for i in tqdm(range(self.epochs)):
            for _ in range(steps_per_epoch):
                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                condvec = self.cond_generator.sample_train(self.batch_size)
                c, m, col, opt = condvec
                c = torch.from_numpy(c).to(self.device)
                m = torch.from_numpy(m).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez = noisez.view(
                    self.batch_size, self.random_dim + self.cond_generator.n_opt, 1, 1
                )
                perm = np.arange(self.batch_size)
                np.random.shuffle(perm)
                real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                real = torch.from_numpy(real.astype("float32")).to(self.device)
                c_perm = c[perm]
                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, self.transformer.output_info)
                fake_cat = torch.cat([fakeact, c], dim=1)
                real_cat = torch.cat([real, c_perm], dim=1)
                real_cat_d = self.Dtransformer.transform(real_cat)
                fake_cat_d = self.Dtransformer.transform(fake_cat)

                optimizerD.zero_grad()
                y_real, _ = discriminator(real_cat_d)
                y_fake, _ = discriminator(fake_cat_d)
                loss_d = -(torch.log(y_real + 1e-4).mean()) - (
                    torch.log(1.0 - y_fake + 1e-4).mean()
                )
                loss_d.backward()
                optimizerD.step()

                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                condvec = self.cond_generator.sample_train(self.batch_size)
                c, m, col, opt = condvec
                c = torch.from_numpy(c).to(self.device)
                m = torch.from_numpy(m).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez = noisez.view(
                    self.batch_size, self.random_dim + self.cond_generator.n_opt, 1, 1
                )

                optimizerG.zero_grad()
                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, self.transformer.output_info)
                fake_cat = torch.cat([fakeact, c], dim=1)
                fake_cat = self.Dtransformer.transform(fake_cat)
                y_fake, info_fake = discriminator(fake_cat)
                _, info_real = discriminator(real_cat_d)
                cross_entropy = cond_loss(faket, self.transformer.output_info, c, m)
                g = -(torch.log(y_fake + 1e-4).mean()) + cross_entropy
                g.backward(retain_graph=True)
                loss_mean = torch.norm(
                    torch.mean(info_fake.view(self.batch_size, -1), dim=0)
                    - torch.mean(info_real.view(self.batch_size, -1), dim=0),
                    1,
                )
                loss_std = torch.norm(
                    torch.std(info_fake.view(self.batch_size, -1), dim=0)
                    - torch.std(info_real.view(self.batch_size, -1), dim=0),
                    1,
                )
                loss_info = loss_mean + loss_std
                loss_info.backward()
                optimizerG.step()

                if problem_type:
                    c_loss = None
                    if (st_ed[1] - st_ed[0]) == 2:
                        c_loss = BCELoss()
                    else:
                        c_loss = CrossEntropyLoss()

                    optimizerC.zero_grad()
                    real_pre, real_label = classifier(real)
                    if (st_ed[1] - st_ed[0]) == 2:
                        real_label = real_label.type_as(real_pre)
                    loss_cc = c_loss(real_pre, real_label)
                    loss_cc.backward()
                    optimizerC.step()

                    optimizerG.zero_grad()
                    fake = self.generator(noisez)
                    faket = self.Gtransformer.inverse_transform(fake)
                    fakeact = apply_activate(faket, self.transformer.output_info)
                    fake_pre, fake_label = classifier(fakeact)
                    if (st_ed[1] - st_ed[0]) == 2:
                        fake_label = fake_label.type_as(fake_pre)
                    loss_cg = c_loss(fake_pre, fake_label)
                    loss_cg.backward()
                    optimizerG.step()

    def sample(self, n):
        self.generator.eval()
        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1
        data = []
        for _ in range(steps):
            noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
            condvec = self.cond_generator.sample(self.batch_size)
            c = condvec
            c = torch.from_numpy(c).to(self.device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez = noisez.view(
                self.batch_size, self.random_dim + self.cond_generator.n_opt, 1, 1
            )
            fake = self.generator(noisez)
            faket = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket, output_info)
            data.append(fakeact.detach().cpu().numpy())
        data = np.concatenate(data, axis=0)
        result = self.transformer.inverse_transform(data)
        return result[0:n]
