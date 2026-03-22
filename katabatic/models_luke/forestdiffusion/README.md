# ForestDiffusion

**ForestDiffusion** generates and imputes tabular data via diffusion and flow-based gradient-boosted tree models, without requiring a GPU.

## Overview

ForestDiffusion replaces the neural network function approximator used in score-based diffusion with XGBoost (or LightGBM / CatBoost / Random Forest). This makes it:

- Trainable entirely on CPU with no GPU required
- Faster to train than deep-learning-based methods on small-to-medium datasets
- Competitive with or superior to neural diffusion and GAN-based methods on standard benchmarks
- Capable of handling mixed-type data (continuous, binary, categorical, integer) natively

Two generation modes are supported:

- **Flow matching** (`diffusion_type='flow'`, default): conditional flow matching via learned vector fields — recommended for synthetic data generation
- **VP-SDE** (`diffusion_type='vp'`): variance-preserving SDE — also supports missing value imputation via `impute()`

Class conditioning is supported: when a label column `label_y` is provided, a separate XGBoost score model is trained per class, enabling class-conditional generation.

## Algorithm

ForestDiffusion learns a denoising function at each of `n_t` discrete noise levels. At training time, each real data point `x_0` is perturbed to noisy versions `x_t` at levels `t = 1, ..., n_t`. The model trains an XGBoost ensemble to predict the score (or velocity field) from `x_t`. At generation time, the learned score is used to denoise samples from a Gaussian prior back to the data manifold.

For flow matching, the velocity field is:

```
v_theta(x_t, t) ≈ x_0 - epsilon
```

where `epsilon ~ N(0, I)` is the injected noise. The ODE solver integrates this field from `t=1` (pure noise) to `t=0` (clean data).

For VP-SDE, the forward process follows:

```
dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW
beta(t) = beta_min + t * (beta_max - beta_min)
```

Data duplication (`duplicate_K`) generates `K` noisy copies of each training row, providing the XGBoost model with a dense coverage of the noise schedule.

## Evaluation Metrics

From the paper (AISTATS 2024, 27 datasets, 9 metrics):

**Generation metrics:**
- **Detection** — can a classifier distinguish real from synthetic? (lower = better)
- **Column-wise statistics** — marginal distribution similarity (Wasserstein distance)
- **pairwise correlations** — Pearson/Spearman correlation matrix similarity
- **TSTR** (Train on Synthetic, Test on Real) — downstream classifier accuracy
- **JSD** (Jensen-Shannon Divergence) — distributional divergence per column

**Imputation metrics:**
- **RMSE** — root mean squared error for continuous columns
- **Macro F1** — classification accuracy for categorical columns

**Comparison baselines:** SMOTE, CTGAN, TVAE, TabDDPM, STaSy, GaussianCopula, BayesianNetwork

Datasets used include: iris, wine, california housing, breast cancer, adult, credit, and 21 additional UCI/Kaggle tabular datasets.

## Installation

```bash
# Future: poetry install -E forestdiffusion
pip install ForestDiffusion
```

ForestDiffusion requires XGBoost. Optional backends: LightGBM (`pip install lightgbm`), CatBoost (`pip install catboost`).

## Quick Start

### Standalone Usage

```python
from katabatic.models_luke.forestdiffusion.models import ForestDiffusionModel

model = ForestDiffusionModel(
    diffusion_type="flow",
    n_t=50,
    duplicate_K=100,
    seed=666,
)

model.train("sample_data/car", synthetic_dir="synthetic/car/forestdiffusion")

df_synth = model.sample(n=1000)
print(df_synth.shape)
```

### Pipeline Usage

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models_luke.forestdiffusion.models import ForestDiffusionModel

pipeline = TrainTestSplitPipeline(
    model=lambda: ForestDiffusionModel(diffusion_type="flow", n_t=50)
)

result = pipeline.run(
    input_csv="discretized_data/car.csv",
    output_dir="sample_data/car",
    synthetic_dir="synthetic/car/forestdiffusion",
    real_test_dir="sample_data/car",
)
print(result)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_t` | int | 50 | Number of discrete noise levels in the diffusion schedule |
| `model` | str | `'xgboost'` | Tree backend: `'xgboost'`, `'random_forest'`, `'lgbm'`, `'catboost'` |
| `diffusion_type` | str | `'flow'` | `'flow'` (flow matching) or `'vp'` (variance-preserving SDE) |
| `max_depth` | int | 7 | Maximum depth of each tree |
| `n_estimators` | int | 100 | Number of boosting rounds per XGBoost model |
| `eta` | float | 0.3 | XGBoost learning rate |
| `tree_method` | str | `'hist'` | XGBoost tree construction algorithm |
| `reg_alpha` | float | 0.0 | L1 regularization weight |
| `reg_lambda` | float | 0.0 | L2 regularization weight |
| `subsample` | float | 1.0 | Fraction of training rows sampled per tree |
| `num_leaves` | int | 31 | LightGBM maximum number of leaves per tree |
| `duplicate_K` | int | 100 | Number of noisy copies per training row — higher = better coverage, slower training |
| `bin_indexes` | list of int | `[]` | Column indices of binary features (auto-detected if None) |
| `cat_indexes` | list of int | `[]` | Column indices of multi-category features (auto-detected if None) |
| `int_indexes` | list of int | `[]` | Column indices of integer-valued continuous features (auto-detected if None) |
| `remove_miss` | bool | False | If True, removes rows with missing values before training |
| `p_in_one` | bool | True | When feasible, trains a single XGBoost model for all predictors (faster) |
| `true_min_max_values` | list or None | None | Manual `[[min_vals], [max_vals]]` for clipping generated values |
| `eps` | float | 1e-3 | Numerical stability constant for VP-SDE |
| `beta_min` | float | 0.1 | Minimum diffusion coefficient (VP-SDE only) |
| `beta_max` | float | 8.0 | Maximum diffusion coefficient (VP-SDE only) |
| `n_z` | int | 10 | Noise samples per query for zero-shot classification |
| `n_jobs` | int | -1 | CPU cores for parallel XGBoost training (`-1` = all cores) |
| `n_batch` | int | 1 | Data streaming batch count (>1 enables memory-efficient mode) |
| `gpu_hist` | bool | False | Enable GPU histogram for XGBoost (requires CUDA) |
| `seed` | int | 666 | Random seed for reproducibility |

## Limitations

- **No GPU training by default** — designed for CPU; GPU support is optional via `gpu_hist=True`
- **Imputation requires VP-SDE** — set `diffusion_type='vp'` and call `.impute()` separately
- **Continuous features assumed** — object-typed columns are factorized to integer codes; the factorization is reversed after generation
- **No extrapolation** — generated values are clipped to observed min/max per column
- **Large `duplicate_K` increases memory** — reduce for very large datasets (>50K rows)
- **No streaming for flow matching** — `n_batch > 1` is only relevant for VP-SDE imputation

## Reference

Jolicoeur-Martineau, A., Fatras, K., & Kachman, T. (2024). Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees. In *Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS 2024)*.

arXiv: https://arxiv.org/abs/2309.09968

GitHub: https://github.com/SamsungSAILMontreal/ForestDiffusion

```bibtex
@inproceedings{jolicoeurmartineau2024forestdiffusion,
  title={Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees},
  author={Jolicoeur-Martineau, Alexia and Fatras, Kilian and Kachman, Tal},
  booktitle={Proceedings of the 27th International Conference on Artificial Intelligence and Statistics},
  year={2024},
  url={https://arxiv.org/abs/2309.09968}
}
```

## Model Contract

### Inputs

- `data_dir/train_full.csv` — preferred: combined features + label (label is last column)
- `data_dir/x_train.csv` + `data_dir/y_train.csv` — fallback if `train_full.csv` absent

### Outputs

- `synthetic_dir/x_synth.csv` — synthetic feature columns only (`index=False`)
- `synthetic_dir/y_synth.csv` — synthetic label column only (`index=False`)
- `synthetic_dir/metadata.json` — schema, column types, training hyperparameters

Column names in `x_synth.csv` match `x_train.csv` exactly.
