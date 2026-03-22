# Gaussian Copula

**Gaussian Copula** is a statistical synthetic data generation model for tabular data that models column correlations using copula theory — no neural networks required.

## Overview

The Gaussian Copula synthesizer models the joint distribution of a table by:

1. Fitting a marginal distribution to each numerical column independently
2. Transforming all values to a uniform [0, 1] space via the marginal CDFs
3. Applying an inverse Gaussian transform to obtain a multivariate normal space
4. Fitting a multivariate Gaussian to capture inter-column correlations (via the covariance matrix)
5. Sampling new rows from the multivariate Gaussian and inverting the transforms

Categorical columns are handled through an ordinal encoding step before the copula transformation.

What makes it distinct from GAN-based models:
- Exact statistical fit — no stochastic training, no loss curves
- Deterministic given a fixed seed — same input always produces same synthetic distribution
- Extremely fast: fits in seconds on any dataset size
- Fully interpretable: the fitted covariance matrix describes the learned structure
- Respects min/max bounds and rounding by construction (via `enforce_min_max_values` and `enforce_rounding`)

## Algorithm

**Sklar's Theorem** underpins copulas: any multivariate joint distribution can be written as:

```
F(x_1, ..., x_d) = C(F_1(x_1), ..., F_d(x_d))
```

where `C` is the copula and `F_i` are marginal CDFs.

For the Gaussian Copula, `C` is the Gaussian copula parameterised by correlation matrix `R`:

```
C_R(u_1, ..., u_d) = Phi_R(Phi^-1(u_1), ..., Phi^-1(u_d))
```

where `Phi` is the standard normal CDF and `Phi_R` is the multivariate Gaussian CDF with correlation `R`.

**Synthesis procedure per new row:**

```
1. Sample z ~ N(0, R)          (multivariate normal with fitted correlation)
2. u_i = Phi(z_i)              (convert to uniform via standard normal CDF)
3. x_i = F_i^{-1}(u_i)        (invert marginal CDF to recover original scale)
```

The marginal distributions `F_i` are fitted per column; available choices include:
`beta`, `gamma`, `gaussian`, `gaussian_kde`, `truncated_gaussian`, `uniform`.

## Evaluation Metrics

The original SDV paper (Patki et al., 2016) evaluates synthetic data quality using:

| Metric | Description |
|--------|-------------|
| **KL Divergence** | Per-column divergence between real and synthetic marginal distributions |
| **Correlation Distance** | Frobenius norm of the difference between real and synthetic correlation matrices |
| **ML Efficacy (TSTR)** | Accuracy of classifiers trained on synthetic, tested on real data |
| **Likelihood** | Log-likelihood of real data under the fitted synthetic model |

Datasets used: retail transactions, insurance claims, and other proprietary tabular datasets.
Comparison baselines: random row shuffling, independent column sampling, Bayesian network (PrivBayes).

## Installation

```bash
poetry install -E gaussian_copula
```

Or with pip:

```bash
pip install sdv
```

## Quick Start

### Standalone Usage

```python
from katabatic.models_luke.gaussian_copula.models import GaussianCopulaModel

model = GaussianCopulaModel(
    default_distribution="beta",
    enforce_min_max_values=True,
    enforce_rounding=True,
)

model.train("sample_data/car", synthetic_dir="synthetic/car/gaussian_copula")
df = model.sample()
print(df.shape)
```

### Pipeline Usage (Recommended)

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models_luke.gaussian_copula.models import GaussianCopulaModel

pipeline = TrainTestSplitPipeline(
    model=lambda: GaussianCopulaModel(default_distribution="beta")
)

results = pipeline.run(
    input_csv="discretized_data/car.csv",
    output_dir="sample_data/car",
    synthetic_dir="synthetic/car/gaussian_copula",
    real_test_dir="sample_data/car",
)
print(results)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_distribution` | `str` | `"beta"` | Marginal distribution to fit for numerical columns. Options: `"beta"`, `"gamma"`, `"gaussian"`, `"gaussian_kde"`, `"truncated_gaussian"`, `"uniform"` |
| `numerical_distributions` | `dict` | `{}` | Per-column distribution overrides, e.g. `{"age": "gamma"}`. Columns not listed use `default_distribution`. |
| `enforce_min_max_values` | `bool` | `True` | Clip sampled values to the observed min/max of each column. Prevents out-of-range extrapolation. |
| `enforce_rounding` | `bool` | `True` | Round sampled values to the same number of decimal places as the training data. |

### Distribution Choices

| Value | Description |
|-------|-------------|
| `"beta"` | Beta distribution — bounded, flexible shape. Good default for [0, 1]-like columns. |
| `"gamma"` | Gamma distribution — right-skewed, positive support. Good for counts or durations. |
| `"gaussian"` | Normal distribution — symmetric, unbounded. Good for standardised features. |
| `"gaussian_kde"` | Kernel density estimate — non-parametric, adapts to any shape. Slower to fit. |
| `"truncated_gaussian"` | Truncated normal — Gaussian with hard bounds. Good for bounded numerical features. |
| `"uniform"` | Uniform distribution — flat density between observed min/max. |

## Limitations

- Assumes Gaussian dependence structure: cannot model non-linear tail dependence (e.g. extreme co-movements)
- Categorical columns are ordinally encoded before copula fitting — the model does not respect category semantics
- No conditional sampling: cannot generate rows conditioned on a specific label value
- Marginal fitting degrades on very small datasets (< 100 rows per column)
- Does not model temporal or sequential structure in data

## Reference

**Paper**: "The Synthetic Data Vault"
**Authors**: Neha Patki, Roy Wedge, Kalyan Veeramachaneni
**Year**: 2016
**Venue**: IEEE International Conference on Data Science and Advanced Analytics (DSAA 2016), pp. 399–410
**DOI**: 10.1109/DSAA.2016.49

**Canonical Implementation**: https://github.com/sdv-dev/SDV (SDV library, `GaussianCopulaSynthesizer`)

```bibtex
@inproceedings{patki2016synthetic,
  title     = {The Synthetic Data Vault},
  author    = {Patki, Neha and Wedge, Roy and Veeramachaneni, Kalyan},
  booktitle = {2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA)},
  pages     = {399--410},
  year      = {2016},
  publisher = {IEEE},
  doi       = {10.1109/DSAA.2016.49}
}
```

## Model Contract

### Inputs

| File | Description |
|------|-------------|
| `data_dir/train_full.csv` | Preferred: full training set (features + label in final column) |
| `data_dir/x_train.csv` | Fallback: training features only |
| `data_dir/y_train.csv` | Fallback: training labels (single column) |

### Outputs

| File | Description |
|------|-------------|
| `synthetic_dir/x_synth.csv` | Synthetic features — column names match `x_train.csv` exactly |
| `synthetic_dir/y_synth.csv` | Synthetic labels — single column, same name as training label |
| `synthetic_dir/metadata.json` | Schema, dtypes, and training configuration |

### Schema Fidelity

- Column names in `x_synth.csv` match `x_train.csv` exactly (via reindex)
- Output size equals the number of training rows by default
- `sample(n)` returns exactly `n` rows from the fitted synthesizer
