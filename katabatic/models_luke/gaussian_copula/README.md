# Gaussian Copula

Statistical tabular data synthesizer based on copula theory. No neural networks — models inter-column correlations via a multivariate Gaussian copula with per-column marginal distributions.

**Reference:** Patki, N., Wedge, R., & Veeramachaneni, K. (2016). *The Synthetic Data Vault.* IEEE DSAA. https://doi.org/10.1109/DSAA.2016.49

**Implementation:** SDV `GaussianCopulaSynthesizer` (`sdv` + `copulas` packages).

---

## How It Works

1. Each column's marginal distribution is fit independently (default: `beta`).
2. All columns are transformed to standard normal via inverse-CDF mapping.
3. A covariance matrix is estimated in the transformed space (captures inter-column correlations).
4. Sampling: draw from the multivariate Gaussian, apply per-column inverse transforms back to original domains.

---

## Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `default_distribution` | `str` | `"beta"` | Marginal distribution for numerical columns. Options: `"beta"`, `"gaussian"`, `"truncated_gaussian"`, `"gamma"`, `"uniform"`. |
| `numerical_distributions` | `dict` | `{}` | Per-column distribution overrides, e.g. `{"age": "gamma"}`. |
| `enforce_min_max_values` | `bool` | `True` | Clip synthetic values to observed min/max. |
| `enforce_rounding` | `bool` | `True` | Round numeric values to match real data precision. |
| `random_state` | `int \| None` | `None` | Random seed for reproducibility. |

---

## Usage

### Standalone

```python
from katabatic.models_luke.gaussian_copula.models import GaussianCopulaModel

model = GaussianCopulaModel(default_distribution="beta", random_state=42)
model.train(
    data_dir="sample_data/adult",
    synthetic_dir="synthetic/adult/gaussian_copula",
)
df_synth = model.sample(n=1000)
```

### Via Pipeline

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models_luke.gaussian_copula.models import GaussianCopulaModel

GC = lambda: GaussianCopulaModel(default_distribution="beta", random_state=42)

pipeline = TrainTestSplitPipeline(model=GC)
pipeline.run(
    input_csv="discretized_data/adult.csv",
    output_dir="sample_data/adult",
    synthetic_dir="synthetic/adult/gaussian_copula",
    real_test_dir="sample_data/adult",
)
```

---

## Output Files

| File | Description |
|---|---|
| `synthetic/<dataset>/gaussian_copula/x_synth.csv` | Synthetic features (columns match `x_train.csv`) |
| `synthetic/<dataset>/gaussian_copula/y_synth.csv` | Synthetic label column |
| `synthetic/<dataset>/gaussian_copula/metadata.json` | Training config, column names, distribution settings |

---

## Install

```bash
pip install sdv copulas
```

Or via Poetry extras (once registered):
```bash
poetry install -E gaussian_copula
```

---

## Strengths & Limitations

**Strengths:**
- Very fast — fits in seconds even on large datasets.
- No mode collapse; stable fitting.
- Preserves marginal distributions and global correlations.
- Fully interpretable (no black-box training).

**Limitations:**
- Assumes Gaussian dependency structure — may miss nonlinear inter-column relationships.
- Performance on downstream ML tasks is typically lower than neural methods (CTGAN, TabDDPM) on complex datasets.
