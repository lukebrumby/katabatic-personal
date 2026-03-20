# CopulaGAN

**CopulaGAN** generates synthetic tabular data by combining Gaussian Copula transformations with a CTGAN-style conditional GAN to better model multivariate correlations.

## Overview

CopulaGAN is part of the Synthetic Data Vault (SDV) framework. It extends CTGAN with a preprocessing step that transforms each numerical column's marginal distribution into a standard normal using a Gaussian Copula CDF mapping. The GAN then operates in this transformed space and the output is inverse-transformed back to the original scale.

This approach improves correlation fidelity for continuous features while retaining CTGAN's conditional sampling mechanism for categorical columns.

Key properties:

- Gaussian Copula preprocessing normalises numerical marginals before GAN training
- CTGAN architecture: mode-specific normalisation + conditional vector + WGAN-GP
- Inverse transformation restores synthetic values to original scale and distribution
- Configurable per-column distribution fitting (`beta`, `norm`, `truncnorm`, `uniform`, `gamma`, `gaussian_kde`)
- Compatible with the Katabatic train-test-split pipeline

## Algorithm

CopulaGAN applies the following transformation to each numerical column x:

```
x_transformed = Phi^{-1}( F_dist(x; theta) )
```

where:
- `F_dist` is the fitted CDF of the chosen univariate distribution (default: Beta)
- `Phi^{-1}` is the standard normal quantile function (probit)
- The GAN trains on `x_transformed` and generates `x_hat_transformed`
- Inverse: `x_hat = F_dist^{-1}( Phi(x_hat_transformed) )`

Categorical columns are handled by CTGAN's mode-specific conditional vector and Gumbel-Softmax.

## Evaluation Metrics

From comparative studies using CopulaGAN (SDV, 2020 onwards):

| Dataset | Metric | Notes |
|---------|--------|-------|
| Adult (income) | F1 Score | Binary classification on synthetic-trained model |
| Covertype | F1 Score | Multi-class classification |
| Credit Card | Precision / Recall / F1 / AUC / MCC | Imbalanced dataset; CopulaGAN tends to oversample majority class |
| NSL-KDD | Accuracy, F1 | Network intrusion detection |
| Air Quality, Air Pollution | KS-test, Chi-squared | Distribution fidelity metrics |

Baselines reported in CTGAN paper (CopulaGAN shares the same evaluation protocol):
- CLBN, PrivBN, TableGAN, CWGAN, MedGAN

## Installation

Install CopulaGAN dependencies:

```bash
poetry install -E copulagan
```

Or with pip:

```bash
pip install sdv
```

## Quick Start

### Standalone Usage

```python
from katabatic.models_luke.copulagan.models import CopulaGANModel

model = CopulaGANModel(
    epochs=300,
    batch_size=500,
    default_distribution="beta",
)

model.train("sample_data/car", synthetic_dir="synthetic/car/copulagan")

import pandas as pd
x = pd.read_csv("synthetic/car/copulagan/x_synth.csv")
y = pd.read_csv("synthetic/car/copulagan/y_synth.csv")
print(x.shape, y.shape)
```

### Pipeline Usage

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models_luke.copulagan.models import CopulaGANModel

pipeline = TrainTestSplitPipeline(model=CopulaGANModel)

result = pipeline.run(
    input_csv="discretized_data/car.csv",
    output_dir="sample_data/car",
    synthetic_dir="synthetic/car/copulagan",
    real_test_dir="sample_data/car",
)
print(result)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 300 | Number of GAN training epochs |
| `batch_size` | int | 500 | Training batch size (must be divisible by `pac`) |
| `embedding_dim` | int | 128 | Generator noise vector size (latent dimension) |
| `generator_dim` | tuple[int, ...] | (256, 256) | Hidden layer sizes for the Generator |
| `discriminator_dim` | tuple[int, ...] | (256, 256) | Hidden layer sizes for the Discriminator |
| `generator_lr` | float | 2e-4 | Generator Adam learning rate |
| `discriminator_lr` | float | 2e-4 | Discriminator Adam learning rate |
| `discriminator_steps` | int | 1 | Discriminator updates per generator update |
| `log_frequency` | bool | True | Log loss each epoch |
| `verbose` | bool | False | Print per-epoch loss values |
| `pac` | int | 10 | PacGAN packing factor for minibatch discrimination |
| `cuda` | bool | True | Use GPU if available |
| `numerical_distributions` | dict | {} | Per-column distribution overrides, e.g. `{"age": "norm"}` |
| `default_distribution` | str | `"beta"` | Default distribution for all numerical columns. Options: `beta`, `norm`, `truncnorm`, `uniform`, `gamma`, `gaussian_kde` |

## Limitations

- Requires SDV library (`pip install sdv`); this pulls in PyTorch and Copulas
- Imbalanced datasets: CopulaGAN may generate predominantly majority-class samples
- `gaussian_kde` distribution is slow and memory-intensive on large datasets
- CopulaGAN does not provide differential privacy guarantees; use PATEGAN for privacy-sensitive data
- All columns are included in copula fitting; very high-cardinality categorical columns may degrade performance
- GPU support depends on PyTorch CUDA availability and is best-effort

## Reference

**Foundation paper**: "The Synthetic Data Vault"
Neha Patki, Roy Wedge, Kalyan Veeramachaneni
IEEE International Conference on Data Science and Advanced Analytics (DSAA), 2016
DOI: 10.1109/DSAA.2016.49

**CTGAN paper**: "Modeling Tabular Data using Conditional GAN"
Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni
NeurIPS 2019
ArXiv: https://arxiv.org/abs/1907.00503

**Canonical implementation**: https://github.com/sdv-dev/SDV

```bibtex
@inproceedings{xu2019ctgan,
  title={Modeling Tabular Data using Conditional GAN},
  author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}

@inproceedings{patki2016sdv,
  title={The Synthetic Data Vault},
  author={Patki, Neha and Wedge, Roy and Veeramachaneni, Kalyan},
  booktitle={IEEE International Conference on Data Science and Advanced Analytics},
  year={2016},
  doi={10.1109/DSAA.2016.49}
}
```

## Model Contract

### Inputs

- `data_dir/train_full.csv`: Full training data (features + label as last column), OR
- `data_dir/x_train.csv` + `data_dir/y_train.csv`: Features and label split

### Outputs

- `synthetic_dir/x_synth.csv`: Synthetic feature columns (index=False, columns match x_train.csv)
- `synthetic_dir/y_synth.csv`: Synthetic label column (index=False)
- `synthetic_dir/metadata.json`: Schema, column types, discrete columns, training config

### Schema Fidelity

- Column order in `x_synth.csv` matches `x_train.csv` exactly
- Numerical columns are constrained to `[min, max]` of training data (`enforce_min_max_values=True`)
- Rounding applied to match decimal precision of training data (`enforce_rounding=True`)
