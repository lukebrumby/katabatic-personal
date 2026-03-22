# TVAE

**TVAE** (Tabular Variational Autoencoder) generates synthetic tabular data by learning a compressed latent representation via a VAE encoder/decoder, handling both continuous and categorical columns through mode-specific normalisation.

## Overview

TVAE was introduced alongside CTGAN in the same NeurIPS 2019 paper. Unlike CTGAN, which uses adversarial training, TVAE uses the variational autoencoder objective: an encoder maps rows to a latent distribution, and a decoder reconstructs rows from samples drawn from that distribution. The training objective is the ELBO (evidence lower bound): reconstruction loss + KL divergence regularisation.

Key properties:

- VAE encoder compresses tabular rows into a Gaussian latent space
- Decoder reconstructs the original row from sampled latent vectors
- Mode-specific normalisation handles multi-modal continuous distributions
- One-hot encoding handles categorical columns
- KL divergence regularises the latent space toward a standard normal
- `loss_factor` controls the relative weight of reconstruction vs KL divergence
- Compatible with the Katabatic train-test-split pipeline

## Algorithm

TVAE applies the following training objective:

```
L_TVAE = -E[log p(x | z)] + loss_factor * KL( q(z|x) || N(0,I) )
```

where:
- `q(z|x)` is the encoder: a MLP that outputs mu and log-sigma for the latent Gaussian
- `p(x|z)` is the decoder: a MLP that reconstructs the row from sampled z
- `z ~ q(z|x)` is sampled via the reparameterisation trick: `z = mu + sigma * eps`, `eps ~ N(0,I)`
- Continuous columns are normalised per-mode using a Gaussian mixture (mode-specific normalisation, same as CTGAN)
- Categorical columns are treated as softmax outputs with cross-entropy reconstruction loss

At sampling time, `z ~ N(0,I)` is sampled directly (no encoder needed) and passed through the decoder.

## Evaluation Metrics

From the original CTGAN paper (Xu et al., NeurIPS 2019), TVAE is evaluated on:

| Dataset | Metric | Notes |
|---------|--------|-------|
| Adult (Census income) | LR / MLP / RF / Decision Tree Accuracy | TSTR: train on synthetic, test on real |
| Census-Income (KDD) | LR / MLP / RF / Decision Tree Accuracy | Large-scale version |
| Credit | LR / MLP / RF / Decision Tree Accuracy | Imbalanced; credit card fraud |
| Intrusion (NSL-KDD) | LR / MLP / RF / Decision Tree Accuracy | Network intrusion detection |
| News (Online News Popularity) | LR / MLP / RF / Decision Tree Accuracy | Regression converted to binary |

Additional metrics reported:
- Gaussian NB Accuracy
- F1 score (macro-averaged)
- F1+DP (fairness-aware variant)

Baselines: CLBN, PrivBN, MedGAN, TableGAN, CWGAN, CTGAN.
TVAE outperformed CTGAN on several datasets in the original paper.

## Installation

Install TVAE dependencies:

```bash
poetry install -E tvae
```

Or with pip:

```bash
pip install sdv
```

## Quick Start

### Standalone Usage

```python
from katabatic.models_luke.tvae.models import TVAEModel

model = TVAEModel(
    embedding_dim=128,
    compress_dims=(128, 128),
    decompress_dims=(128, 128),
    l2scale=1e-5,
    batch_size=500,
    epochs=300,
    loss_factor=2,
    cuda=True,
)

model.train("sample_data/car", synthetic_dir="synthetic/car/tvae")

import pandas as pd
x = pd.read_csv("synthetic/car/tvae/x_synth.csv")
y = pd.read_csv("synthetic/car/tvae/y_synth.csv")
print(x.shape, y.shape)
```

### Pipeline Usage

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models_luke.tvae.models import TVAEModel

pipeline = TrainTestSplitPipeline(model=TVAEModel)

result = pipeline.run(
    input_csv="discretized_data/car.csv",
    output_dir="sample_data/car",
    synthetic_dir="synthetic/car/tvae",
    real_test_dir="sample_data/car",
)
print(result)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_dim` | int | 128 | Size of the latent space (bottleneck dimension) |
| `compress_dims` | tuple[int, ...] | (128, 128) | Hidden layer sizes for the encoder MLP |
| `decompress_dims` | tuple[int, ...] | (128, 128) | Hidden layer sizes for the decoder MLP |
| `l2scale` | float | 1e-5 | L2 weight decay applied to encoder and decoder parameters |
| `batch_size` | int | 500 | Training batch size |
| `epochs` | int | 300 | Number of training epochs |
| `loss_factor` | int | 2 | Multiplier on reconstruction loss relative to KL divergence |
| `cuda` | bool | True | Use GPU if available (falls back to CPU silently) |

## Limitations

- Requires SDV library (`pip install sdv`); this pulls in PyTorch
- VAE assumes a unimodal latent prior — may struggle with highly multimodal data; mode-specific normalisation partially compensates
- Imbalanced datasets: TVAE samples from the full learned distribution and may not balance classes; combine with oversampling if needed
- Very high-cardinality categorical columns inflate the softmax output dimension and slow training
- No differential privacy guarantees; use PATEGAN for privacy-sensitive settings
- GPU support depends on PyTorch CUDA availability and is best-effort

## Reference

**Paper**: "Modeling Tabular data using Conditional GAN"
Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni
Advances in Neural Information Processing Systems (NeurIPS), 2019
ArXiv: https://arxiv.org/abs/1907.00503

**Canonical implementation**: https://github.com/sdv-dev/CTGAN

```bibtex
@inproceedings{xu2019modeling,
  title={Modeling Tabular data using Conditional GAN},
  author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
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
