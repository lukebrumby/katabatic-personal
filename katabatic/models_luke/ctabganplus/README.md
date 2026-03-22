# CTAB-GAN+

CTAB-GAN+ is a conditional tabular GAN for synthesising mixed-type tabular data with improved ML utility via downstream classifier loss and Wasserstein training.

## Overview

CTAB-GAN+ extends the original CTAB-GAN (ACML 2021) with three improvements:

- **Downstream classifier loss**: a supervised classifier is jointly trained with the generator on the target column, pushing the synthetic data toward higher ML utility.
- **Wasserstein loss + SLERP gradient penalty**: replaces the original loss for more stable GAN training.
- **Novel encoder for mixed columns**: handles columns that contain both categorical modes and a continuous distribution (e.g. income with a spike at zero).

The model encodes each row of tabular data into an image-like 2-D tensor, then trains convolutional Generator and Discriminator networks on these representations.

## Algorithm

Key components from the paper:

1. **VGM (Variational Gaussian Mixture)**: each continuous column is modelled with a BayesianGaussianMixture. During transform, a column is represented as a normalised value `alpha` + one-hot mode indicator `beta`.
2. **Conditional vector**: at each training step a random categorical column and one of its values is selected. The generator is conditioned on this vector and penalised via `cond_loss` if it ignores the condition.
3. **Image reshape**: the transformed row vector is zero-padded and reshaped to `side x side` pixels where `side` is the smallest value in `[4, 8, 16, 24, 32, 64]` satisfying `side² >= dim`.
4. **Classifier loss**: an auxiliary classifier predicts the target column from generated rows. Its loss is back-propagated through the generator.

## Evaluation Metrics

From Zhao et al. (2023), evaluated on 7 datasets:

| Dataset     | Task           |
|-------------|----------------|
| Adult       | Classification |
| Covertype   | Classification |
| Credit      | Classification |
| Intrusion   | Classification |
| Loan        | Classification |
| Insurance   | Classification |
| King        | Regression     |

Reported metrics: Accuracy, F1-Score, AUC, Average Precision (classification); Average JSD, Average Wasserstein Distance, Correlation Distance (statistical).

Key results vs baselines (non-DP setting):
- Min +33.5% accuracy over all baselines
- Min +56.4% AUC over all baselines
- Outperforms CTAB-GAN by 37.1% average JSD (classification), 63.4% (regression)

## Installation

```bash
# Install project with ctabganplus extras (future)
poetry install -E ctabganplus

# Or install dependencies directly
pip install torch scikit-learn tqdm pandas numpy
```

## Quick Start

### Standalone

```python
from katabatic.models_luke.ctabganplus.models import CTABGANModel

model = CTABGANModel(epochs=150, batch_size=500)
model.train("sample_data/adult", synthetic_dir="synthetic/adult/ctabganplus")

import pandas as pd
x = pd.read_csv("synthetic/adult/ctabganplus/x_synth.csv")
y = pd.read_csv("synthetic/adult/ctabganplus/y_synth.csv")
print(x.shape, y.shape)
```

### Pipeline

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models_luke.ctabganplus.models import CTABGANModel

pipeline = TrainTestSplitPipeline(model=lambda: CTABGANModel(epochs=150))
result = pipeline.run(
    input_csv="discretized_data/adult.csv",
    output_dir="sample_data/adult",
    synthetic_dir="synthetic/adult/ctabganplus",
    real_test_dir="sample_data/adult",
)
print(result)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 150 | Training epochs |
| `batch_size` | int | 500 | Samples per mini-batch |
| `random_dim` | int | 100 | Generator latent noise dimension |
| `num_channels` | int | 64 | CNN channel count |
| `class_dim` | tuple | (256,256,256,256) | Classifier hidden layer sizes |
| `l2scale` | float | 1e-5 | Adam weight decay |
| `categorical` | list | [] | Column names treated as categorical |
| `log` | list | [] | Columns to log-transform before fitting |
| `mixed` | dict | {} | `{col: [modal_values]}` for mixed-type columns |
| `general` | list | [] | Continuous columns to encode without GMM |
| `non_categorical` | list | [] | Numeric columns to round after generation |
| `integer` | list | [] | Integer columns (rounded in inverse transform) |
| `problem_type` | dict | {} | `{"Classification": "target_col"}` or `{"Regression": "target_col"}`. Auto-detected if empty. |
| `seed` | int | 42 | Random seed |

## Limitations

- Requires PyTorch (CPU or CUDA). Tested on PyTorch >= 1.9.
- `sides` list in `ctabgan_synthesizer.py` tops out at 64; datasets with very many columns (> 4096 transformed features) will fail. Extend the list if needed.
- DataPrep uses integer indices internally for column type maps — mixing column names and indices in the same list is not supported.
- Does not support text or image columns.
- Conditional generation requires at least one categorical column; with no categorical columns the conditional vector is empty and the GAN trains unconditionally.
- GPU recommended for datasets larger than ~50 K rows or epochs > 100.

## Reference

**Paper**: CTAB-GAN+: Enhancing Tabular Data Synthesis
**Authors**: Zilong Zhao, Aditya Kunar, Robert Birke, Lydia Y. Chen
**Year**: 2023 (submitted April 2022)
**Venue**: Frontiers in Big Data
**arXiv**: https://arxiv.org/abs/2204.00401
**Original code**: https://github.com/Team-TUD/CTAB-GAN-Plus

```bibtex
@article{zhao2023ctabganplus,
  title     = {CTAB-GAN+: Enhancing Tabular Data Synthesis},
  author    = {Zhao, Zilong and Kunar, Aditya and Birke, Robert and Chen, Lydia Y.},
  journal   = {Frontiers in Big Data},
  volume    = {6},
  year      = {2023},
  doi       = {10.3389/fdata.2023.1296508},
  url       = {https://arxiv.org/abs/2204.00401}
}
```

## Model Contract

### Inputs

```
data_dir/
  train_full.csv        # preferred: all training rows + label column
  x_train.csv           # fallback: feature columns only
  y_train.csv           # fallback: single label column
```

### Outputs

```
synthetic_dir/
  x_synth.csv           # synthetic feature rows (columns match x_train.csv)
  y_synth.csv           # synthetic label column
  metadata.json         # schema, training config, column type maps
```

Column order in `x_synth.csv` is reindexed to match `x_train.csv` exactly.
