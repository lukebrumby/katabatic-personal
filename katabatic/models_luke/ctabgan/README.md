# CTAB-GAN

**CTAB-GAN** (Conditional TABular GAN) is a conditional generative adversarial network for synthesising mixed-type tabular data, handling continuous, categorical, and mixed columns via mode-specific normalisation and a convolutional image-based architecture.

## Overview

CTAB-GAN improves on CTGAN by introducing:

- **Mixed-column handling**: columns that are partly zero (or another special value) and partly continuous are split into a categorical-mode component and a continuous component, each processed separately.
- **Image-based GAN architecture**: transformed tabular rows are reshaped into square images and fed to a DCGAN-style convolutional discriminator and generator, enabling richer feature interactions.
- **Classifier auxiliary loss**: a supervised classifier on the target column is jointly trained with the generator to preserve semantic integrity between features and label.
- **Mode-specific normalisation**: continuous columns are normalised per Bayesian Gaussian Mixture mode rather than globally, reducing mode collapse.

### Key Features

- Handles categorical, continuous, and mixed column types automatically
- Convolutional (image-based) generator and discriminator
- Auxiliary classifier loss for better label fidelity
- Compatible with the Katabatic pipeline framework

## Algorithm

Given training data **X** with continuous, categorical, and mixed columns:

1. **DataPrep** — label-encode categoricals, apply log transforms, mark mixed-column special values.
2. **DataTransformer** — fit a Bayesian Gaussian Mixture (BGM) per continuous/mixed column; apply mode-specific normalisation: `x_norm = clip((x − μ_k) / (4σ_k), −0.99, 0.99)` where k is the selected BGM mode. One-hot-encode categoricals.
3. **Training** — at each step sample a conditional vector **c** (one-hot over all categorical modes), generate fake data `G(z, c)`, reshape to image, feed to discriminator alongside real data reshaped to image. Loss: `L_D = −log D(real) − log(1 − D(fake))`, `L_G = −log D(fake) + L_cond + L_info`. Auxiliary classifier `C` trained on real data; generator penalised when `C(fake)` disagrees with **c**.
4. **Sampling** — draw noise z, sample c, generate image, inverse-transform back to tabular domain.

(Notation follows Zhao et al., 2021.)

## Evaluation Metrics

From the original paper (Table 2, datasets: Adult, Census, Credit, Covtype, Intrusion):

| Metric | Description |
|--------|-------------|
| **MLE** (Machine Learning Efficacy) | Accuracy / F1 of a downstream classifier trained on synthetic, tested on real (TSTR) |
| **WD** (Wasserstein Distance) | Statistical similarity of marginal distributions |
| **JSD** (Jensen–Shannon Divergence) | Distributional similarity per column |
| **DCR** (Distance to Closest Record) | Privacy: fraction of synthetic rows closer to training than test |
| **NNDR** (Nearest-Neighbour Distance Ratio) | Privacy: ratio of distances to nearest training vs. test neighbour |

Baselines compared: CTGAN, TableGAN, MedGAN, TGAN.

## Installation

Install CTAB-GAN dependencies (PyTorch, scikit-learn, tqdm):

```bash
poetry install -E ctabgan
```

Or with pip:

```bash
pip install torch scikit-learn tqdm
```

## Quick Start

### Standalone Usage

```python
from katabatic.models_luke.ctabgan.models import CTABGANModel

model = CTABGANModel(
    epochs=150,
    batch_size=500,
    class_dim=(256, 256, 256, 256),
    random_dim=100,
    num_channels=64,
    l2scale=1e-5,
    test_ratio=0.20,
)

model.train("sample_data/adult", synthetic_dir="synthetic/adult/ctabgan")

synth = model.sample(n=1000)
print(synth.shape)
```

### Pipeline Usage (Recommended)

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models_luke.ctabgan.models import CTABGANModel

pipeline = TrainTestSplitPipeline(model=CTABGANModel)

result = pipeline.run(
    input_csv="discretized_data/adult.csv",
    output_dir="sample_data/adult",
    synthetic_dir="synthetic/adult/ctabgan",
    real_test_dir="sample_data/adult",
)
print(result)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 150 | Number of training epochs |
| `batch_size` | int | 500 | Mini-batch size |
| `class_dim` | tuple of int | (256, 256, 256, 256) | Hidden layer sizes for the auxiliary classifier |
| `random_dim` | int | 100 | Noise vector dimension fed to the generator |
| `num_channels` | int | 64 | Base channel count for convolutional layers |
| `l2scale` | float | 1e-5 | L2 weight-decay coefficient for Adam optimisers |
| `test_ratio` | float | 0.20 | Fraction of training data held out internally by DataPrep |

Defaults match `Experiment_Script_Adult.ipynb` in the original Team-TUD repository.

## Limitations

- Requires PyTorch; GPU training is auto-detected but not required.
- Column-type detection is heuristic (object dtype → categorical; int with < 20 unique values → categorical; float with > 30 % zeros → mixed). Override by subclassing if needed.
- `test_ratio` causes DataPrep to internally withhold 20 % of rows; the synthesizer trains on the remaining 80 %.
- No conditional sampling API (conditions are sampled internally during generation).
- Designed for classification targets; regression targets are not explicitly supported.
- Performance degrades on very high-cardinality categorical columns (> ~50 categories).

## Reference

**Paper**: "CTAB-GAN: Effective Table Data Synthesizing"
**Authors**: Zilong Zhao, Aditya Kunar, Robert Birke, Lydia Y. Chen
**Year**: 2021
**Venue**: Asian Conference on Machine Learning (ACML 2021)
**Link**: https://proceedings.mlr.press/v157/zhao21a.html

**Original Implementation**: https://github.com/Team-TUD/CTAB-GAN

```bibtex
@inproceedings{zhao2021ctabgan,
  title     = {{CTAB-GAN}: Effective Table Data Synthesizing},
  author    = {Zhao, Zilong and Kunar, Aditya and Birke, Robert and Chen, Lydia Y.},
  booktitle = {Proceedings of The 13th Asian Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {157},
  pages     = {97--112},
  year      = {2021},
  publisher = {PMLR}
}
```

## Model Contract

### Inputs

`data_dir` must contain one of:
- `train_full.csv` — combined features + label (preferred)
- `x_train.csv` + `y_train.csv` — separate feature and label files

### Outputs

Written to `synthetic_dir/`:

| File | Description |
|------|-------------|
| `x_synth.csv` | Synthetic features; column names match `x_train.csv` |
| `y_synth.csv` | Synthetic labels; single column with header |
| `metadata.json` | Schema, dtypes, column types, training config |

Synthetic output size equals `len(train_full)` (not downsampled).
