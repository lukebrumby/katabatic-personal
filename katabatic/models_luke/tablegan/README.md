# TableGAN

GAN-based tabular data synthesiser for the Katabatic framework.

## Overview

TableGAN (Xu & Veeramachaneni, 2018) extends the DCGAN architecture to
tabular data. It trains three fully-connected networks simultaneously:

| Network | Role |
|---|---|
| **Generator** | Maps latent noise + class one-hot → synthetic feature row |
| **Discriminator** | Distinguishes real from synthetic rows |
| **Classifier** | Predicts class label from features; penalises nonsensical outputs |

Three loss terms keep the generator honest:

1. **Adversarial loss** — standard GAN cross-entropy (weighted by `alpha`)
2. **Information loss** — MSE between real and synthetic mean/variance per column (weighted by `beta`)
3. **Classification loss** — cross-entropy on the classifier's prediction of the fake row

Data is normalised to **[-1, 1]** per column before training (matching the
generator's tanh output), then denormalised when saving synthetic CSVs.

## Key Features

- Three-network GAN with auxiliary classifier to improve semantic validity
- Per-column min-max normalisation — no external scaler needed
- Conditional generation: class label one-hot-encoded and concatenated to noise
- Pure PyTorch CPU backend — no GPU required
- Katabatic pipeline compatible (`train` / `sample` / `evaluate`)
- Outputs `x_synth.csv`, `y_synth.csv`, and `metadata.json`

## Installation

```bash
poetry install -E tablegan
# or
pip install katabatic[tablegan]
```

TableGAN requires `torch` and `scikit-learn`. Both are available in the
standard Katabatic extras.

## Quick Start

### Standalone

```python
from katabatic.models_luke.tablegan.models import TableGANModel

model = TableGANModel(num_epochs=200, batch_size=500)
model.train("sample_data/adult", synthetic_dir="synthetic/adult/tablegan")

df_synth = model.sample()          # same length as training data
df_new   = model.sample(n=1000)    # arbitrary new samples
```

### Katabatic Pipeline

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models_luke.tablegan.models import TableGANModel

pipeline = TrainTestSplitPipeline(model=TableGANModel())
pipeline.run(
    input_csv="discretized_data/adult.csv",
    output_dir="sample_data/adult",
    synthetic_dir="synthetic/adult/tablegan",
    real_test_dir="sample_data/adult",
)
```

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `z_dim` | int | 100 | Latent noise dimension |
| `num_epochs` | int | 200 | Training epochs |
| `batch_size` | int | 500 | Mini-batch size |
| `learning_rate` | float | 0.0002 | Adam learning rate |
| `beta1` | float | 0.5 | Adam beta_1 momentum |
| `alpha` | float | 0.5 | Adversarial loss weight |
| `beta` | float | 0.5 | Information loss weight |
| `delta_mean` | float | 0.0 | Mean tolerance (stored in metadata) |
| `delta_var` | float | 0.0 | Variance tolerance (stored in metadata) |
| `random_state` | int | 42 | RNG seed |

## Examples

```python
# High-fidelity — more epochs, tighter loss weights
model = TableGANModel(num_epochs=500, alpha=0.7, beta=0.3)

# Fast smoke test — few epochs
model = TableGANModel(num_epochs=20, batch_size=128)

# Reproduce paper defaults exactly
model = TableGANModel(
    z_dim=100, num_epochs=200, batch_size=500,
    learning_rate=0.0002, beta1=0.5, alpha=0.5, beta=0.5,
)
```

## Model Contract (Katabatic Framework)

**Inputs** (read from `data_dir`):

| File | Required | Description |
|---|---|---|
| `train_full.csv` | preferred | Combined features + label (last column = label) |
| `x_train.csv` | fallback | Features only |
| `y_train.csv` | fallback | Label column |

**Outputs** (written to `synthetic_dir`):

| File | Description |
|---|---|
| `x_synth.csv` | Synthetic feature rows; column names match `x_train.csv` |
| `y_synth.csv` | Synthetic label column; header matches original label name |
| `metadata.json` | Schema, hyperparameters, n_classes, n_features |

**Column alignment**: after generation, `x_synth.csv` columns are reindexed
to match `x_train.csv` exactly.

**`is_fitted`**: set to `False` in `__init__`, `True` after `train()`.
Both `sample()` and `evaluate()` raise `RuntimeError` if called before training.

## Algorithm Details

### Generator

```
z (z_dim,) + y_onehot (n_classes,)
  → Linear(z_dim + n_classes, 256) → ReLU
  → Linear(256, 256) → ReLU
  → Linear(256, n_features) → Tanh   ← output in [-1, 1]
```

### Discriminator

```
x (n_features,) + y_onehot (n_classes,)
  → Linear(n_features + n_classes, 256) → LeakyReLU(0.2)
  → Linear(256, 256) → LeakyReLU(0.2)
  → Linear(256, 1) → Sigmoid
```

### Classifier

```
x (n_features,)
  → Linear(n_features, 256) → ReLU
  → Linear(256, n_classes)             ← logits
```

### Generator loss

```
L_G = alpha * BCE(D(x_fake, y), 1)
    + beta  * MSE(mean(x_fake), mean(x_real)) + MSE(var(x_fake), var(x_real))
    +         CE(C(x_fake), y)
```

## Performance Tips

- Training is CPU-only by default. For large datasets (shuttle, adult) expect
  a few minutes per 100 epochs at batch_size=500.
- Reduce `num_epochs` to 50–100 for a quick smoke test.
- Increase `batch_size` to match available RAM for faster epoch time.
- `z_dim=100` is sufficient for datasets with fewer than ~50 features.

## Limitations

- No GPU support in current implementation (CPU torch only).
- Conditional generation samples classes uniformly — does not preserve class
  imbalance from training data.
- Min-max normalisation is sensitive to outliers; extreme values can compress
  the useful range.
- Less effective on purely categorical datasets without numeric encoding.

## Reference

Xu, L. & Veeramachaneni, K. (2018). **Synthesizing Tabular Data using
Generative Adversarial Networks**. arXiv:1811.11264.
https://arxiv.org/abs/1811.11264

The same authors subsequently published **CTGAN** (arXiv:1907.00503), which
addresses mode collapse on categorical columns via conditional vector sampling
and mode-specific normalisation.

## Citation

```bibtex
@article{xu2018synthesizing,
  title   = {Synthesizing Tabular Data using Generative Adversarial Networks},
  author  = {Xu, Lei and Veeramachaneni, Kalyan},
  journal = {arXiv preprint arXiv:1811.11264},
  year    = {2018}
}
```

## License

MIT
