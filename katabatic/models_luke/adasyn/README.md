# ADASYN

**ADASYN** (Adaptive Synthetic Sampling) is a density-adaptive KNN oversampling algorithm for imbalanced classification datasets.

## Overview

ADASYN generates synthetic minority class samples using KNN interpolation, weighted by local class difficulty. Unlike SMOTE (which distributes new samples uniformly), ADASYN generates more samples near minority instances that are surrounded by majority neighbours — the harder examples get proportionally more attention.

Key distinctions from deep generative models (CTGAN, GReaT, TabDDPM):
- No learned distribution — no encoder, decoder, or latent space
- Generates minority class samples only, not a full synthetic dataset
- Output is the full resampled set: original samples + newly-generated minority samples
- Works on continuous numerical features only

In the Katabatic pipeline, the TSTR synthetic dataset is this full resampled set.

## Algorithm

For each minority class sample `x_i`, a density ratio `r_i` is computed as the proportion of majority neighbours among its K nearest neighbours. Samples with higher `r_i` receive more synthetic samples. Each new sample is created by interpolation:

```
x_new = x_i + λ * (x_zi - x_i)
```

where `x_zi` is a randomly selected minority neighbour of `x_i`, and `λ ~ Uniform(0, 1)`.

## Evaluation Metrics (from paper)

The original ADASYN paper (He et al., 2008) reports results across five metrics:

| Metric | Description |
|--------|-------------|
| **G-mean** | `sqrt(TPR * TNR)` — primary metric for imbalanced evaluation |
| **AUC** | Area under the ROC curve |
| **F-measure** | Harmonic mean of precision and recall (F1) |
| **Accuracy** | Overall classification accuracy (baseline) |
| **ROC analysis** | Full curve comparison across thresholds |

Comparison baselines in the paper: SMOTE, random oversampling, no oversampling.

## Installation

```bash
# Via imbalanced-learn (canonical reference implementation)
pip install imbalanced-learn

# Future: Katabatic extras (not yet registered)
# poetry install -E adasyn
```

## Quick Start

### Standalone usage

```python
from katabatic.models_luke.adasyn.models import ADASYNModel

model = ADASYNModel(
    sampling_strategy='auto',  # balance minority to majority
    n_neighbors=5,
    random_state=42,
)

model.train('sample_data/car', synthetic_dir='synthetic/car/adasyn')

# Retrieve the full resampled DataFrame
df = model.sample()
print(df.shape)
```

### Pipeline usage

```python
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline
from katabatic.models_luke.adasyn.models import ADASYNModel

pipeline = TrainTestSplitPipeline(
    model=ADASYNModel,
    evaluations=None  # uses default TSTREvaluation
)

pipeline.run(
    input_csv='discretized_data/car.csv',
    output_dir='sample_data/car',
    synthetic_dir='synthetic/car/adasyn',
)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampling_strategy` | float / str / dict | `'auto'` | Which classes to resample. `'auto'` balances minority to majority. |
| `n_neighbors` | int | `5` | KNN neighbours for local density estimation. Auto-adjusted if smaller than smallest class size. |
| `random_state` | int | `42` | Reproducibility seed. |

## Limitations

- Continuous numerical features only — does not natively handle categorical columns
- Generates minority class samples only — not a full unconditional generative model
- Cannot extrapolate beyond the convex hull of existing minority training samples
- No GPU acceleration — runs on CPU via scikit-learn KNN
- `n_neighbors` must be less than the size of the smallest class (auto-adjusted if violated)

## Model Contract (Katabatic Framework)

### Inputs

- `data_dir/train_full.csv` (preferred) or `data_dir/x_train.csv` + `data_dir/y_train.csv`

### Outputs

- `synthetic_dir/x_synth.csv` — feature columns of the full resampled set
- `synthetic_dir/y_synth.csv` — label column of the full resampled set
- `synthetic_dir/metadata.json` — schema, config, and sample counts

### Schema fidelity

- Column names in `x_synth.csv` match `x_train.csv` exactly
- Output size is `n_original + n_synthetic_minority` (not truncated back to original size)

## Reference

**Paper:** ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning
**Authors:** Haibo He, Yang Bai, Edwardo A. Garcia, Shutao Li
**Year:** 2008
**Venue:** IEEE IJCNN 2008, pp. 1322–1328
**DOI:** 10.1109/IJCNN.2008.4633969

**Canonical implementation:** https://github.com/scikit-learn-contrib/imbalanced-learn (`imblearn.over_sampling.ADASYN`)

```bibtex
@inproceedings{he2008adasyn,
  title={ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning},
  author={He, Haibo and Bai, Yang and Garcia, Edwardo A. and Li, Shutao},
  booktitle={IEEE International Joint Conference on Neural Networks (IJCNN)},
  pages={1322--1328},
  year={2008},
  doi={10.1109/IJCNN.2008.4633969}
}
```

## Notes

ADASYN is a **comparison baseline** for imbalanced datasets, not a deep generative model. It is categorised separately from CTGAN, GReaT, and TabDDPM in the Katabatic framework. Use it to establish a simple oversampling baseline before benchmarking more complex models.
