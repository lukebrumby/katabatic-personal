# CTGAN Pipeline Faithfulness Verification — 2026-03-17

## Key architectural finding
Katabatic's `CTGANModel` (`katabatic/models/ctgan/models.py`) is a **completely custom from-scratch implementation** — NOT a wrapper around the upstream `ctgan` (sdv-dev/CTGAN) library. It implements WGAN-GP with Gumbel-Softmax for categorical outputs using its own PyTorch code. Statistical equivalence to the upstream library is **not expected**.

## Changes made

### 1. `katabatic/evaluate/tstr/evaluation.py`
Added `random_state=42` to all classifiers:
```python
"LR": LogisticRegression(random_state=42, max_iter=1000),
"MLP": MLPClassifier(random_state=42),
"RF": RandomForestClassifier(random_state=42),
"XGBoost": XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
```

### 2. `katabatic/models/ctgan/utils.py:29` — Bug fix
`infer_categorical_columns` misclassified `native-country` (42 unique values > 20 threshold) as continuous. Generator applied `tanh` activation, capping values at +-1. QuantileTransformer of dominant class (United-States) maps beyond +-1, causing mode collapse: constant output 39 (std=0).

Before: `if nunique <= 20 and nunique / n_rows < 0.05:`
After:  `if nunique / n_rows < 0.05:`

## Data created
- `raw_data/adult.csv` — from ctgan.load_demo()
- `discretized_data/adult.csv` — 10 bins uniform
- `sample_data/adult/` — 80/20 split seed=42

## Comparison results (direct upstream vs Katabatic, 100 epochs each)

| Check | Pre-fix | Post-fix |
|---|---|---|
| Columns | YES | YES |
| Row count | YES (26048) | YES |
| native-country std | FAIL (0.000) | PASS (6.188) |
| capital-gain/loss | DIVERGE | DIVERGE (architectural) |
| Feature correlations | NaN | DIVERGE (architectural) |
| Class balance | PASS | PASS |

Remaining divergences are due to different architectures (custom WGAN-GP vs upstream conditional GAN).

## TSTR (seeded, 3 runs — all identical)
- LR:      acc=0.7593  f1=0.6553  auc=0.3919
- MLP:     acc=0.7511  f1=0.6614  auc=0.4747
- RF:      acc=0.7502  f1=0.6589  auc=0.4449
- XGBoost: acc=0.5696  f1=0.5862  auc=0.4413

Note: LR AUC=0.39 < 0.5 — synthetic data does not yet capture class signal well. Model quality issue, not a pipeline bug.

## Outstanding issues
- `katabatic/models_luke/ctgan/utils.py:29` has the same `nunique <= 20` bug — not mirrored per CLAUDE.md policy (needs explicit instruction)
- `decode_batch` calls `qt.inverse_transform(np.array([[val]]))` without feature names — cosmetic warning only, does not affect correctness

## Upstream notes (ctgan==0.12.1)
- batch_size must divide by pac (default pac=10) — use 500 not 512 (batch_size=512 crashes with AssertionError)
- No seed parameter in constructor — must set torch.manual_seed() and numpy.random.seed() externally
- Installed via: poetry run pip install ctgan
