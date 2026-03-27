# 2026-03-27 — TableGAN Implementation

## Task
Implement `katabatic/models_luke/tablegan/` — a GAN-based tabular data synthesiser based on:
Xu & Veeramachaneni (2018). *Synthesizing Tabular Data using Generative Adversarial Networks*. arXiv:1811.11264.

## Research findings (parallel agents)

**Paper:**
- DCGAN-style 3-network architecture: Generator, Discriminator, Classifier
- Generator loss = alpha * adversarial + beta * information (mean/var MSE) + classification
- Min-max normalisation to [-1, 1]; generator uses tanh output
- Predecessor to CTGAN; same MIT LIDS authors; weaker on high-cardinality categoricals

**GitHub (mahmoodm2/tableGAN):**
- Original uses TF1 session API; implementation here uses PyTorch (more portable)
- Key hyperparams: `z_dim=100`, `num_epochs=200`, `batch_size=500`, `lr=0.0002`, `beta1=0.5`, `alpha=0.5`, `beta=0.5`

## Files created

| File | Status |
|---|---|
| `katabatic/models_luke/tablegan/utils.py` | Created |
| `katabatic/models_luke/tablegan/models.py` | Created |
| `katabatic/models_luke/tablegan/__init__.py` | Already existed (correct) |
| `katabatic/models_luke/tablegan/README.md` | Already existed (correct) |
| `notebooks_luke/new_examples/tablegan.ipynb` | Already existed (correct) |

## Design decisions

1. **PyTorch over TensorFlow 1.x**: original repo uses TF sessions; PyTorch is cleaner, already used by CTGAN, and avoids TF1 compatibility issues.

2. **Networks defined as inner classes inside `train()`**: closures capture `z_dim`, `n_features`, `n_classes`, `torch`, `nn` from the enclosing scope. This allows lazy import of torch while still defining proper `nn.Module` subclasses. The Generator instance survives `train()` returning because it's stored on `self._generator`.

3. **Uniform class sampling in generation**: `_generate_samples` samples class labels uniformly (not proportionally). The paper does not specify class-proportional sampling; uniform is the simpler default and matches the original repo's unconditional-style generation.

4. **Column alignment**: after generation, `x_synth.csv` columns are reindexed to `x_train.csv` column names, matching the CTGAN convention.

5. **`evaluate()` stub**: returns 0.0. Full TSTR runs via `TSTREvaluation` in the pipeline. Consistent with other models_luke implementations.

## Verification

```
import OK
is_fitted: False -> True after train()
x_synth shape: (200, 2), cols: ['a', 'b']  ✓
y_synth cols: ['label']  ✓
sample() shape: (200, 3)  ✓
sample(50) shape: (50, 3)  ✓
evaluate(): 0.0  ✓
sample() before train -> RuntimeError  ✓
evaluate() before train -> RuntimeError  ✓
x/y fallback path  ✓
poetry run pytest: no tests (expected — no test suite exists)
```

## Risk: low

Rollback: delete `katabatic/models_luke/tablegan/models.py` and `utils.py`. No other files modified.
