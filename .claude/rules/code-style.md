# Code Style — Katabatic

- Python 3.11 strictly (TensorFlow constraint) — do not use 3.12+ syntax
- Heavy deps (torch, tensorflow, transformers): import lazily inside methods via `importlib.import_module`, never at module top-level
- Synthetic output: always two files — `x_synth.csv` (features only) and `y_synth.csv` (label column only)
- Column names in `x_synth.csv` must match `x_train.csv` exactly — use explicit reindex or column copy logic
- `self.is_fitted = True` must be set after successful training
- TSTR results path is derived from `synthetic_dir` — never hardcode it
- Never hardcode dataset or model names as string literals outside the registry
- `synthetic_dir` defaults to `synthetic/<dataset>/<model_name>/` when None
