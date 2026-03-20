# Katabatic Status

Last updated: 2026-03-19

## Active branch
luke

## Modified files (uncommitted)
- katabatic/evaluate/tstr/evaluation.py (staged)
- katabatic/models/copulagan/models.py (staged)
- katabatic/models/ctgan/utils.py (staged)
- katabatic/models/registry.py (staged)
- katabatic/models_luke/adasyn/ (staged - binary cache files)
- katabatic/models_luke/copulagan/models.py (staged)
- pyproject.toml (staged)
- katabatic/models/base_model.py (untracked — new file)
- verify_ctgan/ (untracked — new directory)

## Completed tasks

### 2026-03-19 — ADASYN models_luke/ fixes
- Fixed downsampling bug in `train()`: output is now the full resampled set (original + synthetic minority), not truncated back to original size
- Stored `_X_resampled` / `_y_resampled` on `self` during `train()` so `sample()` does not re-fit
- Fixed `sample()` to return the full resampled DataFrame from stored arrays instead of re-running `fit_resample`
- Extracted `adjust_n_neighbors` and `build_resampled_df` into `katabatic/models_luke/adasyn/utils.py`
- Created `katabatic/models_luke/adasyn/README.md`
- `n_returned` in metadata.json now reflects true output size

## In progress
(none)

## Notes
- Run `poetry run pytest` before marking any task done
- Do not commit without explicit request
