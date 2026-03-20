Read `.claude/status.md` for current project state.

Key architecture:
- Models: `katabatic/models/<name>/models.py` (production), `katabatic/models_luke/<name>/` (working copies)
- Pipeline: raw_data -> discretized_data -> sample_data -> synthetic -> Results
- TSTR: `katabatic/evaluate/tstr/evaluation.py` (LR/MLP/RF/XGBoost trained on synthetic, tested on real)
- Registry: `katabatic/models/registry.py` — model lookup by string name
- Base class: `katabatic/models/base_model.py` — all models must implement train(), sample(), evaluate()

Run `git branch --show-current` and `git status --short` to get current branch and changed files.

Summarize: which models are modified, what is uncommitted, and what task is in progress per status.md.
