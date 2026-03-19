# Katabatic — CLAUDE.md

## CRITICAL (read first)
- NEVER delete files in `raw_data/`, `discretized_data/`, or `Results/` without confirmation
- NEVER commit directly to main — always use feature branches
- Always run `poetry run pytest` before marking any task complete
- Models under `models_luke/` are Luke's working copies — do NOT auto-mirror to `models/` without being asked

## Git Remotes — READ BEFORE ANY PUSH
There are two remotes. NEVER confuse them:
- `origin`   → https://github.com/katabatic-mentorship/katabatic-mentorship-repo.git — TEAM REPO. Only push branches that are ready for the team to see.
- `personal` → https://github.com/lukebrumby/katabatic-personal.git — Luke's private repo for Colab/GPU work. Push freely here.

When pushing, always specify the remote explicitly:
  git push personal <branch>   ← safe, Luke's private repo
  git push origin <branch>     ← team repo — confirm with Luke first

## Project Overview
Katabatic is a Python framework for training generative models on tabular data and evaluating them via TSTR (Train on Synthetic, Test on Real). It provides a unified pipeline that splits raw datasets, trains a synthetic data model, generates synthetic samples, then evaluates downstream classifier performance trained on synthetic data and tested on real data. It is used for research into tabular data synthesis methods (GANs, diffusion, LLM-based, etc.).

## Stack
- Language: Python 3.11 strictly (TensorFlow constraint)
- Package manager: Poetry (`pyproject.toml`)
- Models: CTGAN, GANBLR, GReaT, TabDDPM, TabSyn, PATEGAN, and others (each as an optional extra)
- Evaluation: scikit-learn (LR, MLP, RF, XGBoost)
- Notebooks: Jupyter Lab

## Key Files & Folders
- `katabatic/models/base_model.py` — abstract `Model` base class
- `katabatic/models/registry.py` — `ModelRegistry`, loads models by string name
- `katabatic/pipeline/base_pipeline.py` — abstract `Pipeline` base class
- `katabatic/evaluate/tstr/evaluation.py` — `TSTREvaluation` (LR/MLP/RF/XGBoost TSTR)
- `katabatic/models_luke/` — Luke's in-progress model implementations
- `raw_data/` — original CSVs (do not delete)
- `discretized_data/` — preprocessed CSVs
- `sample_data/<dataset>/` — train/test splits
- `synthetic/<dataset>/<model>/` — generated synthetic CSVs
- `Results/<dataset>/` — TSTR evaluation outputs
- `dev_deps.py` — installs per-model dev dependencies
- `utils.py` — `discretize_preprocess()` and shared helpers

## Commands

```bash
# Install core dependencies only
poetry install

# Install with a specific model's extras
poetry install -E ganblr
poetry install -E ctgan
poetry install -E tabddpm
poetry install -E tabsyn
poetry install -E pategan
poetry install -E great

# Install everything
poetry install -E all

# Install model-specific dev environment (for models with their own pyproject.toml)
python dev_deps.py install <model_name>
python dev_deps.py install all
python dev_deps.py list

# Run tests
poetry run pytest

# Format
poetry run autopep8 --in-place --aggressive --aggressive <file.py>
poetry run isort .

# Lint / type-check
poetry run ruff check .
poetry run mypy katabatic/

# Clear Python cache
make clear-cache

# Launch notebooks
poetry run jupyter lab
```

## Architecture

### Package layout

```
katabatic/
├── models/               # Production models (used by main pipelines)
│   ├── base_model.py     # Abstract Model base class
│   ├── registry.py       # ModelRegistry — dynamic load by string name
│   └── <model>/          # One directory per model (ctgan, ganblr, great, etc.)
│       ├── models.py     # Model class implementation
│       └── utils.py      # Model-specific helpers
├── models_luke/          # Luke's in-progress model implementations (mirrors models/)
├── pipeline/
│   ├── base_pipeline.py          # Abstract Pipeline base class
│   ├── train_test_split/         # TrainTestSplitPipeline
│   └── cross_validation/         # CrossValidationPipeline
└── evaluate/
    ├── base_evaluation.py        # Abstract Evaluation base class
    └── tstr/evaluation.py        # TSTREvaluation — LR, MLP, RF, XGBoost
```

### Core abstractions

**`Model`** (`katabatic/models/base_model.py`) — all models implement:
- `train(data_dir, synthetic_dir, ...)` — reads CSVs from `data_dir`, saves synthetic output to `synthetic_dir`
- `sample(n, ...)` — returns a DataFrame of synthetic rows
- `evaluate(...)` — returns a float metric

**`Pipeline`** (`katabatic/pipeline/base_pipeline.py`) — orchestrates the full workflow:
1. Splits raw discretized CSV via `split_dataset` → writes `x_train.csv`, `y_train.csv`, `x_test.csv`, `y_test.csv`, `train_full.csv` into `output_dir`
2. Calls `model.train(output_dir)`
3. Runs each registered `Evaluation` (default: `TSTREvaluation`)

**`TSTREvaluation`** (`katabatic/evaluate/tstr/evaluation.py`) — loads `x_synth.csv`/`y_synth.csv` from `synthetic_dir` and `x_test.csv`/`y_test.csv` from `real_test_dir`, trains LR/MLP/RF/XGBoost classifiers on synthetic, tests on real, saves results to `Results/<dataset>/<model>_tstr.csv`.

**`ModelRegistry`** (`katabatic/models/registry.py`) — loads a model class by string name, checks dependencies, raises with install hint if missing.

### Data flow

```
raw_data/<dataset>.csv
  → utils.discretize_preprocess()
  → discretized_data/<dataset>.csv
  → TrainTestSplitPipeline.run(input_csv=..., output_dir='sample_data/<dataset>')
  → sample_data/<dataset>/{x_train, y_train, x_test, y_test, train_full}.csv
  → model.train('sample_data/<dataset>', synthetic_dir='synthetic/<dataset>/<model>')
  → synthetic/<dataset>/<model>/{x_synth, y_synth}.csv
  → TSTREvaluation → Results/<dataset>/<model>_tstr.csv
```

### Adding a new model

1. Create `katabatic/models/<name>/models.py` with a class extending `Model`
2. Implement `train()`, `sample()`, `evaluate()`
3. Register in `katabatic/models/registry.py` under `_models`
4. Add extras in `pyproject.toml` under `[tool.poetry.extras]`

Models under `katabatic/models_luke/` follow the same pattern and are Luke's working copies — mirror any changes to the corresponding entry in `katabatic/models/` when ready.

## Conventions
- Python 3.11 strictly (TensorFlow constraint)
- Models that need torch import it lazily via `importlib.import_module` and fall back gracefully if unavailable
- Synthetic output is always split into `x_synth.csv` (features) and `y_synth.csv` (label), matching the naming of real data splits
- TSTR results path is derived from `synthetic_dir`: `Results/<dataset_name>/<model_name>_tstr.csv`
- Never hardcode dataset or model names as string literals outside the registry

## Parallel Agents

Subagents are for research and planning only — NEVER use them to implement code. Implementation always runs in a single agent context window.

## Context Loading

- /plan (global)
- /execute (global)
- /handoff (global)
