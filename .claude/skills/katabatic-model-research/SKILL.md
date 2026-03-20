---
name: katabatic-model-research
description: Research new tabular data synthesis models for potential integration into the Katabatic framework. Use this skill whenever the user asks to research, find, explore, or evaluate any new or candidate model for Katabatic — including when they name a specific model they want to assess (e.g. "can we add REaLTabFormer?"), ask "what's the state of the art in tabular synthesis?", mention wanting to "add a model", or ask how a model would fit into the pipeline. Trigger even for vague phrases like "look into X" or "check if Y works" in the Katabatic context.
---

# Katabatic Model Research

You are researching a new tabular data synthesis model for potential integration into Katabatic. Produce a complete research report and integration plan, printed to the terminal.

## What Katabatic is

Katabatic is a Python 3.11 framework for generative tabular data synthesis and TSTR (Train on Synthetic, Test on Real) evaluation.

**Hard constraint:** Python `>=3.11,<3.12` due to TensorFlow 2.19.

### Existing registered models
`ganblr`, `great`, `tabsyn`, `tabddpm`, `pategan`, `ctgan`

Also implemented but not yet registered: `adasyn`, `copulagan`, `codi`, `ctabgan`, `ctabganplus`, `forestdiffusion`, `forestflow`, `gaussian_copula`, `medgan`, `smote`, `smotenc`, `tablegan`, `tvae`

### Core abstractions

Every model extends `Model` from `katabatic/models_luke/base_model.py` and implements:

- `train(data_dir, synthetic_dir, ...)` — reads `train_full.csv` (or `x_train.csv` + `y_train.csv`) from `data_dir`, saves `x_synth.csv` and `y_synth.csv` to `synthetic_dir`
- `sample(n, ...)` — returns a `pd.DataFrame` of n synthetic rows
- `evaluate(...)` — returns a float metric
- `get_required_dependencies()` — class method returning list of pip import names

### Key invariants

- `synthetic_dir` defaults to `synthetic/<dataset>/<model_name>/` if None
- Output is always two files: `x_synth.csv` (features only) and `y_synth.csv` (label column only)
- Column names in `x_synth.csv` must match `x_train.csv` exactly
- `self.is_fitted = True` must be set after successful training
- Heavy imports (torch, TF) must be done lazily inside methods via `importlib.import_module`, not at module level

### Current dependency stack (pyproject.toml)

Available optional deps: `tensorflow==2.19.*`, `torch>=2.7.1`, `transformers>=4.53.2`, `scipy>=1.13,<1.15`, `numpy>=1.24`, `pandas>=2.3.1`, `scikit-learn>=1.3`, `xgboost>=3.0.2`, `tqdm`, `category-encoders`, `rtdl_revisiting_models`

---

## Research process

If the user names a specific model, research it directly and skip discovery. If they ask generally ("what's new?", "what should we add?"), find 2–3 high-impact recent models first, then report on each.

### Step 1: Gather information

Use WebSearch and WebFetch. Good starting points: arxiv, Papers with Code, GitHub. Search terms: `tabular data synthesis 2024 2025`, `tabular generative model benchmark site:github.com`.

For each candidate model, collect:

- **Paper**: title, venue (NeurIPS/ICML/ICLR/etc.), year
- **GitHub repo**: URL, stars, last commit date, license
- **Core approach**: one sentence on the method (GAN, diffusion, VAE, flow, LLM fine-tuning, etc.)
- **Dependencies**: exact Python packages required, including version constraints
- **Python 3.11 support**: does it publish 3.11 wheels? Does it use APIs that broke in 3.11/3.12?

Do not guess URLs. If you cannot find a repo or paper, say so.

### Step 2: Assess feasibility

Score each area and give a one-line rationale:

**Python 3.11 compatibility** — pass / fail / uncertain
- TensorFlow 2.19 is installed; any model requiring TF <2.x will conflict
- PyTorch >=2.7.1 is available
- Flag any dep without 3.11 wheels

**Dependency conflict risk** — low / medium / high
- Check against existing extras in `pyproject.toml`
- Flag version pins that conflict (e.g. numpy <1.24, scipy >=1.15)

**Licensing** — permissive / copyleft / unclear
- MIT, Apache 2.0: safe
- GPL: Katabatic would inherit GPL if distributed — flag it
- "Research only" / "non-commercial": blocks production use

**Integration complexity** — low / medium / high
- Low: wraps a pip-installable library with a clean fit/sample API
- Medium: needs custom preprocessing, submodules, or significant glue code
- High: requires its own config files, external binaries, or non-standard data formats

**GPU requirement** — yes / no / optional
- Katabatic CI runs on CPU; flag any model that cannot train on CPU at all

### Step 3: Integration plan (for each feasible model)

Produce exact code for all four touch points.

#### 3a. Model class skeleton

Path: `katabatic/models/<name>/models.py`

```python
from __future__ import annotations
import os
import importlib
from typing import Optional
import pandas as pd
from katabatic.models.base_model import Model


class <ClassName>(Model):

    def __init__(self, ...):
        super().__init__()
        # hyperparameters here

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["<import_name1>", "<import_name2>"]

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        **kwargs,
    ) -> "<ClassName>":
        # 1. Load: prefer train_full.csv, else x_train.csv + y_train.csv
        # 2. Fit the upstream model (import lazily via importlib)
        # 3. Call self.sample() and split into x_synth / y_synth
        # 4. Save to synthetic_dir (default: synthetic/<dataset>/<name>/)
        # 5. Ensure column names in x_synth match x_train.csv
        self.is_fitted = True
        return self

    def sample(self, n: Optional[int] = None, **kwargs) -> pd.DataFrame:
        # Return full DataFrame with all columns (features + label)
        ...

    def evaluate(self, **kwargs) -> float:
        return 0.0
```

Explain any non-obvious adaptation: e.g. if the upstream model uses a fit/transform API instead of a Katabatic-style train loop, show the mapping explicitly.

#### 3b. Registry entry

Add to `_models` dict in `katabatic/models/registry.py`:

```python
'<name>': {
    'module': 'katabatic.models.<name>.models',
    'class': '<ClassName>',
    'dependencies': ['<import_name1>'],   # Python import names, not pip names
    'extra': '<name>',
},
```

#### 3c. pyproject.toml changes

New optional dep declarations (if not already present):
```toml
<package> = {version = "^x.y.z", optional = true}
```

New extras entry:
```toml
<name> = ["<package1>", "<package2>"]
```

Updated `all` extra — show the full new list with the additions highlighted.

#### 3d. models_luke copy

Remind the user: create `katabatic/models_luke/<name>/` as a working copy following the same structure. Do not mirror to `katabatic/models/` automatically.

---

### Step 4: Comparison table

Produce a markdown table comparing candidate models against key existing ones (CTGAN, TabDDPM, GReaT as reference points):

| Model | Method | Mixed types | CPU training | TSTR vs CTGAN | Unique strength |
|-------|--------|-------------|--------------|---------------|-----------------|

Use published benchmark results where available; otherwise note "unknown from literature".

---

## Output format

Print to terminal in this structure:

```
=== KATABATIC MODEL RESEARCH REPORT ===
Date: <today>
Models researched: <list>

--- MODEL: <Name> ---
Paper:    <title, venue, year>
Repo:     <url> (<N> stars, last commit <date>, <license>)
Approach: <one sentence>

FEASIBILITY
  Python 3.11:          pass/fail/uncertain — <rationale>
  Dependency risk:      low/medium/high — <rationale>
  Licensing:            permissive/copyleft/unclear — <rationale>
  Integration effort:   low/medium/high — <rationale>
  GPU required:         yes/no/optional
  Verdict: RECOMMENDED / POSSIBLE / NOT RECOMMENDED

INTEGRATION PLAN
  [code snippets for 3a, 3b, 3c, 3d]

--- MODEL: <Next> ---
  ...

COMPARISON TABLE
  [markdown table]

RECOMMENDATION
  Priority order with one-line justification per model.

=== END OF REPORT ===
```

If only one model was researched, omit the RECOMMENDATION section.
