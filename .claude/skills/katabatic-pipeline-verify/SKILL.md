---
name: katabatic-pipeline-verify
description: Verify that Katabatic produces synthetic data faithful to the original upstream GitHub repository of each model by cloning the real repo, running it directly on the same data, then running the Katabatic pipeline on the same data, and comparing x_synth.csv outputs. Diagnoses and fixes any divergence. Use this skill whenever the user asks to "verify the pipeline", "check faithfulness", "compare pipeline vs original repo", "validate the implementation", or "make sure Katabatic matches the real model" for a specific model.
---

# Katabatic Pipeline Faithfulness Verification

You are verifying that the Katabatic pipeline produces synthetic data that is statistically identical to the **original upstream GitHub repository** of the model — same data, same seed, same parameters.

The TSTR evaluation classifier has no random seed, so TSTR metrics shift between runs even with identical synthetic data. **TSTR numbers are not the comparison target.** The comparison is on the synthetic data itself: column distributions, feature correlations, and class balance.

## Input

The user will specify:
- **Model name** — e.g. "copulagan", "ctgan", "tvae", "ganblr", "great", "tabddpm"
- **Dataset name** — default "adult" if not specified
- **Seed** — default 42

Project root: `/home/lbrum14/projects/Katabatic`

---

## Step 0 — Fix TSTREvaluation first

Read `katabatic/evaluate/tstr/evaluation.py`. Add `random_state=42` to every classifier so TSTR is reproducible within the pipeline:

```python
# Before
"LR": LogisticRegression(),
"MLP": MLPClassifier(),
"RF": RandomForestClassifier(),
"XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight)

# After
"LR": LogisticRegression(random_state=42, max_iter=1000),
"MLP": MLPClassifier(random_state=42),
"RF": RandomForestClassifier(random_state=42),
"XGBoost": XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
```

Apply this fix before any verification work begins.

---

## Step 1 — Find and clone the original upstream repository

Use WebSearch to find the canonical GitHub repository for the model. Search for:
- `"<model_name>" tabular data synthesis github`
- Look for the repo linked in papers, the SDV org, or the model's own README

Known starting points:
- `ctgan` / `copulagan` / `tvae` → search `sdv-dev CTGAN github` and `sdv-dev SDV github`
- `ganblr` → search `tulip-lab GANBLR github`
- `great` → search `kathrinse be_great github`
- `tabddpm` → search `yandex-research tab-ddpm github`
- `tabsyn` → search `tabsyn tabular synthesis github`
- `pategan` → search `PATEGAN synthetic data github`

Once you have the repo URL, fetch its README and any example scripts:
```
WebFetch: https://raw.githubusercontent.com/<org>/<repo>/main/README.md
WebFetch: https://raw.githubusercontent.com/<org>/<repo>/main/examples/<example>.py  (if exists)
```

Read the README to understand:
1. How to install the library from the cloned repo
2. The exact API call to train the model and generate synthetic data
3. How to set a random seed
4. What input format the model expects

Clone the repo to `/tmp/upstream_<model>/`:
```bash
git clone https://github.com/<org>/<repo>.git /tmp/upstream_<model>
```

Install it into the Poetry environment (or a venv):
```bash
cd /home/lbrum14/projects/Katabatic && poetry run pip install /tmp/upstream_<model>
```

If the upstream repo has its own `requirements.txt` or `pyproject.toml` with extra deps, install those too.

---

## Step 2 — Prepare shared data

Check if `sample_data/<dataset>/x_train.csv` already exists. If it does, use it — do not re-split. Both runs must use the exact same files.

If it does not exist:
```bash
cd /home/lbrum14/projects/Katabatic && poetry run python -c "
from utils import discretize_preprocess
from katabatic.utils.split_dataset import split_dataset
import os
os.makedirs('sample_data/<dataset>', exist_ok=True)
discretize_preprocess('raw_data/<dataset>.csv', 'discretized_data/<dataset>.csv', bins=10, strategy='uniform')
split_dataset('discretized_data/<dataset>.csv', 'sample_data/<dataset>', test_size=0.2, seed=<seed>)
"
```

Record: shape of `x_train.csv`, column names, label distribution in `y_train.csv`.

---

## Step 3 — Direct run using the upstream repo

Write a script at `/tmp/direct_run_<model>_<dataset>.py` that:
1. Imports from the **cloned repo** (not from Katabatic's wrapper)
2. Loads `sample_data/<dataset>/train_full.csv` (or `x_train.csv` + `y_train.csv` — match whatever Katabatic's `train()` loads)
3. Trains the model with the **same parameters** that Katabatic's `train()` passes to it (read `katabatic/models/<name>/models.py` to find these)
4. Sets the same random seed
5. Generates the same number of synthetic rows as the training set
6. Saves `x_synth.csv` and `y_synth.csv` to `/tmp/verify_<model>_<dataset>/direct/`

The script must use only the upstream library's public API — no Katabatic imports. Derive the exact API call from the repo's README and example code you fetched in Step 1.

Run it:
```bash
cd /home/lbrum14/projects/Katabatic && poetry run python /tmp/direct_run_<model>_<dataset>.py
```

If it fails, read the error, check the upstream repo for the correct API usage, and fix the script. Do not proceed until the direct run produces `x_synth.csv` and `y_synth.csv`.

---

## Step 4 — Pipeline run

Run the Katabatic pipeline with the same parameters, skipping TSTR for now:

Write `/tmp/pipeline_run_<model>_<dataset>.py`:
```python
import os, importlib
from katabatic.pipeline.train_test_split.pipeline import TrainTestSplitPipeline

module = importlib.import_module('katabatic.models.<name>.models')
ModelClass = getattr(module, '<ClassName>')  # read from registry.py

synth_dir = '/tmp/verify_<model>_<dataset>/pipeline'
os.makedirs(synth_dir, exist_ok=True)

pipeline = TrainTestSplitPipeline(
    model=lambda: ModelClass(<same params as direct run>,),
    evaluations=[],
    override_evaluations=True
)
pipeline.run(
    input_csv='discretized_data/<dataset>.csv',
    output_dir='sample_data/<dataset>',
    synthetic_dir=synth_dir,
    real_test_dir='sample_data/<dataset>'
)
print("Pipeline run complete.")
```

```bash
cd /home/lbrum14/projects/Katabatic && poetry run python /tmp/pipeline_run_<model>_<dataset>.py
```

---

## Step 5 — Compare synthetic data

Run this comparison script:

```python
import pandas as pd
import numpy as np

direct_x = pd.read_csv('/tmp/verify_<model>_<dataset>/direct/x_synth.csv')
direct_y = pd.read_csv('/tmp/verify_<model>_<dataset>/direct/y_synth.csv')
pipe_x   = pd.read_csv('/tmp/verify_<model>_<dataset>/pipeline/x_synth.csv')
pipe_y   = pd.read_csv('/tmp/verify_<model>_<dataset>/pipeline/y_synth.csv')
train_x  = pd.read_csv('sample_data/<dataset>/x_train.csv')

issues = []

# Column names
cols_ok = list(direct_x.columns) == list(pipe_x.columns) == list(train_x.columns)
print(f"Columns match: {cols_ok}")
if not cols_ok:
    print(f"  direct:   {list(direct_x.columns)}")
    print(f"  pipeline: {list(pipe_x.columns)}")
    print(f"  train:    {list(train_x.columns)}")
    issues.append("column name/order mismatch")

# Row count
print(f"\nRow count — direct={len(direct_x)}  pipeline={len(pipe_x)}  train={len(train_x)}")
if len(direct_x) != len(pipe_x):
    issues.append(f"row count mismatch: direct={len(direct_x)} pipeline={len(pipe_x)}")

# Numeric distributions
print("\n=== NUMERIC DISTRIBUTIONS ===")
for col in direct_x.select_dtypes(include='number').columns:
    dm, ds = direct_x[col].mean(), direct_x[col].std()
    pm, ps = pipe_x[col].mean(),   pipe_x[col].std()
    mean_pct = abs(dm - pm) / (abs(dm) + 1e-9) * 100
    std_pct  = abs(ds - ps) / (abs(ds) + 1e-9) * 100
    flag = " *** DIVERGE" if mean_pct > 5 or std_pct > 10 else ""
    print(f"  {col}: mean {dm:.3f}/{pm:.3f} ({mean_pct:.1f}%)  std {ds:.3f}/{ps:.3f} ({std_pct:.1f}%){flag}")
    if flag:
        issues.append(f"'{col}' mean={mean_pct:.1f}% std={std_pct:.1f}%")

# Categorical distributions
print("\n=== CATEGORICAL DISTRIBUTIONS ===")
for col in direct_x.select_dtypes(exclude='number').columns:
    dv = direct_x[col].value_counts(normalize=True).sort_index()
    pv = pipe_x[col].value_counts(normalize=True).sort_index()
    aligned = dv.align(pv, fill_value=0)
    max_diff = (aligned[0] - aligned[1]).abs().max() * 100
    flag = " *** DIVERGE" if max_diff > 5 else ""
    print(f"  {col}: max proportional diff={max_diff:.1f}%{flag}")
    if max_diff > 5:
        issues.append(f"'{col}' categorical diff={max_diff:.1f}%")

# Feature correlations
print("\n=== FEATURE CORRELATIONS ===")
num_cols = direct_x.select_dtypes(include='number').columns.tolist()
if len(num_cols) >= 2:
    dcorr = direct_x[num_cols].corr().values
    pcorr = pipe_x[num_cols].corr().values
    diff  = np.abs(dcorr - pcorr)
    max_d = diff.max()
    flag = " *** DIVERGE" if max_d > 0.10 else ""
    print(f"  Max diff={max_d:.4f}  Mean diff={diff.mean():.4f}{flag}")
    if max_d > 0.10:
        idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"  Worst: '{num_cols[idx[0]]}' vs '{num_cols[idx[1]]}'  direct={dcorr[idx]:.3f}  pipeline={pcorr[idx]:.3f}")
        issues.append(f"correlation max_diff={max_d:.4f}")

# Class balance
print("\n=== CLASS BALANCE ===")
dc = direct_y.iloc[:,0].value_counts(normalize=True).sort_index()
pc = pipe_y.iloc[:,0].value_counts(normalize=True).sort_index()
aligned = dc.align(pc, fill_value=0)
max_bal = (aligned[0] - aligned[1]).abs().max() * 100
print(f"  direct:   {dc.to_dict()}")
print(f"  pipeline: {pc.to_dict()}")
flag = " *** DIVERGE" if max_bal > 5 else ""
print(f"  Max class diff={max_bal:.1f}%{flag}")
if max_bal > 5:
    issues.append(f"class balance diff={max_bal:.1f}%")

# Summary
print("\n=== RESULT ===")
if issues:
    print("DIVERGE — issues found:")
    for i in issues: print(f"  - {i}")
else:
    print("MATCH — synthetic data is statistically equivalent to upstream repo output")
```

**Acceptance thresholds:**
- Numeric mean diff < 5% per column
- Numeric std diff < 10% per column
- Categorical proportional diff < 5% per column
- Correlation max diff < 0.10
- Class balance diff < 5%

---

## Step 6 — Diagnose divergence (if DIVERGE)

Work through in order, stop at first confirmed cause:

**6a. Column mismatch** — are column names/order in `x_synth.csv` different from `x_train.csv`? Check how the Katabatic wrapper extracts features from generated data vs how the direct run does it.

**6b. Label column leakage** — is the label being included in `x_synth.csv` in one path but not the other? Read the label splitting logic in both.

**6c. Discrete column detection** — does the Katabatic wrapper mark the same set of columns as categorical/discrete as the direct run? Mismatched column types cause the synthesizer to model them differently. Compare the column type detection code in `katabatic/models/<name>/models.py` with what the upstream repo's examples do.

**6d. CSV dtype drift** — does saving and reloading via CSV change dtypes (int→float, categorical→numeric)? Print `.dtypes` for both outputs and compare.

**6e. Parameter mismatch** — are the model hyperparameters (epochs, batch_size, etc.) actually the same? Log the parameters passed in the direct run vs what the Katabatic `train()` passes to the upstream library. Any silent default difference will shift distributions.

**6f. Seed not propagated** — is the upstream library's random seed being set the same way? Some libraries set the seed in the constructor, others in `fit()`. Check the upstream repo source code, not just the README.

**6g. Row count mismatch** — does the direct run generate a different number of rows? Check what `n` is passed to the upstream library's sample call in each path.

**6h. API version mismatch** — has the upstream library's API changed since the Katabatic wrapper was written? Compare the method signatures in the cloned repo against what `katabatic/models/<name>/models.py` calls. Flag any deprecated or renamed arguments.

---

## Step 7 — Fix

Once root cause is found:
1. Read the full file before editing
2. Make the minimal fix in `katabatic/models/<name>/models.py`
3. Re-run Step 4 (pipeline run only — direct run is the ground truth and does not change)
4. Re-run Step 5 (comparison)
5. Repeat until all thresholds pass

Never adjust the direct run to match the pipeline.

---

## Step 8 — TSTR reproducibility check

With synthetic data verified as matching, run the seeded TSTR on the pipeline output three times to confirm it is now reproducible:

```bash
cd /home/lbrum14/projects/Katabatic && for i in 1 2 3; do poetry run python -c "
from katabatic.evaluate.tstr.evaluation import TSTREvaluation
e = TSTREvaluation(
    synthetic_dir='/tmp/verify_<model>_<dataset>/pipeline',
    real_test_dir='sample_data/<dataset>'
)
e.evaluate()
"; done
```

All three runs must produce identical numbers. If they do not, the random_state fix in Step 0 was not fully applied — re-check evaluation.py.

---

## Final report

```
╔══════════════════════════════════════════════════════════════╗
║  KATABATIC PIPELINE VERIFY — <MODEL> / <DATASET>            ║
╚══════════════════════════════════════════════════════════════╝

UPSTREAM REPO
  Repository: <GitHub URL>
  Cloned to:  /tmp/upstream_<model>
  API version: <version or commit>

PRE-FIX
  [FIXED] TSTREvaluation classifiers: random_state=42 added

SYNTHETIC DATA COMPARISON
  Columns match:        YES / NO
  Row count match:      YES / NO  (direct=N  pipeline=N)
  Numeric distributions: MATCH / N columns diverge (worst: <col> mean=N%)
  Categorical dists:    MATCH / N columns diverge (worst: <col> N%)
  Feature correlations: MATCH / max_diff=N (worst: <col1> vs <col2>)
  Class balance:        MATCH / diff=N%
  VERDICT:              MATCH / DIVERGE

ROOT CAUSE:
  <file:line — exact cause>

FIXES APPLIED:
  <what changed and why>

TSTR (seeded, 3 runs):
  Run 1 — LR: acc=N f1=N | MLP: acc=N f1=N | RF: acc=N f1=N | XGB: acc=N f1=N
  Run 2 — LR: acc=N f1=N | MLP: acc=N f1=N | RF: acc=N f1=N | XGB: acc=N f1=N
  Run 3 — LR: acc=N f1=N | MLP: acc=N f1=N | RF: acc=N f1=N | XGB: acc=N f1=N
  Reproducible: YES / NO

STATUS: VERIFIED / NEEDS FIXES / BLOCKED
```

---

## Cleanup

```bash
rm -rf /tmp/verify_<model>_<dataset> /tmp/direct_run_<model>_<dataset>.py /tmp/pipeline_run_<model>_<dataset>.py /tmp/upstream_<model>
```

---

## Notes

- Never modify `raw_data/`, `discretized_data/`, or `Results/`
- Never modify `katabatic/evaluate/tstr/evaluation.py` beyond the random_state fix in Step 0 without user approval
- The direct run using the cloned upstream repo is always the ground truth
- If the upstream repo has changed its API since Katabatic's wrapper was written, document this as an upstream compatibility issue and flag it separately from distribution divergence
- If the existing notebook `notebooks_luke/<model>_full_test.ipynb` has already been run, read its output cells for expected TSTR score ranges before starting
- For models with stochastic training that genuinely cannot be seeded (some neural nets), use ±10% tolerance and average 3 runs
