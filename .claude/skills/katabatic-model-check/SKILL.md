---
name: katabatic-model-check
description: Check the functionality and completeness of a model implementation in the Katabatic framework using sub-agents. Use this skill whenever the user asks to verify, test, check, validate, or audit a model in Katabatic — including "does X work?", "is the model complete?", "run a check on Y", "verify the implementation", or "make sure the model is properly integrated". Trigger even for vague phrases like "test the model" or "check everything is set up" in the Katabatic context.
---

# Katabatic Model Check

You are verifying that a model is fully and correctly implemented in the Katabatic framework. Use sub-agents to run three categories of checks in parallel, then aggregate into a single pass/fail report printed to the terminal.

## Input

The user will name a model (e.g. "ctgan", "tabddpm", "forestflow"). If they don't specify which directory (models vs models_luke), check both and note any differences.

Project root: `/home/lbrum14/projects/Katabatic`

## Architecture reminder

Every Katabatic model must:
- Live in `katabatic/models/<name>/models.py` (production) and optionally `katabatic/models_luke/<name>/models.py` (Luke's working copy)
- Extend `Model` from `katabatic/models/base_model.py` (or `models_luke/base_model.py`)
- Implement `train(data_dir, synthetic_dir, ...)`, `sample(n, ...)`, `evaluate(...)`, `get_required_dependencies()`
- Set `self.is_fitted = True` after successful training
- Save `x_synth.csv` (features) and `y_synth.csv` (label, one column) into `synthetic_dir`
- Use lazy imports for heavy deps (torch, tensorflow) via `importlib.import_module` inside methods — never at module top-level
- Be registered in `katabatic/models/registry.py` under `_models`
- Have an extras entry in `pyproject.toml` under `[tool.poetry.extras]`

---

## Sub-agent assignments

Spawn all three agents in the same turn so they run in parallel.

---

### Agent 1 — Code analysis

**Scope:** `katabatic/models/<name>/` and `katabatic/models_luke/<name>/` (if present)

**Task:** Read `models.py` (and any other `.py` files in the directory) and check every item in this list. For each item, output PASS, FAIL, or WARN with a one-line explanation.

Checklist:
1. `models.py` exists in `katabatic/models/<name>/`
2. `__init__.py` exists in `katabatic/models/<name>/`
3. Class inherits from `Model` (import path: `katabatic.models.base_model.Model`)
4. `train(self, data_dir, synthetic_dir, ...)` is implemented (not just `...` or `pass`)
5. `sample(self, n, ...)` is implemented
6. `evaluate(self, ...)` is implemented
7. `get_required_dependencies(cls)` is a `@classmethod` returning a list
8. `self.is_fitted = True` appears in `train()`
9. `train()` attempts to load `train_full.csv` OR `x_train.csv` + `y_train.csv` from `data_dir`
10. `train()` saves `x_synth.csv` and `y_synth.csv` to `synthetic_dir`
11. `synthetic_dir` defaults to `synthetic/<dataset>/<model_name>/` when None
12. Column names in the saved `x_synth.csv` are aligned to `x_train.csv` (check if there's explicit reindex or column copy logic)
13. Heavy deps (torch, tensorflow, transformers) are imported lazily inside methods, not at module top-level
14. `models_luke/<name>/models.py` exists (WARN if missing, not FAIL)
15. If both exist: the two implementations are structurally consistent (same class name, same method signatures)

**Output:** Write results to a temp file `/tmp/katabatic_check_<name>_code.txt` using this format:
```
=== CODE ANALYSIS: <name> ===
[PASS] 1. models.py exists
[FAIL] 9. train() does not load from data_dir — hardcoded path found
[WARN] 14. models_luke copy missing
...
SCORE: N/15 checks passed
```

---

### Agent 2 — Config analysis

**Scope:** `katabatic/models/registry.py` and `pyproject.toml`

**Task:** Read both files and verify the model is correctly registered and its dependencies are declared. Output PASS/FAIL/WARN for each item.

Checklist:
1. Model name appears as a key in `_models` dict in `registry.py`
2. `module` value matches the actual file path (convert dots to slashes, check file exists)
3. `class` value matches the actual class name in `models.py`
4. `dependencies` list uses Python import names (e.g. `sklearn` not `scikit-learn`, `torch` not `pytorch`) — flag any that look like pip names (contain hyphens)
5. `extra` value matches the key name used in `pyproject.toml` extras
6. `pyproject.toml` has an entry under `[tool.poetry.extras]` matching the model name
7. Every package listed in the extras entry is declared as an optional dependency in `[tool.poetry.dependencies]` (i.e. has `optional = true`)
8. The `all` extra in `pyproject.toml` includes all packages from this model's extras entry

**Output:** Write to `/tmp/katabatic_check_<name>_config.txt`:
```
=== CONFIG ANALYSIS: <name> ===
[PASS] 1. registry.py entry exists
[FAIL] 7. 'some-package' listed in extras but not declared as optional dep
...
SCORE: N/8 checks passed
```

---

### Agent 3 — Runtime test

**Scope:** Runs actual Python code against the model in the Katabatic project directory

**Task:** Generate a minimal synthetic dataset, run `train()`, run `sample()`, and verify the outputs are structurally correct. The test must complete on CPU without any model-specific extras installed (if the model has a numpy/fallback path, use it; otherwise note that a full runtime test requires `poetry install -E <name>`).

Steps:

**Step 1 — Create a temp data directory** at `/tmp/katabatic_test_<name>/`:
Write these files using Python (inline script via bash):

`train_full.csv` (50 rows, 4 feature columns + 1 binary label):
```python
import pandas as pd, numpy as np, os
rng = np.random.default_rng(42)
n = 50
df = pd.DataFrame({
    'age':    rng.integers(18, 80, n),
    'income': rng.integers(20000, 120000, n),
    'score':  rng.uniform(0, 1, n).round(3),
    'region': rng.choice(['north', 'south', 'east', 'west'], n),
    'label':  rng.integers(0, 2, n),
})
os.makedirs('/tmp/katabatic_test_<name>', exist_ok=True)
df.to_csv('/tmp/katabatic_test_<name>/train_full.csv', index=False)
df.iloc[:, :-1].to_csv('/tmp/katabatic_test_<name>/x_train.csv', index=False)
df.iloc[:, -1:].to_csv('/tmp/katabatic_test_<name>/y_train.csv', index=False)
```

**Step 2 — Run train() and sample()** via `poetry run python` from the project root:
```python
import sys
sys.path.insert(0, '/home/lbrum14/projects/Katabatic')
from katabatic.models.<name>.models import <ClassName>

model = <ClassName>()
model.train(
    data_dir='/tmp/katabatic_test_<name>',
    synthetic_dir='/tmp/katabatic_test_<name>/synthetic'
)
df = model.sample(n=20)
print("SAMPLE_SHAPE:", df.shape)
print("SAMPLE_COLS:", list(df.columns))
print("IS_FITTED:", model.is_fitted)
```

Substitute the real class name from `models.py`.

**Step 3 — Check outputs:**
1. `train()` completed without raising an exception
2. `model.is_fitted` is `True`
3. `/tmp/katabatic_test_<name>/synthetic/x_synth.csv` exists
4. `/tmp/katabatic_test_<name>/synthetic/y_synth.csv` exists
5. `x_synth.csv` has the same column names as `x_train.csv`
6. `y_synth.csv` has exactly 1 column
7. `sample(20)` returned a DataFrame with 20 rows
8. No null values in `x_synth.csv` (or report count if any)
9. No null values in `y_synth.csv` (or report count if any)
10. `y_synth.csv` values are within the expected label range (0 or 1 for the test data)

**Output:** Write to `/tmp/katabatic_check_<name>_runtime.txt`:
```
=== RUNTIME TEST: <name> ===
[PASS] 1. train() completed without error
[FAIL] 5. x_synth.csv columns ['col_0','col_1',...] do not match x_train.csv columns ['age','income',...]
[WARN] 8. x_synth.csv contains 3 null values
...
SCORE: N/10 checks passed
Note: ran without model extras — install with `poetry install -E <name>` for full test
```

If `train()` raises an `ImportError` for a missing optional dep, log:
```
[WARN] train() requires extras not installed. Run: poetry install -E <name>
       Skipping runtime output checks (checks 3-10).
```
and continue checking whatever is possible.

---

## Aggregation (after all agents complete)

Read all three output files and print the final report:

```
╔══════════════════════════════════════════════════════╗
║       KATABATIC MODEL CHECK — <NAME>                 ║
╚══════════════════════════════════════════════════════╝

CODE ANALYSIS            [N/15]
  [PASS] ...
  [FAIL] ...

CONFIG ANALYSIS          [N/8]
  [PASS] ...
  [FAIL] ...

RUNTIME TEST             [N/10]
  [PASS] ...
  [WARN] ...

──────────────────────────────────────────────────────
TOTAL: N/33 checks passed

STATUS: READY / NEEDS FIXES / BLOCKED

ISSUES TO FIX:
  1. <first failing check with actionable detail>
  2. <second failing check>
  ...
```

**STATUS** rules:
- `READY` — all checks pass (WARNs are fine)
- `NEEDS FIXES` — any FAIL in code or config; runtime FAILs that are not ImportError
- `BLOCKED` — runtime raised a non-ImportError exception, or core abstract methods are missing

After printing the report, clean up temp files:
```bash
rm -rf /tmp/katabatic_test_<name> /tmp/katabatic_check_<name>_*.txt
```

## Notes

- If the user asks to check `models_luke/<name>` specifically, run Agent 1 against that path and skip Agent 2 (registry/pyproject only covers `katabatic/models/`)
- If the model class name is unknown, read `models.py` first and extract it before spawning agents
- Do not modify any project files — this is a read-only audit
- Never delete files in `raw_data/`, `discretized_data/`, or `Results/`
