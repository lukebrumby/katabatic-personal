"""
Compare synthetic data from direct upstream run vs Katabatic pipeline run.
"""
import os
import sys
sys.path.insert(0, '/home/lbrum14/projects/Katabatic')
os.chdir('/home/lbrum14/projects/Katabatic')

import pandas as pd
import numpy as np

direct_x = pd.read_csv("verify_ctgan/direct/x_synth.csv")
direct_y = pd.read_csv("verify_ctgan/direct/y_synth.csv")
pipe_x   = pd.read_csv("verify_ctgan/pipeline/x_synth.csv")
pipe_y   = pd.read_csv("verify_ctgan/pipeline/y_synth.csv")
train_x  = pd.read_csv("sample_data/adult/x_train.csv")

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
    print(f"  {col:20s}: mean {dm:.3f}/{pm:.3f} ({mean_pct:.1f}%)  std {ds:.3f}/{ps:.3f} ({std_pct:.1f}%){flag}")
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
    print(f"  {col:20s}: max proportional diff={max_diff:.1f}%{flag}")
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
    for i in issues:
        print(f"  - {i}")
else:
    print("MATCH — synthetic data is statistically equivalent")
