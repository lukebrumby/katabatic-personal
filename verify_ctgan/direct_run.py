"""
Direct run using upstream sdv-dev/CTGAN (ctgan==0.12.1).
Mirrors Katabatic's column treatment: int columns with nunique<=20 and pct<0.05 -> discrete.
"""
import os
import sys
sys.path.insert(0, '/home/lbrum14/projects/Katabatic')
os.chdir('/home/lbrum14/projects/Katabatic')

import numpy as np
import pandas as pd
import torch

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

from ctgan import CTGAN

train_full = pd.read_csv("sample_data/adult/train_full.csv")
print("Loaded train_full.csv  shape:", train_full.shape)

# Mirror Katabatic's infer_categorical_columns logic
n_rows = len(train_full)
discrete_columns = []
for col in train_full.columns:
    s = train_full[col]
    dt = str(s.dtype)
    if dt == "object" or dt.startswith("category"):
        discrete_columns.append(col)
    elif dt.startswith("int"):
        nunique = s.nunique(dropna=True)
        if nunique <= 20 and nunique / n_rows < 0.05:
            discrete_columns.append(col)

print("Discrete columns:", discrete_columns)

ctgan_upstream = CTGAN(
    epochs=100,
    embedding_dim=128,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    generator_lr=2e-4,
    discriminator_lr=2e-4,
    batch_size=500,   # default; must be divisible by pac=10
    discriminator_steps=1,
    pac=10,
    verbose=True,
    enable_gpu=False,
)

print("\nFitting upstream CTGAN (100 epochs)...")
ctgan_upstream.fit(train_full, discrete_columns)

print("\nSampling", len(train_full), "rows...")
synth = ctgan_upstream.sample(len(train_full))
print("Synthetic shape:", synth.shape)

label_col = train_full.columns[-1]
x_synth = synth[train_full.columns[:-1]].copy()
y_synth = synth[[label_col]].copy()

out_dir = "verify_ctgan/direct"
os.makedirs(out_dir, exist_ok=True)
x_synth.to_csv(os.path.join(out_dir, "x_synth.csv"), index=False)
y_synth.to_csv(os.path.join(out_dir, "y_synth.csv"), index=False)
print(f"\nSaved to {out_dir}/")
print("Direct run complete.")
