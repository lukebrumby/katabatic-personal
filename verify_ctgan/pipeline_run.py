"""
Run Katabatic CTGANModel on adult dataset (train only, no TSTR).
Uses same data splits as direct run.
"""
import os
import sys
sys.path.insert(0, '/home/lbrum14/projects/Katabatic')
os.chdir('/home/lbrum14/projects/Katabatic')

from katabatic.models.ctgan.models import CTGANModel

synth_dir = "verify_ctgan/pipeline"
os.makedirs(synth_dir, exist_ok=True)

model = CTGANModel(
    epochs=100,
    batch_size=512,
    noise_dim=128,
    generator_hidden=(256, 256),
    discriminator_hidden=(256, 256),
    lr=2e-4,
    n_critic=5,
    seed=42,
    device="cpu",
    backend="torch",
)

model.train(data_dir="sample_data/adult", synthetic_dir=synth_dir)
print("Pipeline (model.train) run complete.")
print(f"Output: {synth_dir}/x_synth.csv, {synth_dir}/y_synth.csv")
