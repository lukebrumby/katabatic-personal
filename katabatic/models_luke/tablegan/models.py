"""
TableGAN — GAN-based tabular data synthesiser.

Reference:
    Xu, L. & Veeramachaneni, K. (2018). Synthesizing Tabular Data using
    Generative Adversarial Networks. arXiv:1811.11264.

Architecture: Generator + Discriminator + Classifier (DCGAN-style, fully-connected).
Three loss components: adversarial, information (mean/variance), classification.
Data normalised to [-1, 1] with per-column min-max scaling.
"""

import importlib
import os
from typing import Optional

import numpy as np
import pandas as pd

from katabatic.models.base_model import Model
from katabatic.models_luke.tablegan.utils import (
    MinMaxNormalizer,
    load_training_data,
    save_metadata,
)


class TableGANModel(Model):
    """Tabular GAN (tableGAN) synthetic data generator.

    Parameters
    ----------
    z_dim : int
        Dimension of the latent noise vector fed to the generator. Default: 100.
    num_epochs : int
        Number of full passes over the training data. Default: 200.
    batch_size : int
        Mini-batch size for all three networks. Default: 500.
    learning_rate : float
        Adam optimiser learning rate. Default: 0.0002.
    beta1 : float
        Adam beta_1 momentum parameter. Default: 0.5.
    alpha : float
        Weight of the adversarial loss term in the generator objective. Default: 0.5.
    beta : float
        Weight of the information loss term (mean + variance matching). Default: 0.5.
    delta_mean : float
        Tolerance for the mean constraint (informational; stored in metadata). Default: 0.0.
    delta_var : float
        Tolerance for the variance constraint (informational; stored in metadata). Default: 0.0.
    random_state : int
        Seed for torch and numpy RNGs. Default: 42.
    """

    def __init__(
        self,
        z_dim: int = 100,
        num_epochs: int = 200,
        batch_size: int = 500,
        learning_rate: float = 0.0002,
        beta1: float = 0.5,
        alpha: float = 0.5,
        beta: float = 0.5,
        delta_mean: float = 0.0,
        delta_var: float = 0.0,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.alpha = alpha
        self.beta = beta
        self.delta_mean = delta_mean
        self.delta_var = delta_var
        self.random_state = random_state

        # Populated during train()
        self._generator = None
        self._normalizer: MinMaxNormalizer | None = None
        self._label_encoder = None
        self._n_features: int | None = None
        self._n_classes: int | None = None
        self._x_columns: list | None = None
        self._label_column: str | None = None
        self._column_names: list | None = None
        self._device = None
        # Cached from last train() call — returned by sample() when n is None
        self._X_synth: np.ndarray | None = None
        self._y_synth: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Framework contract
    # ------------------------------------------------------------------

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return ["torch", "sklearn"]

    def train(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        **kwargs,
    ) -> "TableGANModel":
        """Train TableGAN on split tabular data.

        Parameters
        ----------
        data_dir : str
            Directory containing ``train_full.csv`` (or ``x_train.csv`` +
            ``y_train.csv``).
        synthetic_dir : str, optional
            Where to write ``x_synth.csv``, ``y_synth.csv``, and
            ``metadata.json``.  Defaults to
            ``synthetic/<dataset_name>/tablegan``.

        Returns
        -------
        self
        """
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        from sklearn.preprocessing import LabelEncoder

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # ---- load data ------------------------------------------------
        df, x_columns, label_col = load_training_data(data_dir)
        self._x_columns = x_columns
        self._label_column = label_col
        self._column_names = x_columns + [label_col]

        X = df[x_columns].values.astype(np.float32)
        y = df[label_col].values

        le = LabelEncoder()
        y_enc = le.fit_transform(y).astype(np.int64)
        self._label_encoder = le
        n_classes = len(le.classes_)
        n_features = X.shape[1]
        self._n_classes = n_classes
        self._n_features = n_features

        # ---- normalise X to [-1, 1] -----------------------------------
        normalizer = MinMaxNormalizer()
        X_norm = normalizer.fit_transform(X)
        self._normalizer = normalizer

        # ---- build networks (closures capture n_features, n_classes, z_dim) ----
        z_dim = self.z_dim

        class Generator(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.net = nn.Sequential(
                    nn.Linear(z_dim + n_classes, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, n_features),
                    nn.Tanh(),
                )

            def forward(self_, z, y_oh):
                return self_.net(torch.cat([z, y_oh], dim=1))

        class Discriminator(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.net = nn.Sequential(
                    nn.Linear(n_features + n_classes, 256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, 256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, 1),
                    nn.Sigmoid(),
                )

            def forward(self_, x, y_oh):
                return self_.net(torch.cat([x, y_oh], dim=1))

        class Classifier(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.net = nn.Sequential(
                    nn.Linear(n_features, 256),
                    nn.ReLU(),
                    nn.Linear(256, n_classes),
                )

            def forward(self_, x):
                return self_.net(x)

        device = torch.device("cpu")
        self._device = device

        G = Generator().to(device)
        D = Discriminator().to(device)
        C = Classifier().to(device)
        self._generator = G

        opt_G = torch.optim.Adam(
            G.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999)
        )
        opt_D = torch.optim.Adam(
            D.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999)
        )
        opt_C = torch.optim.Adam(
            C.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999)
        )

        bce = nn.BCELoss()
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()

        # ---- tensors & DataLoader ------------------------------------
        X_t = torch.FloatTensor(X_norm).to(device)
        y_t = torch.LongTensor(y_enc).to(device)
        y_oh_full = torch.zeros(len(y_enc), n_classes, device=device)
        y_oh_full.scatter_(1, y_t.unsqueeze(1), 1.0)

        dataset = torch.utils.data.TensorDataset(X_t, y_oh_full, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        # ---- training loop -------------------------------------------
        for epoch in range(self.num_epochs):
            for X_real, y_oh_batch, y_batch in loader:
                bsz = X_real.size(0)
                ones = torch.ones(bsz, 1, device=device)
                zeros = torch.zeros(bsz, 1, device=device)

                # Discriminator step
                opt_D.zero_grad()
                z = torch.randn(bsz, self.z_dim, device=device)
                X_fake = G(z, y_oh_batch).detach()
                d_loss = bce(D(X_real, y_oh_batch), ones) + bce(
                    D(X_fake, y_oh_batch), zeros
                )
                d_loss.backward()
                opt_D.step()

                # Classifier step
                opt_C.zero_grad()
                c_loss = ce(C(X_real), y_batch)
                c_loss.backward()
                opt_C.step()

                # Generator step
                opt_G.zero_grad()
                z = torch.randn(bsz, self.z_dim, device=device)
                X_fake = G(z, y_oh_batch)

                g_adv = bce(D(X_fake, y_oh_batch), ones)
                info = mse(X_fake.mean(0), X_real.mean(0)) + mse(
                    X_fake.var(0), X_real.var(0)
                )
                g_cls = ce(C(X_fake), y_batch)
                g_loss = self.alpha * g_adv + self.beta * info + g_cls
                g_loss.backward()
                opt_G.step()

            if (epoch + 1) % 50 == 0:
                print(
                    f"[TableGAN] Epoch {epoch + 1}/{self.num_epochs}  "
                    f"d={d_loss.item():.4f}  g={g_loss.item():.4f}"
                )

        # ---- generate + save ----------------------------------------
        n_samples = len(df)
        X_s_norm, y_s_enc = self._generate_samples(n_samples, torch)
        X_s = normalizer.inverse_transform(X_s_norm)
        y_s = le.inverse_transform(y_s_enc)
        self._X_synth = X_s
        self._y_synth = y_s

        if synthetic_dir is None:
            dataset_name = os.path.basename(os.path.normpath(data_dir))
            synthetic_dir = os.path.join("synthetic", dataset_name, "tablegan")
        os.makedirs(synthetic_dir, exist_ok=True)

        x_synth_df = pd.DataFrame(X_s, columns=x_columns)
        real_x_path = os.path.join(data_dir, "x_train.csv")
        if os.path.exists(real_x_path):
            real_cols = pd.read_csv(real_x_path, nrows=0).columns.tolist()
            if len(real_cols) == x_synth_df.shape[1]:
                x_synth_df.columns = real_cols
                x_synth_df = x_synth_df.reindex(columns=real_cols)

        y_synth_df = pd.DataFrame({label_col: y_s})

        x_synth_df.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
        y_synth_df.to_csv(os.path.join(synthetic_dir, "y_synth.csv"), index=False)

        save_metadata(
            os.path.join(synthetic_dir, "metadata.json"),
            x_columns=x_columns,
            label_col=label_col,
            n_classes=n_classes,
            n_features=n_features,
            training_config={
                k: getattr(self, k)
                for k in [
                    "z_dim",
                    "num_epochs",
                    "batch_size",
                    "learning_rate",
                    "beta1",
                    "alpha",
                    "beta",
                    "delta_mean",
                    "delta_var",
                    "random_state",
                ]
            },
        )

        self.is_fitted = True
        return self

    def sample(self, n: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Return a DataFrame of synthetic rows.

        Parameters
        ----------
        n : int, optional
            Number of rows to generate.  When *None* the cached output from
            the last ``train()`` call is returned (same length as training
            data).

        Returns
        -------
        pd.DataFrame
            Columns match the original training data (features + label).
        """
        if not self.is_fitted:
            raise RuntimeError("Call train() before sample().")

        if n is None:
            X = self._X_synth
            y = self._y_synth
        else:
            torch = importlib.import_module("torch")
            X_norm, y_enc = self._generate_samples(n, torch)
            X = self._normalizer.inverse_transform(X_norm)
            y = self._label_encoder.inverse_transform(y_enc)

        df = pd.DataFrame(X, columns=self._x_columns)
        df[self._label_column] = y
        return df[self._column_names]

    def evaluate(self, *args, **kwargs) -> float:
        """Stub evaluation — returns 0.0.

        Full TSTR evaluation is handled by ``TSTREvaluation`` in the pipeline.
        """
        if not self.is_fitted:
            raise RuntimeError("Call train() before evaluate().")
        return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_samples(
        self, n_samples: int, torch
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate *n_samples* rows in normalised space.

        Parameters
        ----------
        n_samples : int
            Number of rows to generate.
        torch : module
            Already-imported torch module (avoids re-importing).

        Returns
        -------
        X_norm : np.ndarray, shape (n_samples, n_features)
            Synthetic features in [-1, 1] normalised space.
        y_enc : np.ndarray, shape (n_samples,)
            Integer-encoded class labels.
        """
        self._generator.eval()
        X_parts = []
        y_parts = []
        remaining = n_samples

        with torch.no_grad():
            while remaining > 0:
                bsz = min(self.batch_size, remaining)
                z = torch.randn(bsz, self.z_dim, device=self._device)
                y_idx = torch.randint(
                    0, self._n_classes, (bsz,), device=self._device
                )
                y_oh = torch.zeros(bsz, self._n_classes, device=self._device)
                y_oh.scatter_(1, y_idx.unsqueeze(1), 1.0)
                X_fake = self._generator(z, y_oh).cpu().numpy()
                X_parts.append(X_fake)
                y_parts.append(y_idx.cpu().numpy())
                remaining -= bsz

        self._generator.train()
        X_all = np.vstack(X_parts)[:n_samples]
        y_all = np.concatenate(y_parts)[:n_samples].astype(int)
        return X_all, y_all
