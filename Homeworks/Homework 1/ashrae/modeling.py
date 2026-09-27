"""Datasets, networks and the Lightning module that trains them."""
import logging
import warnings

import lightning.pytorch as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import METERS

# The data lives in memory, so DataLoader workers wouldn't help; hide the warnings that suggest them.
warnings.filterwarnings("ignore", ".*does not have many workers.*")
# Deprecation notice from inside Lightning, not from this code.
warnings.filterwarnings("ignore", ".*LeafSpec.*")
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


class EnergyDataset(Dataset):
    """Scaled features `x` plus meter, building and log1p target of each row of `df`, all as tensors."""

    def __init__(self, df, x):
        self.x = torch.from_numpy(x)
        self.meter = torch.from_numpy(df.meter.to_numpy().astype("int64"))
        self.building = torch.from_numpy(df.building_id.to_numpy().astype("int64"))
        self.y = torch.from_numpy(np.log1p(df.meter_reading.to_numpy(dtype="float32")))

    def __len__(self):
        return len(self.y)

    def __getitems__(self, idx):  # a whole batch in one indexing step, instead of row by row
        idx = torch.as_tensor(idx)
        return {"x": self.x[idx], "meter": self.meter[idx], "building": self.building[idx], "y": self.y[idx]}


def make_loaders(train_df, val_df, X_train, X_val, batch_size=4096):
    """Training and validation loaders, with a scaler fitted on the training rows. With `val_df=None` (final
    training on all data) the validation loader is None."""
    scaler = StandardScaler().fit(X_train)
    keep_batch = lambda batch: batch  # the dataset already returns a batched dict
    return (
        DataLoader(EnergyDataset(train_df, scaler.transform(X_train).astype("float32")),
                   batch_size=batch_size, shuffle=True, collate_fn=keep_batch),
        None if val_df is None else DataLoader(EnergyDataset(val_df, scaler.transform(X_val).astype("float32")),
                                               batch_size=65536, collate_fn=keep_batch),
        scaler,
    )


def hidden_layers(n_in, batch_norm=True, dropout=0.0, n_layers=2):
    """`n_layers` hidden layers of 64 units; dropout goes after BatchNorm and ReLU."""
    block = lambda n: [nn.Linear(n, 64), *([nn.BatchNorm1d(64)] if batch_norm else []), nn.ReLU(),
                       *([nn.Dropout(dropout)] if dropout else [])]
    return nn.Sequential(*block(n_in), *[m for _ in range(n_layers - 1) for m in block(64)])


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def relu(x):
    return torch.clamp(x, min=0.0)


# It is implemented by hand (taken from the lecture notes :) )
class CustomMLP(torch.nn.Module):
    def __init__(self, layer_sizes, manual_optimization=True):
        super().__init__()
        assert len(layer_sizes) == 4
        D_in, H1, H2, D_out = layer_sizes

        if manual_optimization:
            self.W1 = torch.randn(D_in, H1)
            self.b1 = torch.zeros(1, H1)
            self.W2 = torch.randn(H1, H2)
            self.b2 = torch.zeros(1, H2)
            self.W3 = torch.randn(H2, D_out)
            self.b3 = torch.zeros(1, D_out)
            for p in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]:
                p.requires_grad_(False)
        else:
            self.W1 = torch.nn.Parameter(torch.randn(D_in, H1))
            self.b1 = torch.nn.Parameter(torch.zeros(1, H1))
            self.W2 = torch.nn.Parameter(torch.randn(H1, H2))
            self.b2 = torch.nn.Parameter(torch.zeros(1, H2))
            self.W3 = torch.nn.Parameter(torch.randn(H2, D_out))
            self.b3 = torch.nn.Parameter(torch.zeros(1, D_out))

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = relu(Z2)
        pred = A2 @ self.W3 + self.b3 
        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "pred": pred}
        return pred, cache

    def loss(self, y_true, pred):
        return torch.sqrt(torch.mean((pred - y_true) ** 2))

    def backward(self, cache, y_true):
        X, Z1, A1, Z2, A2, pred = (cache["X"], cache["Z1"], cache["A1"], cache["Z2"], cache["A2"], cache["pred"])
        N = X.shape[0]
        
        residual = pred - y_true
        rmsle = torch.sqrt(torch.mean(residual**2))
        dpred = residual / (N * (rmsle + 1e-12))

        dW3 = A2.T @ dpred
        db3 = dpred.sum(dim=0, keepdim=True)

        dA2 = dpred @ self.W3.T
        dZ2 = dA2 * (Z2 > 0).float()

        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(dim=0, keepdim=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0).float()

        dW1 = X.T @ dZ1
        db1 = dZ1.sum(dim=0, keepdim=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def step(self, grads, lr):
        with torch.no_grad():
            self.W1 -= lr * grads["W1"]
            self.b1 -= lr * grads["b1"]
            self.W2 -= lr * grads["W2"]
            self.b2 -= lr * grads["b2"]
            self.W3 -= lr * grads["W3"]
            self.b3 -= lr * grads["b3"]

    def predict(self, X):
        pred, _ = self.forward(X)
        return pred.clamp(min=0).view(-1)

class MLP(nn.Module):
    def __init__(self, n_features, batch_norm=False):
        super().__init__()
        self.hidden = hidden_layers(n_features, batch_norm)
        self.out = nn.Linear(64, 1)

    def forward(self, x, meter, building):
        return self.out(self.hidden(x)).squeeze(1)


class MeterMLP(nn.Module):
    """One output per meter type; with `n_buildings`, a learned `emb_dim`-number vector per building is added to
    the input."""

    def __init__(self, n_features, n_buildings=0, emb_dim=16, dropout=0.0, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(n_buildings, emb_dim) if n_buildings else None
        self.hidden = hidden_layers(n_features + (emb_dim if n_buildings else 0), dropout=dropout, n_layers=n_layers)
        self.out = nn.Linear(64, len(METERS))

    def forward(self, x, meter, building):
        if self.emb is not None:
            x = torch.cat([x, self.emb(building)], dim=1)
        # one prediction per meter type; keep the row's own
        return self.out(self.hidden(x)).gather(1, meter.unsqueeze(1)).squeeze(1)


class EnergyModel(L.LightningModule):
    """Trains any of the networks above with MSE on log1p readings, which is exactly RMSLE.

    Default optimizer: AdamW with `weight_decay` on weight matrices and embeddings only (with 0 it's plain Adam).
    `optimizer`, a function of the parameters, replaces it."""

    def __init__(self, net, lr=1e-3, weight_decay=0.0, optimizer=None):
        super().__init__()
        self.net, self.optimizer = net, optimizer
        self.save_hyperparameters(ignore=["net", "optimizer"])
        self.train_rmsle = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmsle = torchmetrics.MeanSquaredError(squared=False)

    def forward(self, batch):
        return self.net(batch["x"], batch["meter"], batch["building"])

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        self.train_rmsle(pred, batch["y"])
        self.log("train_rmsle", self.train_rmsle, on_step=False, on_epoch=True)
        return F.mse_loss(pred, batch["y"])

    def validation_step(self, batch, batch_idx):
        self.val_rmsle(self(batch).clamp(min=0), batch["y"])  # readings can't be negative
        self.log("val_rmsle", self.val_rmsle, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.optimizer is not None:
            return self.optimizer(self.parameters())
        weights = [p for p in self.parameters() if p.ndim >= 2]  # weight matrices and the embedding
        others = [p for p in self.parameters() if p.ndim < 2]  # biases and BatchNorm
        return torch.optim.AdamW(
            [{"params": weights, "weight_decay": self.hparams.weight_decay}, {"params": others, "weight_decay": 0.0}],
            lr=self.hparams.lr,
        )


@torch.no_grad()
def predict(model, loader):
    """log1p predictions for every row of `loader`, clipped at 0."""
    model.eval()
    preds = [model({k: v.to(model.device) for k, v in batch.items()}).clamp(min=0).cpu() for batch in loader]
    return torch.cat(preds).numpy()

