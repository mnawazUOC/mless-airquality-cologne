"""
evaluate.py
===========

Reusable metrics and a train/evaluate loop for the NO2 forecasting models.

- compute_metrics: MAE and RMSE in the original ug/m3 scale.
- persistence_baseline: the naive "tomorrow = today" forecast.
- train_and_eval: trains a model on standardized data and returns test MAE/RMSE
  after inverse-transforming predictions back to ug/m3.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Return (MAE, RMSE) in the same units as the inputs (ug/m3)."""
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return mae, rmse


def persistence_baseline(X_test_unscaled: np.ndarray, y_true: np.ndarray):
    """Naive baseline: predict the NO2 value of the last input day.

    X_test_unscaled has shape (samples, window, features); NO2 is feature 0.
    """
    preds = X_test_unscaled[:, -1, 0]
    return compute_metrics(y_true, preds)


def train_and_eval(
    model_factory,
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test_unscaled: np.ndarray,
    target_mean: float,
    target_std: float,
    epochs: int = 100,
    lr: float = 0.01,
    seed: int = 42,
):
    """Train a model and return (MAE, RMSE) on the test set in ug/m3.

    `model_factory` is a zero-argument callable that builds and returns the model
    (e.g. `lambda: FFN(window=7, n_features=1)`). The seed is set BEFORE the
    factory is called, so random weight initialization is reproducible and
    matches the notebook's ordering (seed -> create model -> train).

    Inputs are standardized; predictions are inverse-transformed
    (pred * target_std + target_mean) before computing metrics.
    """
    torch.manual_seed(seed)
    model = model_factory()          # weights initialized under the fixed seed

    Xt = torch.tensor(X_train_scaled, dtype=torch.float32)
    yt = torch.tensor(y_train_scaled, dtype=torch.float32)
    Xv = torch.tensor(X_test_scaled, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(Xv).numpy() * target_std + target_mean

    return compute_metrics(y_test_unscaled, preds)
