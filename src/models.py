"""
models.py
=========

Neural network models for next-day NO2 forecasting: a feed-forward network (FFN)
and an LSTM. Both take a window of shape (batch, window_days, n_features) and
output a single value per sample (the next-day NO2 prediction).

The FFN flattens the window and has no notion of time order. The LSTM reads the
window sequentially and predicts from its final time step, so it can model
temporal dependence.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FFN(nn.Module):
    """Feed-forward network. Flattens the window, then two dense layers."""

    def __init__(self, window: int, n_features: int, hidden: int = 32):
        super().__init__()
        input_size = window * n_features
        self.net = nn.Sequential(
            nn.Flatten(),                    # (batch, window, feat) -> (batch, window*feat)
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)       # (batch,) not (batch, 1)


class LSTMModel(nn.Module):
    """LSTM that reads the window sequentially and predicts from the last step."""

    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            batch_first=True,                # input shape (batch, days, features)
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                # process the whole sequence
        last = out[:, -1, :]                 # output at the final day
        return self.fc(last).squeeze(-1)     # (batch,)
