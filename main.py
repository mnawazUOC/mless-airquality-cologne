"""
main.py
=======

Reproduce the full NO2 forecasting study from the command line:

    python main.py path/to/eea_daily.parquet

It runs:
  1. Data loading, station selection, and preprocessing.
  2. Persistence baseline.
  3. FFN and LSTM (univariate: NO2 only).
  4. Ablation (multivariate: NO2 + PM10, forward-filled).
  5. Prints a comparison table of all models (MAE, RMSE in ug/m3).

Results should match those documented in docs/PROJECT.md (window = 7).
Requires: numpy, pandas, pyarrow, torch.
"""

from __future__ import annotations

import sys

from src.data_processing import (
    load_cologne_area,
    build_station_series,
    make_windows,
    chronological_split,
    standardize,
    TARGET_STATION,
)
from src.models import FFN, LSTMModel
from src.evaluate import compute_metrics, persistence_baseline, train_and_eval

WINDOW = 7
HORIZON = 1


def prepare(series, features):
    """Windows -> split -> scale. Returns everything needed to train/evaluate."""
    X, y, d = make_windows(series, target="NO2", features=features,
                           window=WINDOW, horizon=HORIZON)
    Xtr, ytr, dtr, Xte, yte, dte = chronological_split(X, y, d)
    Xtr_s, Xte_s, mean, std = standardize(Xtr, Xte)
    no2_mean, no2_std = mean[0], std[0]          # NO2 is feature 0
    ytr_s = (ytr - no2_mean) / no2_std
    return Xtr_s, ytr_s, Xte_s, yte, Xte, no2_mean, no2_std


def main(parquet_path: str):
    print("Loading and preprocessing data ...")
    df = load_cologne_area(parquet_path)

    # univariate series (NO2 only) and multivariate series (NO2 + PM10, ffilled)
    series_uni = build_station_series(df, features=("NO2",), ffill=False)
    series_multi = build_station_series(df, features=("NO2", "PM10"), ffill=True)

    results = {}

    # ---- univariate prep ----
    Xtr_s, ytr_s, Xte_s, yte, Xte_raw, no2_m, no2_s = prepare(series_uni, ["NO2"])
    print(f"Station {TARGET_STATION}: train={len(ytr_s)}  test={len(yte)}\n")

    # ---- persistence baseline ----
    mae, rmse = persistence_baseline(Xte_raw, yte)
    results["Persistence"] = (mae, rmse)

    # ---- FFN univariate ----
    mae, rmse = train_and_eval(
        lambda: FFN(window=WINDOW, n_features=1), Xtr_s, ytr_s, Xte_s, yte, no2_m, no2_s)
    results["FFN (uni)"] = (mae, rmse)

    # ---- LSTM univariate ----
    mae, rmse = train_and_eval(
        lambda: LSTMModel(n_features=1), Xtr_s, ytr_s, Xte_s, yte, no2_m, no2_s)
    results["LSTM (uni)"] = (mae, rmse)

    # ---- multivariate prep (NO2 + PM10) ----
    Xtr_s, ytr_s, Xte_s, yte, Xte_raw, no2_m, no2_s = prepare(
        series_multi, ["NO2", "PM10"])

    mae, rmse = train_and_eval(
        lambda: FFN(window=WINDOW, n_features=2), Xtr_s, ytr_s, Xte_s, yte, no2_m, no2_s)
    results["FFN (multi)"] = (mae, rmse)

    mae, rmse = train_and_eval(
        lambda: LSTMModel(n_features=2), Xtr_s, ytr_s, Xte_s, yte, no2_m, no2_s)
    results["LSTM (multi)"] = (mae, rmse)

    # ---- report ----
    print("=== Results (test set 2022-2023, window = 7) ===")
    print(f"{'Model':16s} {'MAE':>7} {'RMSE':>7}")
    for name, (mae, rmse) in results.items():
        print(f"{name:16s} {mae:>7.2f} {rmse:>7.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py path/to/eea_daily.parquet")
        sys.exit(1)
    main(sys.argv[1])
