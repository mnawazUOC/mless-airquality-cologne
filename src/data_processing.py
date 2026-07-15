"""
data_processing.py
==================

Data acquisition and preprocessing for the Cologne NO2 forecasting project.

Pipeline:
  1. Load the EEA daily air-quality dataset (Parquet) and filter to Cologne.
  2. Summarize per-station coverage and select the target station.
  3. Reconstruct a daily DatetimeIndex from (year, doy) for one station.
  4. Build gap-aware supervised windows (skip windows spanning missing days).
  5. Chronological train/test split (no shuffling -> no leakage).
  6. Standardize features using training statistics only.

The public functions use plain numpy arrays (X, y, dates), matching the notebook
and the sibling modules models.py / evaluate.py.

Data source
-----------
Heisig, J. / European Environment Agency (2024):
"EEA Air Quality In-Situ Measurement Station Data", Zenodo,
https://doi.org/10.5281/zenodo.14513586  (CC-BY-4.0)
File: ..._2.daily_pnt_20150101_20231231_eu_epsg.3035_v20240718.parquet (~204 MB).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COLOGNE_BBOX = dict(lon_min=6.75, lon_max=7.15, lat_min=50.80, lat_max=51.10)

TARGET_STATION = "DENW212"
TARGET_POLLUTANT = "NO2"
STATION_COL = "Air.Quality.Station.EoI.Code"


# ---------------------------------------------------------------------------
# Step 1-2: load, filter, inspect coverage
# ---------------------------------------------------------------------------

def load_cologne_area(parquet_path: str) -> pd.DataFrame:
    """Load only the Cologne-area rows using Parquet predicate pushdown."""
    dataset = ds.dataset(parquet_path, format="parquet")
    flt = (
        (ds.field("Countrycode") == "DE")
        & (ds.field("Longitude") >= COLOGNE_BBOX["lon_min"])
        & (ds.field("Longitude") <= COLOGNE_BBOX["lon_max"])
        & (ds.field("Latitude") >= COLOGNE_BBOX["lat_min"])
        & (ds.field("Latitude") <= COLOGNE_BBOX["lat_max"])
    )
    return dataset.to_table(filter=flt).to_pandas()


def station_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-station data coverage to justify station selection."""
    pollutants = ["NO2", "PM2.5", "PM10", "O3"]

    def _stats(g: pd.DataFrame) -> pd.Series:
        out = {
            "Station.Type": g["Station.Type"].iloc[0],
            "Station.Area": g["Station.Area"].iloc[0],
            "lon": g["Longitude"].iloc[0],
            "lat": g["Latitude"].iloc[0],
            "n_days_total": len(g),
            "year_min": int(g["year"].min()),
            "year_max": int(g["year"].max()),
        }
        for p in pollutants:
            out[f"{p}_pct"] = round(g[p].notna().mean() * 100, 1)
        return pd.Series(out)

    summary = df.groupby(STATION_COL).apply(_stats).reset_index()
    return summary.sort_values("NO2_pct", ascending=False)


# ---------------------------------------------------------------------------
# Step 3: reconstruct a daily time series for one station
# ---------------------------------------------------------------------------

def build_station_series(df, station=TARGET_STATION,
                         features=("NO2", "PM10"), ffill=False):
    """Return a date-indexed DataFrame for a single station.

    Date reconstructed from (year, doy). Index is NOT reindexed to a continuous
    range, so gap-awareness is handled in the windowing step. If ffill=True,
    feature gaps are forward-filled (leading NaNs back-filled) - used for PM10.
    """
    s = df[df[STATION_COL] == station].copy()
    s["date"] = pd.to_datetime(
        s["year"].astype(int).astype(str), format="%Y"
    ) + pd.to_timedelta(s["doy"].astype(int) - 1, unit="D")
    s = s.sort_values("date").set_index("date")
    out = s[list(features)].copy()
    if ffill:
        out = out.ffill().bfill()
    return out


# ---------------------------------------------------------------------------
# Step 4: gap-aware supervised windowing (array-based API)
# ---------------------------------------------------------------------------

def make_windows(series, target="NO2", features=None, window=7, horizon=1):
    """Turn a date-indexed DataFrame into (X, y, dates) windows.

    Only keeps windows where all days (window inputs + target) are strictly
    consecutive calendar dates and contain no NaN.

    Returns
    -------
    X : ndarray (n_samples, window, n_features)
    y : ndarray (n_samples,)
    dates : ndarray  target date of each sample
    """
    if features is None:
        features = [target]

    sub = series[features]
    values = sub.values
    dates = sub.index.values
    target_idx = features.index(target)

    X_list, y_list, d_list = [], [], []
    span = window + horizon
    one_day = np.timedelta64(1, "D")

    for i in range(len(sub) - span + 1):
        block_dates = dates[i : i + span]
        if np.all(np.diff(block_dates) == one_day):
            block_vals = values[i : i + span]
            if not np.isnan(block_vals).any():
                X_list.append(values[i : i + window])
                y_list.append(values[i + window + horizon - 1, target_idx])
                d_list.append(dates[i + window + horizon - 1])

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(d_list),
    )


# ---------------------------------------------------------------------------
# Step 5: chronological train/test split
# ---------------------------------------------------------------------------

def chronological_split(X, y, dates, split_year=2022):
    """Split by target date: train < split_year <= test.

    Returns Xtr, ytr, dtr, Xte, yte, dte.
    """
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    tr = years < split_year
    te = ~tr
    return X[tr], y[tr], dates[tr], X[te], y[te], dates[te]


# ---------------------------------------------------------------------------
# Step 6: standardization (fit on train only)
# ---------------------------------------------------------------------------

def standardize(X_train, X_test):
    """Standardize features using TRAIN statistics only.

    Returns scaled train, scaled test, and per-feature (mean, std) arrays.
    """
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    X_train_s = ((X_train - mean) / std).astype("float32")
    X_test_s = ((X_test - mean) / std).astype("float32")
    return X_train_s, X_test_s, mean, std


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/eea_daily.parquet"
    df = load_cologne_area(path)
    print(f"Loaded {len(df):,} Cologne-area rows\n")
    print(station_coverage(df).to_string(index=False))

    series = build_station_series(df, ffill=False)
    print(f"\n{TARGET_STATION} series: {series.index.min().date()} -> "
          f"{series.index.max().date()}, {len(series)} rows")

    X, y, d = make_windows(series, target="NO2", features=["NO2"], window=7)
    print(f"Univariate windows: {len(y)} samples, X shape {X.shape}")
    Xtr, ytr, dtr, Xte, yte, dte = chronological_split(X, y, d)
    print(f"Train: {len(ytr)}  Test: {len(yte)}")
