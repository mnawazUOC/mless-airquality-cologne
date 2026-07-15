# Forecasting Urban NO2 in Cologne (MLESS SoSe 2026)

One-step-ahead forecasting of daily-mean **NO2** concentration at a Cologne
traffic station (DENW212), comparing a persistence baseline, a feed-forward
network (FFN) and an LSTM, with a univariate-vs-multivariate (NO2 + PM10)
ablation and a window-length study.

Full report: [`docs/PROJECT.md`](docs/PROJECT.md).

## Key results (test set 2022–2023, window = 7)

| Model | Features | MAE | RMSE |
|---|---|---|---|
| Persistence | — | 6.22 | 7.84 |
| FFN | NO2 | **5.73** | **7.14** |
| LSTM | NO2 | 5.85 | 7.23 |

All neural models beat the baseline. The FFN is marginally best at short windows,
but the LSTM overtakes it at longer windows (30–60 days). Adding PM10 gives only a
small, model-dependent benefit. See the report for the full analysis.

## Repository structure

```
.
├── README.md               # this file
├── requirements.txt
├── .gitignore
├── main.py                 # reproduce the full study from the command line
├── data/                   # raw parquet (gitignored — download from Zenodo)
├── src/
│   ├── __init__.py         # marks src as a package
│   ├── data_processing.py  # load, filter, station-select, gap-aware windows, split, scale
│   ├── models.py           # FFN and LSTM model definitions
│   └── evaluate.py         # metrics (MAE, RMSE), persistence baseline, train/eval loop
├── notebooks/
│   └── CODE_MLESS_2026.ipynb   # full control flow: EDA, models, experiments, plots
├── results/                # saved figures (plotA, plotB, plotC)
└── docs/
    └── PROJECT.md          # planning, implementation, results, conclusions
```

## Data

EEA Air Quality In-Situ Measurement Station Data (daily aggregate),
DOI [10.5281/zenodo.14513586](https://doi.org/10.5281/zenodo.14513586), CC-BY-4.0.
Download the `..._2.daily_...parquet` file into `data/`.

## How to run

Two independent ways to run the study:

**1. Reproduce the core results from the command line:**
```bash
pip install -r requirements.txt
python main.py data/<daily_parquet_filename>.parquet
```
This runs data prep, the persistence baseline, FFN and LSTM (univariate), and the
PM10 ablation, then prints the results table.

**2. Explore the full study interactively:**
Open `notebooks/CODE_MLESS_2026.ipynb` — it contains the EDA, all experiments
(including the window-length study) and the plots.

## Method summary

- **Target:** next-day daily-mean NO2 (ug/m3) at station DENW212.
- **Input:** sliding windows of the previous N days (gap-aware — windows spanning
  missing days are dropped).
- **Split:** chronological (train 2015–2021, test 2022–2023) to avoid leakage.
- **Models:** persistence baseline, FFN, LSTM (Adam, MSE, 100 epochs, seed 42).
- **Metrics:** MAE and RMSE, in ug/m3.

## Getting the data

The raw parquet (~204 MB) is not stored in this repository (see .gitignore).
To reproduce the results, download the daily aggregate file from Zenodo:

- Dataset: EEA Air Quality In-Situ Measurement Station Data
- DOI: https://doi.org/10.5281/zenodo.14513586
- File: airquality.no2.o3.so2.pm10.pm2p5_2.daily_pnt_20150101_20231231_eu_epsg.3035_v20240718.parquet

Place the downloaded file in the data/ folder, then run:

    python main.py data/airqualitymless.parquet
