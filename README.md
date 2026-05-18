# PCM Thermal Forecasting

One-step-ahead temperature prediction for a Phase-Change Material (PCM) brick compartment.
Three models (Ridge, Random Forest, HistGradientBoosting) are compared across four feature
configurations under Leave-One-PCM-Out cross-validation.

## Setup

```bash
bash setup.sh
```

This creates a local `.venv/` and installs `pandas`, `scikit-learn`, `matplotlib`, `openpyxl`, `pyarrow`.

## Run

```bash
.venv/bin/python shared/prepare_data.py                    # raw xlsx -> data_clean/*.parquet
.venv/bin/python experiments/01_ridge/tune.py              # ~5 s
.venv/bin/python experiments/02_random_forest/tune.py      # ~1 min
.venv/bin/python experiments/03_gradient_boosting/tune.py  # ~10-15 min
```

Each `tune.py` runs the full grid search inside LOPO CV and writes its own
`metrics.csv`, `best_params.csv`, `grid_search_log.csv`, `predictions.csv`, and 7 PNG plots.

## Folder layout

```
PCM-Thermal-Forecasting/
├── Charging_Dataset.xlsx
├── Discharging_Dataset.xlsx
├── requirements.txt
├── setup.sh
├── data_clean/        regenerated parquet files
├── shared/            prepare_data.py, experiment.py, plots.py
├── experiments/
│   ├── 01_ridge/
│   ├── 02_random_forest/
│   └── 03_gradient_boosting/
├── figures/           consolidated plots (incl. comparison_rmse.png, feature_correlation.png)
├── all_metrics.csv    aggregated 72-row results table
├── all_best_params.csv
└── summary.csv
```

## Results

### Comparison RMSE

![Comparison RMSE](figures/comparison_rmse.png)

### Feature Correlation

![Feature Correlation](figures/feature_correlation.png)
