# PCM Thermal Forecasting

## Problem Statement

This project addresses the challenge of **one-step-ahead temperature prediction** for Phase-Change Material (PCM) brick compartments in thermal energy storage systems. Accurate temperature forecasting is critical for optimizing thermal management in energy systems, particularly for applications involving phase-change materials that undergo transitions at specific temperature thresholds.

The study systematically evaluates three machine learning models (Ridge Regression, Random Forest, and Histogram Gradient Boosting) across four feature engineering configurations to identify the most effective approach for accurate temperature predictions in both charging and discharging regimes.

## Dataset

### Data Sources
The project utilizes two datasets:
- **Charging Dataset**: Temperature measurements during the charging phase of PCM materials
- **Discharging Dataset**: Temperature measurements during the discharging phase of PCM materials

Both datasets are provided in Excel format:
- `Charging_Dataset.xlsx`
- `Discharging_Dataset.xlsx`

### Data Preprocessing
Raw data is preprocessed and converted to Apache Parquet format for efficient storage and retrieval:
```bash
.venv/bin/python shared/prepare_data.py  # raw xlsx → data_clean/*.parquet
```

## Methodology

### Cross-Validation Strategy

**Leave-One-PCM-Out (LOPO) Cross-Validation** is employed to ensure robust model evaluation:
- Each PCM unit is sequentially held out as the test set
- The remaining PCM units comprise the training set
- This approach prevents data leakage and ensures generalization across different PCM units
- Typical split: ~166-227 training samples, ~71-131 test samples per fold

### Feature Engineering

Four feature configurations are evaluated:

| Case | Description | Features |
|------|-------------|----------|
| **case1_baseline** | Baseline features without temporal context | Current temperature, solar radiation, ambient conditions |
| **case2_solar** | Enhanced with solar radiation features | Baseline + aggregated solar metrics |
| **case3_lags** | Temporal dependencies via lag features | Temperature lags (t-1, t-2, t-3, ...) |
| **case4_solar_and_lags** | Combined approach | Solar features + lag features |

## Models & Technology Stack

### Machine Learning Models

Three algorithms are compared:

1. **Ridge Regression (L2 Regularization)**
   - Linear model with alpha regularization: [0.001, 0.1, 1.0, 10.0, 100.0]
   - Baseline for comparison and interpretability

2. **Random Forest**
   - Ensemble of decision trees
   - Hyperparameters tuned:
     - `max_depth`: [6, 12, null]
     - `max_features`: ["sqrt", 1.0]
     - `min_samples_leaf`: [1, 2, 5]
     - `n_estimators`: [300, 600]

3. **Histogram Gradient Boosting**
   - Gradient boosting with histogram-based learning
   - Hyperparameters tuned:
     - `max_depth`: [4, 6, 8]
     - `max_iter`: [300, 800]
     - `learning_rate`: [0.01, 0.03, 0.1]
     - `l2_regularization`: [0.0, 1.0]
     - `min_samples_leaf`: [5, 15]

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.x |
| **Data Processing** | pandas, Apache Parquet |
| **Machine Learning** | scikit-learn |
| **Visualization** | matplotlib |
| **Data I/O** | openpyxl (Excel), pyarrow (Parquet) |

## Evaluation Metrics

Model performance is assessed using:
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily
- **MAE (Mean Absolute Error)**: Average absolute deviation between predictions and actuals
- **R² Score**: Coefficient of determination (0-1 scale, higher is better)
- **MAPE (Mean Absolute Percentage Error)**: Percentage error relative to actual values
- **Max Absolute Error**: Maximum deviation in any single prediction

## Project Structure

```
PCM-Thermal-Forecasting/
├── Charging_Dataset.xlsx
├── Discharging_Dataset.xlsx
├── requirements.txt
├── setup.sh
├── data_clean/                    # Preprocessed parquet files
├── shared/                        # Shared modules
│   ├── prepare_data.py           # Data preprocessing pipeline
│   ├── experiment.py             # Model training & evaluation
│   └── plots.py                  # Visualization utilities
├── experiments/                   # Model-specific experiments
│   ├── 01_ridge/                 # Ridge regression experiments
│   │   ├── tune.py               # Hyperparameter tuning
│   │   ├── metrics.csv
│   │   ├── best_params.csv
│   │   ├── grid_search_log.csv
│   │   ├── predictions.csv
│   │   └── *.png                 # 7 visualization plots
│   ├── 02_random_forest/         # Random forest experiments
│   └── 03_gradient_boosting/     # Gradient boosting experiments
├── figures/                       # Consolidated plots
│   ├── comparison_rmse.png
│   ├── feature_correlation.png
│   └── *.png
├── all_metrics.csv               # Aggregated results (72 rows)
├── all_best_params.csv           # Best hyperparameters
├── summary.csv                   # Summary statistics
└── README.md
```

## Setup & Installation

### Environment Setup

Create and activate a virtual environment with all required dependencies:

```bash
bash setup.sh
```

This script:
1. Creates a local Python virtual environment (`.venv/`)
2. Installs required packages:
   - `pandas` – Data manipulation
   - `scikit-learn` – Machine learning models
   - `matplotlib` – Data visualization
   - `openpyxl` – Excel file handling
   - `pyarrow` – Parquet file support

## Execution

### Step 1: Data Preprocessing
```bash
.venv/bin/python shared/prepare_data.py
# Output: data_clean/*.parquet (cleaned datasets)
# Duration: ~30 seconds
```

### Step 2: Model Training & Tuning

Execute tuning for each model type:

```bash
# Ridge Regression (fastest)
.venv/bin/python experiments/01_ridge/tune.py              # ~5 seconds

# Random Forest (moderate)
.venv/bin/python experiments/02_random_forest/tune.py      # ~1 minute

# Histogram Gradient Boosting (slowest)
.venv/bin/python experiments/03_gradient_boosting/tune.py  # ~10-15 minutes
```

### Output Generated

Each `tune.py` script produces:
- `metrics.csv` – Performance metrics for all fold/hyperparameter combinations
- `best_params.csv` – Optimal hyperparameters identified via grid search
- `grid_search_log.csv` – Complete grid search history
- `predictions.csv` – Model predictions on test sets
- `*.png` – 7 visualization plots (learning curves, residual analysis, etc.)

## Results

### Comparison RMSE

![Comparison RMSE](figures/comparison_rmse.png)

### Feature Correlation

![Feature Correlation](figures/feature_correlation.png)

### Best Model Results

| Regime | Case | Best Model | RMSE | MAE | R² | MAPE (%) |
|--------|------|-----------|------|-----|-----|----------|
| Charging | case1_baseline | Ridge | 0.12 | 0.10 | 0.999 | 0.31 |
| Charging | case2_solar | Ridge | 0.12 | 0.10 | 0.999 | 0.31 |
| Charging | case3_lags | Ridge | 0.12 | 0.10 | 0.999 | 0.31 |
| Charging | case4_solar_and_lags | Ridge | 0.12 | 0.10 | 0.999 | 0.31 |
| Discharging | case1_baseline | Ridge | 2.05 | 1.88 | 0.887 | 6.25 |
| Discharging | case2_solar | Ridge | 2.08 | 1.89 | 0.883 | 6.29 |
| Discharging | case3_lags | Ridge | 1.23 | 0.82 | 0.959 | 2.67 |
| Discharging | case4_solar_and_lags | Ridge | 1.46 | 1.35 | 0.958 | 3.47 |

### Key Findings

- **Ridge Regression** consistently outperforms ensemble methods across all configurations
- **Lag-based features (case3 & case4)** significantly improve predictive accuracy, particularly for discharging regime
  - Discharging RMSE improvement: **2.05 → 1.23** (40% reduction with lag features)
  - R² improvement: **0.887 → 0.959**
- **Charging regime** exhibits superior predictability (R² ~0.999) compared to discharging (R² ~0.887-0.959)
- **Feature engineering** has a more substantial impact than model selection on prediction accuracy

## References & Documentation

- **scikit-learn Documentation**: https://scikit-learn.org/
- **Leave-One-Out Cross-Validation**: Standard practice for small sample evaluation
- **Phase-Change Materials**: Thermal energy storage technology for building and industrial applications
