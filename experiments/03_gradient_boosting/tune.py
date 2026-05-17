"""HistGradientBoostingRegressor with hyperparameters grid-searched via LORO CV.

sklearn substitute for XGBoost (libomp not installed on this machine).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from experiment import run_grid_search_experiment
from plots import make_all_plots

MODEL_NAME = "HistGradientBoosting"
PARAM_GRID = {
    "max_iter":         [300, 800],
    "learning_rate":    [0.03, 0.1],
    "max_depth":        [4, 8, None],
    "min_samples_leaf": [5, 15],
    "l2_regularization": [0.0, 1.0],
}


def make_model(params: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=42, **params)


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    metrics_df = run_grid_search_experiment(
        model_name=MODEL_NAME,
        make_model_with_params=make_model,
        param_grid=PARAM_GRID,
        out_dir=out_dir,
        needs_scaling=False,
    )

    make_all_plots(
        out_dir / "metrics.csv",
        out_dir / "predictions.csv",
        MODEL_NAME,
        out_dir / "figures",
    )

    summary = metrics_df.groupby(["regime", "case"])["rmse"].mean().reset_index()
    print(f"\n=== {MODEL_NAME}: mean RMSE per (regime, case) ===")
    print(summary.to_string(index=False))

    print(f"\n=== {MODEL_NAME}: best params per cell ===")
    print(pd.read_csv(out_dir / "best_params.csv").to_string(index=False))


if __name__ == "__main__":
    main()
