"""Ridge regression with alpha grid-searched via LORO CV.

Replaces plain LinearRegression. Ridge subsumes OLS: at very small alpha
(~1e-3) the model behaves essentially like OLS, so we lose nothing by tuning.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))

import pandas as pd
from sklearn.linear_model import Ridge

from experiment import run_grid_search_experiment
from plots import make_all_plots

MODEL_NAME = "Ridge"
PARAM_GRID = {
    "alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
}


def make_model(params: dict) -> Ridge:
    return Ridge(random_state=42, **params)


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    metrics_df = run_grid_search_experiment(
        model_name=MODEL_NAME,
        make_model_with_params=make_model,
        param_grid=PARAM_GRID,
        out_dir=out_dir,
        needs_scaling=True,
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
