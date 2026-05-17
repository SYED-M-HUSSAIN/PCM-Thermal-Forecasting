"""Shared experiment scaffolding.

All 4 model agents import from this module so the feature definitions,
splits, and metrics are identical across the comparison.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_clean"

TARGET = "PCM Brick Compartment (°C)"

BASE_FEATURES = [
    "sin_hour",
    "cos_hour",
    "Ambient (°C)",
    "Clay Brick Compartment (°C)",
    "PCM Tube (°C)",
]

CASES: dict[str, list[str]] = {
    "case1_baseline":       BASE_FEATURES,
    "case2_solar":          BASE_FEATURES + ["Solar_proxy"],
    "case3_lags":           BASE_FEATURES + ["Lag1", "Lag2"],
    "case4_solar_and_lags": BASE_FEATURES + ["Solar_proxy", "Lag1", "Lag2"],
}

REGIMES = ["charging", "discharging"]


def load_regime(regime: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{regime}.parquet")


def loro_splits(df: pd.DataFrame) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (held_out_run_id, train_idx, test_idx) for leave-one-run-out CV."""
    for held_out in sorted(df["run_id"].unique()):
        test_idx  = df.index[df["run_id"] == held_out].to_numpy()
        train_idx = df.index[df["run_id"] != held_out].to_numpy()
        yield int(held_out), train_idx, test_idx


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    return {
        "rmse":        float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":         float(mean_absolute_error(y_true, y_pred)),
        "r2":          float(r2_score(y_true, y_pred)),
        "mape_pct":    float(np.mean(np.abs(err / y_true)) * 100),
        "max_abs_err": float(np.max(np.abs(err))),
    }


def run_experiment(
    model_name: str,
    make_model: Callable[[], object],
    out_dir: Path,
    needs_scaling: bool = False,
) -> pd.DataFrame:
    """Run the full 2 regimes × 4 cases × 3 folds matrix and return metrics."""
    from sklearn.preprocessing import StandardScaler

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    predictions: dict[tuple, pd.DataFrame] = {}

    for regime in REGIMES:
        df = load_regime(regime).reset_index(drop=True)
        y_all = df[TARGET].to_numpy()

        for case_name, feats in CASES.items():
            X_all = df[feats].to_numpy()

            for held_out, tr, te in loro_splits(df):
                model = make_model()
                if needs_scaling:
                    scaler = StandardScaler().fit(X_all[tr])
                    Xtr, Xte = scaler.transform(X_all[tr]), scaler.transform(X_all[te])
                else:
                    Xtr, Xte = X_all[tr], X_all[te]

                model.fit(Xtr, y_all[tr])
                y_pred = model.predict(Xte)

                m = compute_metrics(y_all[te], y_pred)
                m.update({
                    "model": model_name,
                    "regime": regime,
                    "case": case_name,
                    "fold_holdout_run": held_out,
                    "n_train": len(tr),
                    "n_test":  len(te),
                })
                rows.append(m)

                predictions[(regime, case_name, held_out)] = pd.DataFrame({
                    "row_idx": te,
                    "time":    df.loc[te, "Time (HH:MM)"].to_numpy(),
                    "y_true":  y_all[te],
                    "y_pred":  y_pred,
                })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    # save predictions for plotting
    preds_long = []
    for (regime, case_name, fold), pdf in predictions.items():
        pdf = pdf.copy()
        pdf["regime"], pdf["case"], pdf["fold"] = regime, case_name, fold
        preds_long.append(pdf)
    pd.concat(preds_long, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)

    return metrics_df


def _expand_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]


def _score_loro(make_model_with_params, params: dict, df: pd.DataFrame,
                feats: list[str], needs_scaling: bool) -> tuple[float, list[float]]:
    """Mean RMSE across 3 LORO folds for a given (params, feature set)."""
    from sklearn.preprocessing import StandardScaler
    y = df[TARGET].to_numpy()
    X = df[feats].to_numpy()
    fold_rmses: list[float] = []
    for _, tr, te in loro_splits(df):
        if needs_scaling:
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        else:
            Xtr, Xte = X[tr], X[te]
        m = make_model_with_params(params)
        m.fit(Xtr, y[tr])
        pred = m.predict(Xte)
        fold_rmses.append(float(np.sqrt(mean_squared_error(y[te], pred))))
    return float(np.mean(fold_rmses)), fold_rmses


def run_grid_search_experiment(
    model_name: str,
    make_model_with_params: Callable[[dict], object],
    param_grid: dict[str, list],
    out_dir: Path,
    needs_scaling: bool = False,
) -> pd.DataFrame:
    """For each (regime, case): grid-search hyperparameters via LORO CV,
    pick the params with lowest mean RMSE, then write the LORO predictions
    and per-fold metrics for that winning configuration.
    """
    from sklearn.preprocessing import StandardScaler

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = _expand_grid(param_grid)

    metric_rows: list[dict] = []
    best_rows: list[dict] = []
    grid_rows: list[dict] = []
    predictions: dict[tuple, pd.DataFrame] = {}

    for regime in REGIMES:
        df = load_regime(regime).reset_index(drop=True)
        y_all = df[TARGET].to_numpy()

        for case_name, feats in CASES.items():
            X_all = df[feats].to_numpy()

            # ---- grid search via LORO mean RMSE ----
            best_params, best_score = None, np.inf
            for params in candidates:
                score, fold_rmses = _score_loro(
                    make_model_with_params, params, df, feats, needs_scaling,
                )
                grid_rows.append({
                    "model": model_name, "regime": regime, "case": case_name,
                    "params": json.dumps(params, sort_keys=True),
                    "mean_rmse": score,
                    **{f"fold{i}_rmse": v for i, v in enumerate(fold_rmses)},
                })
                if score < best_score:
                    best_score, best_params = score, params

            best_rows.append({
                "model": model_name, "regime": regime, "case": case_name,
                "best_params": json.dumps(best_params, sort_keys=True),
                "best_mean_rmse": best_score,
            })

            # ---- refit on each LORO fold with best params, record full metrics + preds ----
            for held_out, tr, te in loro_splits(df):
                if needs_scaling:
                    sc = StandardScaler().fit(X_all[tr])
                    Xtr, Xte = sc.transform(X_all[tr]), sc.transform(X_all[te])
                else:
                    Xtr, Xte = X_all[tr], X_all[te]
                m = make_model_with_params(best_params)
                m.fit(Xtr, y_all[tr])
                y_pred = m.predict(Xte)

                mm = compute_metrics(y_all[te], y_pred)
                mm.update({
                    "model": model_name, "regime": regime, "case": case_name,
                    "fold_holdout_run": held_out,
                    "n_train": len(tr), "n_test": len(te),
                    "best_params": json.dumps(best_params, sort_keys=True),
                })
                metric_rows.append(mm)
                predictions[(regime, case_name, held_out)] = pd.DataFrame({
                    "row_idx": te,
                    "time": df.loc[te, "Time (HH:MM)"].to_numpy(),
                    "y_true": y_all[te],
                    "y_pred": y_pred,
                })

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(best_rows).to_csv(out_dir / "best_params.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(out_dir / "grid_search_log.csv", index=False)

    preds_long = []
    for (regime, case_name, fold), pdf in predictions.items():
        pdf = pdf.copy()
        pdf["regime"], pdf["case"], pdf["fold"] = regime, case_name, fold
        preds_long.append(pdf)
    pd.concat(preds_long, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)

    return metrics_df
