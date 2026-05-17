"""Shared plotting helpers — every model agent calls these so plots look the same."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 180

CASE_LABEL = {
    "case1_baseline":       "Case 1\nBaseline",
    "case2_solar":          "Case 2\n+ Solar",
    "case3_lags":           "Case 3\n+ Lags",
    "case4_solar_and_lags": "Case 4\n+ Solar + Lags",
}
CASE_LABEL_SHORT = {
    "case1_baseline":       "Case 1\n(Baseline)",
    "case2_solar":          "Case 2\n(+ Solar)",
    "case3_lags":           "Case 3\n(+ Lags)",
    "case4_solar_and_lags": "Case 4\n(Full)",
}


def _ensure(dir_: Path) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    return dir_


def _pcm_label(fold: int) -> str:
    return f"Held-out: PCM #{int(fold) + 1}"


def plot_pred_vs_actual_timeseries(preds: pd.DataFrame, model_name: str, out_dir: Path) -> None:
    """One PNG per regime. 4 cases × 3 PCMs grid showing predicted vs actual over time."""
    out_dir = _ensure(out_dir)
    cases = sorted(preds["case"].unique())
    folds = sorted(preds["fold"].unique())

    for regime in sorted(preds["regime"].unique()):
        fig, axes = plt.subplots(len(cases), len(folds),
                                 figsize=(4.6 * len(folds), 2.9 * len(cases)),
                                 sharex=False, sharey=True)
        for i, case in enumerate(cases):
            for j, fold in enumerate(folds):
                ax = axes[i][j] if len(cases) > 1 else axes[j]
                sub = preds[(preds["regime"] == regime)
                            & (preds["case"] == case)
                            & (preds["fold"] == fold)].sort_values("row_idx")
                ax.plot(sub["y_true"].to_numpy(), label="Actual",    lw=1.7, color="#1f4e79")
                ax.plot(sub["y_pred"].to_numpy(), label="Predicted", lw=1.4, ls="--", color="#c0504d")
                if i == 0:
                    ax.set_title(_pcm_label(fold), fontsize=11, fontweight="bold")
                if j == 0:
                    ax.set_ylabel(CASE_LABEL[case], fontsize=10, fontweight="bold")
                if i == len(cases) - 1:
                    ax.set_xlabel("Time step (5-min intervals)", fontsize=9)
                ax.tick_params(labelsize=8)
                ax.grid(alpha=0.25, linestyle=":")
        axes[0][0].legend(fontsize=9, loc="upper left", framealpha=0.9)
        fig.suptitle(
            f"{model_name} — {regime.title()}: Predicted vs Actual PCM Temperature\n"
            f"(Leave-One-Material-Out CV; rows = feature case, columns = held-out PCM)",
            fontsize=12, fontweight="bold")
        fig.supylabel("PCM Brick Compartment Temperature (°C)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"timeseries_{regime}.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def plot_pred_vs_actual_scatter(preds: pd.DataFrame, model_name: str, out_dir: Path) -> None:
    """One scatter plot per regime, all PCM folds combined, one panel per case."""
    out_dir = _ensure(out_dir)
    cases = sorted(preds["case"].unique())
    pcm_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

    for regime in sorted(preds["regime"].unique()):
        fig, axes = plt.subplots(1, len(cases), figsize=(3.8 * len(cases), 3.8),
                                 sharex=True, sharey=True)
        for ax, case in zip(axes, cases):
            sub = preds[(preds["regime"] == regime) & (preds["case"] == case)]
            for fold, color in pcm_colors.items():
                sub_f = sub[sub["fold"] == fold]
                if len(sub_f):
                    ax.scatter(sub_f["y_true"], sub_f["y_pred"], s=10, alpha=0.6,
                               color=color, label=f"PCM #{fold + 1}", edgecolors="none")
            lo = min(sub["y_true"].min(), sub["y_pred"].min())
            hi = max(sub["y_true"].max(), sub["y_pred"].max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
            ax.set_title(CASE_LABEL_SHORT[case].replace("\n", " "), fontsize=10, fontweight="bold")
            ax.set_xlabel("Actual (°C)", fontsize=9)
            ax.set_ylabel("Predicted (°C)", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.25, linestyle=":")
        axes[0].legend(fontsize=8, loc="upper left", framealpha=0.9)
        fig.suptitle(f"{model_name} — {regime.title()}: Predicted vs Actual (held-out PCM colored)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_{regime}.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def plot_residuals(preds: pd.DataFrame, model_name: str, out_dir: Path) -> None:
    out_dir = _ensure(out_dir)
    cases = sorted(preds["case"].unique())
    pcm_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

    for regime in sorted(preds["regime"].unique()):
        fig, axes = plt.subplots(1, len(cases), figsize=(3.8 * len(cases), 3.4), sharey=True)
        for ax, case in zip(axes, cases):
            sub = preds[(preds["regime"] == regime) & (preds["case"] == case)].sort_values("row_idx")
            for fold, color in pcm_colors.items():
                sub_f = sub[sub["fold"] == fold]
                if len(sub_f):
                    ax.scatter(sub_f["row_idx"], sub_f["y_pred"] - sub_f["y_true"],
                               s=10, alpha=0.6, color=color, label=f"PCM #{fold + 1}",
                               edgecolors="none")
            ax.axhline(0, color="k", lw=0.7)
            ax.set_title(CASE_LABEL_SHORT[case].replace("\n", " "), fontsize=10, fontweight="bold")
            ax.set_xlabel("Sample index", fontsize=9)
            ax.set_ylabel("Residual: predicted − actual (°C)", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.25, linestyle=":")
        axes[0].legend(fontsize=8, loc="upper right", framealpha=0.9)
        fig.suptitle(f"{model_name} — {regime.title()}: Prediction Residuals",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / f"residuals_{regime}.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def plot_metrics_bar(metrics: pd.DataFrame, model_name: str, out_dir: Path) -> None:
    """RMSE per case, per regime. Bars = mean across PCM folds, error bars = std."""
    out_dir = _ensure(out_dir)
    agg = metrics.groupby(["regime", "case"])["rmse"].agg(["mean", "std"]).reset_index()
    regimes = sorted(agg["regime"].unique())
    cases   = sorted(agg["case"].unique())
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.2 * len(regimes), 3.8), sharey=True)
    for ax, regime in zip(axes, regimes):
        sub = agg[agg["regime"] == regime].set_index("case").loc[cases]
        x = np.arange(len(cases))
        ax.bar(x, sub["mean"], yerr=sub["std"], capsize=5, color="#4f81bd", edgecolor="#1f4e79")
        ax.set_xticks(x)
        ax.set_xticklabels([CASE_LABEL_SHORT[c] for c in cases], fontsize=9)
        ax.set_title(regime.title(), fontsize=11, fontweight="bold")
        ax.set_ylabel("RMSE (°C)", fontsize=10)
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        for i, v in enumerate(sub["mean"]):
            ax.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.suptitle(f"{model_name}: RMSE by Feature Case\n"
                 "(mean ± std across 3 Leave-One-Material-Out folds)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_by_case.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def make_all_plots(metrics_csv: Path, predictions_csv: Path, model_name: str, out_dir: Path) -> None:
    metrics = pd.read_csv(metrics_csv)
    preds   = pd.read_csv(predictions_csv)
    plot_pred_vs_actual_timeseries(preds, model_name, out_dir)
    plot_pred_vs_actual_scatter(preds, model_name, out_dir)
    plot_residuals(preds, model_name, out_dir)
    plot_metrics_bar(metrics, model_name, out_dir)
