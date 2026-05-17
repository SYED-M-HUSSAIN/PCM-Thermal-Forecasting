"""Shared data preparation for the PCM thermal experiments.

Loads both xlsx files, identifies the 3 stacked experimental runs in each,
recomputes lag features per-run, rescales Solar_proxy, and writes two parquet
files that all 4 model agents consume.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = {
    "charging":    ROOT / "Charging_Dataset.xlsx",
    "discharging": ROOT / "Discharging_Dataset.xlsx",
}
OUT_DIR = ROOT / "data_clean"
OUT_DIR.mkdir(exist_ok=True)


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, skiprows=2)
    df.columns = df.columns.str.strip()
    # normalize column casing (source has "Cos_hour" with capital C)
    df = df.rename(columns={"Cos_hour": "cos_hour"})

    # parse time
    df["time"] = pd.to_datetime(df["Time (HH:MM)"], format="%H:%M").dt.time

    # identify 3 stacked runs by detecting where time decreases
    times_dt = pd.to_datetime(df["Time (HH:MM)"], format="%H:%M")
    run_id = (times_dt.diff().dt.total_seconds() < 0).cumsum()
    df["run_id"] = run_id.astype(int)

    # rescale Solar_proxy from 0-1000 -> 0-1
    df["Solar_proxy"] = df["Solar_proxy"] / 1000.0

    # recompute lags per-run (don't trust file's lag columns)
    target = "PCM Brick Compartment (°C)"
    df["Lag1"] = df.groupby("run_id")[target].shift(1)
    df["Lag2"] = df.groupby("run_id")[target].shift(2)

    # drop first 2 rows of each run (no lag history)
    df = df.dropna(subset=["Lag1", "Lag2"]).reset_index(drop=True)

    # final column order
    cols = [
        "run_id", "Time (HH:MM)", "sin_hour", "cos_hour", "Solar_proxy",
        "Ambient (°C)", "Clay Brick Compartment (°C)", "PCM Tube (°C)",
        "Lag1", "Lag2", target,
    ]
    return df[cols]


def main() -> None:
    for regime, path in RAW.items():
        df = load_and_clean(path)
        out = OUT_DIR / f"{regime}.parquet"
        df.to_parquet(out, index=False)
        print(f"[{regime}] rows={len(df)} runs={df['run_id'].nunique()} -> {out}")
        print(df.head(3).to_string())
        print()


if __name__ == "__main__":
    main()
