"""Run the rolling-origin backtest and save the results table.

From the project folder:
    python backtest/run_backtest.py --data data/raw
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from m5_forecasting.backtest import M5Data, run_origin  # noqa: E402

# last known day for each origin, each one forecasts the next 28 days
ORIGINS = [1857, 1885, 1913]


def log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--rounds", type=int, default=1250)
    parser.add_argument("--out-dir", default=str(PROJECT_DIR / "backtest"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    log("loading data")
    data = M5Data(args.data)

    origin_results = []
    for origin in ORIGINS:
        results, importance = run_origin(data, origin, args.train_days, args.rounds, log=log)
        log(results.round(4).to_string(index=False))
        origin_results.append(results)

    # feature importance from the last origin's model
    importance.sort_values(ascending=False).to_csv(
        out_dir / "feature_importance.csv", header=["splits"]
    )

    table = pd.concat(origin_results, ignore_index=True)
    means = table.groupby("model", sort=False)[["RMSE", "WAPE", "WRMSSE"]].mean().reset_index()
    means.insert(0, "window", "mean")
    means.insert(0, "origin", "")
    table = pd.concat([table, means], ignore_index=True).round(4)
    table.to_csv(out_dir / "results.csv", index=False)
    log("")
    log(table.to_string(index=False))


if __name__ == "__main__":
    main()
