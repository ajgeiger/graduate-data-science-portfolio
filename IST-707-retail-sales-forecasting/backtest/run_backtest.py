"""Run the rolling-origin backtest and print the results table.

Usage, from the project folder:
    python backtest/run_backtest.py --data data/raw
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import M5Data, run_origin  # noqa: E402

# last known day for each origin; each forecasts the next 28 days
ORIGINS = [1857, 1885, 1913]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--rounds", type=int, default=1250)
    parser.add_argument("--out", default="backtest/results.csv")
    args = parser.parse_args()

    print("loading data", flush=True)
    data = M5Data(args.data)

    results = []
    for origin in ORIGINS:
        res, imp = run_origin(data, origin, args.train_days, args.rounds,
                              log=lambda m: print(m, flush=True))
        print(res.round(4).to_string(index=False), flush=True)
        results.append(res)
        if origin == ORIGINS[-1]:
            imp.sort_values(ascending=False).to_csv(
                args.out.replace(".csv", "_importance.csv"), header=["splits"])

    table = pd.concat(results, ignore_index=True)
    mean = (table.groupby("model", sort=False)[["RMSE", "WAPE", "WRMSSE"]]
            .mean().reset_index())
    mean.insert(0, "window", "mean")
    mean.insert(0, "origin", "")
    table = pd.concat([table, mean], ignore_index=True)
    table.round(4).to_csv(args.out, index=False)
    print()
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
