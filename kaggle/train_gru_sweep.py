"""Optional GPU sweep for the GRU forecaster (Kaggle or any CUDA/MPS machine).

STATUS: PENDING. Not run for the reported results. The default GRU (hidden 64, one seed) trains on
CPU in about a minute and is what `results/forecast_metrics.csv` reports. This sweep exists to test
whether a larger model or several seeds change the ranking against gradient-boosted trees.

Usage on Kaggle (GPU runtime), after uploading `data/processed/panel.parquet` as a dataset:

    pip install -q "git+file:///kaggle/working/bus-scheduling-optimization"   # or pip install -e .
    python kaggle/train_gru_sweep.py --panel /kaggle/input/busopt-panel/panel.parquet

Local equivalent:  uv run --extra neural python kaggle/train_gru_sweep.py --panel data/processed/panel.parquet
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from busopt.config import load_config
from busopt.forecast.evaluate import metrics
from busopt.forecast.neural import GRUForecaster


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", default="gru_sweep.json")
    args = ap.parse_args()
    cfg = load_config()
    panel = pd.read_parquet(args.panel)
    hours = range(*cfg["network"]["service_hours"])
    rows = []
    for h, hidden, seed in itertools.product(cfg["forecast"]["horizons_days"], (64, 128, 256), (17, 18, 19)):
        model = GRUForecaster(h, seed=seed, hidden=hidden, max_epochs=60)
        pred = model.fit_predict(panel, cfg["forecast"]["train_end"], cfg["forecast"]["valid_end"])
        m = pred.merge(panel[["ts", "route_id", "ridership"]], on=["ts", "route_id"]).dropna()
        m = m[m.ts.dt.hour.isin(hours)]
        rows.append(
            {
                "horizon": h,
                "hidden": hidden,
                "seed": seed,
                "best_epoch": model.best_epoch,
                **metrics(m.ridership.to_numpy(), m.gru.to_numpy()),
            }
        )
        print(rows[-1], flush=True)
    json.dump(rows, open(args.out, "w"), indent=2)
    df = pd.DataFrame(rows)
    print(df.groupby(["horizon", "hidden"]).mae.agg(["mean", "std"]).round(3))
    print("seed-to-seed MAE std:", float(np.mean(df.groupby(["horizon", "hidden"]).mae.std())))


if __name__ == "__main__":
    main()
