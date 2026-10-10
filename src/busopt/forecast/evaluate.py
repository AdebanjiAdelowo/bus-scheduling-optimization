"""Forecast comparison (RQ1). Run: ``python -m busopt.forecast.evaluate``.

Temporal split: train <= train_end < validation <= valid_end < test. Non-learned baselines use only
lagged values; learned models are fit on train (tuned on validation) and refit on train+validation.
The test period is touched once, for the final table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from busopt.config import load_config, resolve
from busopt.data.build import load_tables
from busopt.forecast.features import make_features
from busopt.forecast.models import GBTModel, HistoricalAverage, RidgeModel, RollingAverage, SeasonalNaive
from busopt.forecast.neural import GRUForecaster
from busopt.tracking import Run


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e = p - y
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "wape": float(np.abs(e).sum() / max(np.abs(y).sum(), 1e-9)),
        "bias": float(e.mean()),
        "n": int(len(y)),
    }


def split(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    f = cfg["forecast"]
    te, ve = (
        pd.Timestamp(f["train_end"]) + pd.Timedelta(hours=23),
        pd.Timestamp(f["valid_end"]) + pd.Timedelta(hours=23),
    )
    # drop outage rows (no target) and the first weeks, which have no same-weekday history
    frame = frame[frame.y.notna() & frame.mean_w4.notna()]
    return frame[frame.ts <= te], frame[(frame.ts > te) & (frame.ts <= ve)], frame[frame.ts > ve]


def build_models(seed: int) -> list:
    return [
        SeasonalNaive(),
        HistoricalAverage(),
        RollingAverage(),
        RidgeModel(False),
        RidgeModel(True),
        GBTModel(False, seed),
        GBTModel(True, seed),
    ]


def run(
    cfg: dict | None = None, horizons: list[int] | None = None, save: bool = True, neural: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_config()
    tables = load_tables(cfg)
    tracker = Run("forecast", cfg)
    hours = range(*cfg["network"]["service_hours"])
    rows, preds, valid_preds = [], [], []
    for h in horizons or cfg["forecast"]["horizons_days"]:
        frame = make_features(tables["panel"], tables["weather"], h, cfg["forecast"]["train_end"])
        train, valid, test = split(frame, cfg)
        out = test[["ts", "route_id", "y", "hour", "daytype", "holiday"]].copy()
        out["horizon"] = h
        models = build_models(cfg["forecast"]["seed"])
        if neural:
            models.append(GRUForecaster(h, cfg["forecast"]["seed"]))
        for model in models:
            if isinstance(model, GRUForecaster):
                gp = model.fit_predict(
                    tables["panel"], cfg["forecast"]["train_end"], cfg["forecast"]["valid_end"]
                )
                p = (
                    test[["ts", "route_id"]]
                    .merge(gp, on=["ts", "route_id"], how="left")[model.name]
                    .to_numpy()
                )
                tracker.log(**{f"gru_h{h}_best_epoch": model.best_epoch})
            else:
                model.fit(train, valid)
                p = model.predict(test)
            if model.name == "gbt":
                vp = valid[["ts", "route_id", "y", "hour", "daytype"]].copy()
                vp["gbt"], vp["horizon"] = model.valid_pred, h
                valid_preds.append(vp)
            out[model.name] = p
            svc = test.hour.isin(hours).to_numpy()
            base = {"horizon_days": h, "model": model.name}
            rows.append({**base, "slice": "all_hours", **metrics(test.y.to_numpy(), p)})
            rows.append({**base, "slice": "service_hours", **metrics(test.y.to_numpy()[svc], p[svc])})
            for dt, name in enumerate(("weekday", "saturday", "sunday")):
                m = svc & (test.daytype.to_numpy() == dt)
                rows.append({**base, "slice": name, **metrics(test.y.to_numpy()[m], p[m])})
            m = svc & (test.holiday.to_numpy() == 1)
            if m.any():
                rows.append({**base, "slice": "holiday", **metrics(test.y.to_numpy()[m], p[m])})
            pk = test.hour.isin([7, 8, 9, 16, 17, 18]).to_numpy() & (test.daytype.to_numpy() == 0)
            rows.append({**base, "slice": "weekday_peak", **metrics(test.y.to_numpy()[pk], p[pk])})
            if isinstance(model, GBTModel):
                tracker.log(**{f"{model.name}_h{h}_best_iter": model.best_iter})
            print(
                f"h={h} {model.name:24s} service-hours MAE={metrics(test.y.to_numpy()[svc], p[svc])['mae']:.2f}"
            )
        preds.append(out)
    table = pd.DataFrame(rows)
    pred = pd.concat(preds, ignore_index=True)
    if save:
        res = resolve("results")
        res.mkdir(exist_ok=True)
        table.to_csv(res / "forecast_metrics.csv", index=False)
        pred.to_parquet(
            resolve(cfg["data"]["processed_dir"]) / "forecast_test_predictions.parquet", index=False
        )
        pd.concat(valid_preds, ignore_index=True).to_parquet(
            resolve(cfg["data"]["processed_dir"]) / "forecast_valid_predictions.parquet", index=False
        )
        by_hour = pred.assign(
            err=lambda d: (d.gbt - d.y).abs(), err_hist=lambda d: (d.historical_average - d.y).abs()
        )
        by_hour.groupby(["horizon", "hour"])[["err", "err_hist", "y"]].mean().rename(
            columns={"err": "gbt_mae", "err_hist": "historical_average_mae", "y": "mean_demand"}
        ).reset_index().to_csv(res / "forecast_error_by_hour.csv", index=False)
        tracker.log(n_test_rows=int(len(pred)), test_start=str(pred.ts.min()), test_end=str(pred.ts.max()))
        tracker.finish()
    return table, pred


if __name__ == "__main__":
    t, _ = run()
    pd.set_option("display.width", 200)
    print(
        t[t.slice == "service_hours"]
        .pivot(index="model", columns="horizon_days", values=["mae", "rmse", "wape"])
        .round(3)
    )
