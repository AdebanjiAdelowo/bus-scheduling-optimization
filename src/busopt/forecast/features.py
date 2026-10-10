"""Leak-free feature construction for day-ahead demand forecasting.

A forecast for every hour of target day D is issued at the end of day D - h (horizon h days).
Every demand-derived feature therefore uses observations dated D - h or earlier. This is enforced
structurally: all lags are whole multiples of 24 hours with a minimum of 24 * h hours, and rolling
statistics are computed on series already shifted by 24 * h hours.

Calendar features (hour, weekday, public holiday) are known in advance. Weather features are the
*observed* weather of the target hour, which no forecaster has in advance; models using them are
labelled "weather oracle" and reported separately as an optimistic bound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

WEATHER_COLS = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m"]


def holiday_flags(dates: pd.DatetimeIndex) -> np.ndarray:
    hol = USFederalHolidayCalendar().holidays(start=dates.min(), end=dates.max())
    return dates.normalize().isin(hol)


def wide(panel: pd.DataFrame, value: str = "ridership") -> pd.DataFrame:
    return panel.pivot(index="ts", columns="route_id", values=value).sort_index().asfreq("h")


def make_features(
    panel: pd.DataFrame, weather: pd.DataFrame, horizon_days: int, train_end: str
) -> pd.DataFrame:
    y = wide(panel)
    h = 24 * horizon_days
    m = int(np.ceil(horizon_days / 7.0))  # weeks back to the most recent admissible same weekday
    feats: dict[str, pd.DataFrame] = {}
    for j in range(4):
        feats[f"lag_w{j}"] = y.shift(24 * 7 * (m + j))
    feats["lag_recent"] = y.shift(h)
    feats["lag_recent2"] = y.shift(h + 24)
    base = y.shift(h)  # everything below is built on data dated <= D - h
    feats["roll7_same_hour"] = (
        pd.concat([base.shift(24 * k) for k in range(7)], axis=0).groupby(level=0).mean()
    )
    daily = y.groupby(y.index.normalize()).transform(
        lambda s: s.sum(min_count=1)
    )  # total of the calendar day
    feats["day_total_recent"] = daily.shift(h)
    feats["day_total_w0"] = daily.shift(24 * 7 * m)
    same_wd = pd.concat([feats[f"lag_w{j}"] for j in range(4)], axis=0).groupby(level=0)
    feats["mean_w4"] = same_wd.mean()

    long = pd.concat({k: v.stack(future_stack=True) for k, v in feats.items()}, axis=1)
    long["y"] = y.stack(future_stack=True)
    long = long.reset_index().rename(columns={"level_0": "ts"})
    ts = pd.DatetimeIndex(long.ts)
    long["hour"] = ts.hour
    long["dow"] = ts.dayofweek
    long["month"] = ts.month
    long["doy_sin"] = np.sin(2 * np.pi * ts.dayofyear / 365.25)
    long["doy_cos"] = np.cos(2 * np.pi * ts.dayofyear / 365.25)
    long["holiday"] = holiday_flags(ts).astype(int)
    long["holiday_w0"] = holiday_flags(ts - pd.Timedelta(days=7 * m)).astype(
        int
    )  # was the lag day a holiday?
    long["daytype"] = np.where(long.dow < 5, 0, np.where(long.dow == 5, 1, 2))

    # static profile from the training window only
    tr = long[long.ts <= pd.Timestamp(train_end) + pd.Timedelta(hours=23)]
    prof = tr.groupby(["route_id", "daytype", "hour"]).y.mean().rename("train_profile").reset_index()
    long = long.merge(prof, on=["route_id", "daytype", "hour"], how="left")
    long = long.merge(weather.rename(columns={"time": "ts"})[["ts", *WEATHER_COLS]], on="ts", how="left")
    long["route_code"] = long.route_id.astype("category").cat.codes
    long["horizon"] = horizon_days
    return long


DEMAND_FEATURES = [
    "lag_w0",
    "lag_w1",
    "lag_w2",
    "lag_w3",
    "lag_recent",
    "lag_recent2",
    "roll7_same_hour",
    "day_total_recent",
    "day_total_w0",
    "mean_w4",
    "train_profile",
]
CALENDAR_FEATURES = ["hour", "dow", "month", "doy_sin", "doy_cos", "holiday", "holiday_w0", "daytype"]
