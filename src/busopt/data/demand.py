"""Observed demand tables.

Two observed sources are combined, each used only for what it can support:

* **Fare-payment ridership** (route x hour, complete time series) is the demand *level*. It counts
  OMNY/MetroCard entries, so it excludes passengers who did not pay; it is a lower bound on true
  boardings and is treated throughout as "fare-paying demand".
* **APC stop-level counts** (route x direction x stop x day-of-week x hour of day, training window
  only) supply demand *shape*: the directional split, the share of boardings and alightings at each
  stop and hence the ratio of peak passenger load to boardings.

Stop-level demand used downstream is therefore a *model-based disaggregation* of observed route
totals with observed shape profiles; it is not an observed stop-level time series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from busopt.data.gtfs import DAYTYPES, GTFSData

MIN_PROFILE_BOARDINGS = 300  # below this an (hour) profile falls back to the day-type profile


def daytype_of(ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    dow = pd.DatetimeIndex(ts).dayofweek
    return np.where(dow < 5, "weekday", np.where(dow == 5, "saturday", "sunday"))


def build_panel(fare: pd.DataFrame, routes: list[str]) -> pd.DataFrame:
    """Complete route x hour panel of fare-paying boardings. Outage days are NaN, never imputed."""
    f = fare[fare.bus_route.isin(routes)].rename(columns={"bus_route": "route_id", "transit_timestamp": "ts"})
    idx = pd.MultiIndex.from_product(
        [sorted(routes), pd.date_range(f.ts.min().floor("D"), f.ts.max(), freq="h")], names=["route_id", "ts"])
    panel = f.set_index(["route_id", "ts"])[["ridership", "transfers"]].reindex(idx).reset_index()
    panel["date"] = panel.ts.dt.normalize()
    # Days on which the whole network reports zero fare payments are reporting outages, not zero
    # demand. They are set to missing and excluded from training and evaluation.
    day_total = panel.groupby("date").ridership.transform("sum")
    panel["outage"] = day_total == 0
    panel.loc[panel.outage, ["ridership", "transfers"]] = np.nan
    panel["hour"] = panel.ts.dt.hour
    panel["daytype"] = daytype_of(panel.ts)
    return panel


def map_apc_directions(apc: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    """Match APC compass directions to GTFS ``direction_id`` by stop-set overlap."""
    rows = []
    pat_sets = {k: set(v.stop_id) for k, v in patterns.groupby(["route_id", "direction_id"])}
    for (route, d), sub in apc.groupby(["route_id", "direction"]):
        stops = set(sub.stop_id.astype(str))
        scores = {gd: len(stops & pat_sets.get((route, gd), set())) for gd in (0, 1)}
        best = max(scores, key=scores.get)
        rows.append({"route_id": route, "direction": d, "direction_id": best,
                     "overlap": scores[best], "other_overlap": scores[1 - best], "apc_stops": len(stops)})
    return pd.DataFrame(rows)


def build_profiles(apc: pd.DataFrame, gtfs: GTFSData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (stop_shares, summary, direction_map).

    ``stop_shares``: route, direction_id, daytype, hour, seq, stop_id, board_share, alight_share.
    ``summary``: route, direction_id, daytype, hour, sigma (direction share of route boardings),
    rho (peak load / boardings), apc_boardings, fallback (bool).
    """
    apc = apc.copy()
    apc["stop_id"] = apc.stop_id.astype(str)
    apc["daytype"] = np.where(apc.dow == 0, "sunday", np.where(apc.dow == 6, "saturday", "weekday"))
    dmap = map_apc_directions(apc, gtfs.patterns)
    apc = apc.merge(dmap[["route_id", "direction", "direction_id"]], on=["route_id", "direction"])
    pat = gtfs.patterns[["route_id", "direction_id", "seq", "stop_id"]]
    total_b = apc.groupby("route_id").boardings.sum()
    apc = apc.merge(pat, on=["route_id", "direction_id", "stop_id"], how="inner")
    coverage = (apc.groupby("route_id").boardings.sum() / total_b).rename("pattern_coverage")
    dmap = dmap.merge(coverage, on="route_id", how="left")

    cell = apc.groupby(["route_id", "direction_id", "daytype", "hod", "seq"])[["boardings", "alightings"]].sum()
    day = apc.groupby(["route_id", "direction_id", "daytype", "seq"])[["boardings", "alightings"]].sum()
    hours = range(24)
    share_rows, sum_rows = [], []
    for (route, gd), p in pat.groupby(["route_id", "direction_id"]):
        n = len(p)
        for dt in DAYTYPES:
            try:
                dprof = day.loc[(route, gd, dt)].reindex(range(n), fill_value=0.0).to_numpy(float)
            except KeyError:
                dprof = np.zeros((n, 2))
            for h in hours:
                try:
                    hp = cell.loc[(route, gd, dt, h)].reindex(range(n), fill_value=0.0).to_numpy(float)
                except KeyError:
                    hp = np.zeros((n, 2))
                apc_b = hp[:, 0].sum()
                fallback = apc_b < MIN_PROFILE_BOARDINGS
                prof = dprof if fallback else hp
                b, a = prof[:, 0].copy(), prof[:, 1].copy()
                if b.sum() <= 0:  # no APC information at all: uniform boardings, alight uniformly downstream
                    b = np.ones(n)
                    b[-1] = 0.0
                    a = np.ones(n)
                    a[0] = 0.0
                b[-1] = 0.0  # nobody boards at the terminal
                a[0] = 0.0
                bs = b / b.sum()
                as_ = a / a.sum() if a.sum() > 0 else np.r_[np.zeros(n - 1), 1.0]
                load = np.clip(np.cumsum(bs - as_), 0.0, None)
                sum_rows.append({"route_id": route, "direction_id": gd, "daytype": dt, "hour": h,
                                 "rho": float(load.max()), "apc_boardings": float(apc_b), "fallback": bool(fallback)})
                share_rows.append(pd.DataFrame({"route_id": route, "direction_id": gd, "daytype": dt, "hour": h,
                                                "seq": np.arange(n), "stop_id": p.stop_id.to_numpy(),
                                                "board_share": bs, "alight_share": as_}))
    summary = pd.DataFrame(sum_rows)
    tot = summary.groupby(["route_id", "daytype", "hour"]).apc_boardings.transform("sum")
    day_b = summary.groupby(["route_id", "direction_id", "daytype"]).apc_boardings.transform("sum")
    day_tot = summary.groupby(["route_id", "daytype"]).apc_boardings.transform("sum")
    day_sigma = np.where(day_tot > 0, day_b / day_tot.where(day_tot > 0, 1.0), 0.5)
    summary["sigma"] = np.where(tot >= MIN_PROFILE_BOARDINGS, summary.apc_boardings / tot.where(tot > 0, 1.0), day_sigma)
    return pd.concat(share_rows, ignore_index=True), summary, dmap
