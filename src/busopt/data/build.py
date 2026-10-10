"""Build processed tables from raw downloads. Run: ``python -m busopt.data.build``."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from busopt.config import load_config, resolve
from busopt.data.demand import build_panel, build_profiles
from busopt.data.gtfs import load_gtfs
from busopt.network.graph import build_graph, edge_table
from busopt.network.validate import validate_network

TABLES = ["panel", "route_info", "route_hour", "patterns", "stop_shares", "profile_summary", "stops", "edges",
          "weather", "direction_map", "network_validation"]


def build(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    raw, out = resolve(cfg["data"]["raw_dir"]), resolve(cfg["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    gtfs = load_gtfs(raw / "gtfs_manhattan.zip", cfg["data"]["route_prefix"])
    fare = pd.read_parquet(raw / "fare_route_hour.parquet")
    apc = pd.read_parquet(raw / "apc_profile.parquet")
    weather = pd.read_parquet(raw / "weather_hourly.parquet")

    fare_tot = fare.groupby("bus_route").ridership.sum()
    gtfs_routes = set(gtfs.patterns.route_id)
    routes = sorted(r for r in gtfs_routes if fare_tot.get(r, 0) > 0)
    panel = build_panel(fare, routes)
    stop_shares, summary, dmap = build_profiles(apc[apc.route_id.isin(routes)], gtfs)

    train = panel[panel.ts <= pd.Timestamp(cfg["forecast"]["train_end"]) + pd.Timedelta(hours=23)]
    n_days = train.date.nunique()
    daily = train.groupby("route_id").ridership.sum() / n_days
    o, n = cfg["optimization"], cfg["network"]
    info = pd.DataFrame({"route_id": routes})
    info["mean_daily_boardings_train"] = info.route_id.map(daily).to_numpy()
    info["articulated"] = info.route_id.str.endswith(o["articulated_suffix"])
    for kind in ("guideline", "crush"):
        c = o[f"capacity_{kind}"]
        info[f"capacity_{kind}"] = np.where(info.articulated, c["articulated"], c["standard"])
    plen = gtfs.patterns.groupby(["route_id", "direction_id"]).agg(length_m=("cum_dist_m", "max"), n_stops=("seq", "size"))
    info["length_km"] = info.route_id.map(plen.groupby("route_id").length_m.mean() / 1000).to_numpy()
    info["n_stops_mean"] = info.route_id.map(plen.groupby("route_id").n_stops.mean()).to_numpy()
    info["in_instance"] = info.mean_daily_boardings_train >= n["min_daily_boardings"]
    info["fare_total"] = info.route_id.map(fare_tot).to_numpy()
    apc_tot = apc.groupby("route_id").boardings.sum()
    fare_train = train.groupby("route_id").ridership.sum()
    info["apc_to_fare_ratio_train"] = (info.route_id.map(apc_tot) / info.route_id.map(fare_train)).to_numpy()

    # route-hour service table
    svc = gtfs.service[gtfs.service.route_id.isin(routes)]
    piv = svc.pivot_table(index=["route_id", "daytype", "hour"], columns="direction_id", values=["trips", "run_time_min"])
    med_run = svc.groupby(["route_id", "daytype", "direction_id"]).run_time_min.median().unstack()
    rh = pd.DataFrame(index=piv.index)
    trips = piv["trips"].reindex(columns=[0, 1]).fillna(0.0)
    rh["gtfs_trips_dir0"], rh["gtfs_trips_dir1"] = trips[0], trips[1]
    rh["gtfs_freq"] = np.round(trips.mean(axis=1) + 1e-9)
    run = piv["run_time_min"].reindex(columns=[0, 1])
    fill = med_run.reindex(run.index.droplevel("hour")).set_axis(run.index)
    run = run.fillna(fill)
    rh["run_min_dir0"], rh["run_min_dir1"] = run[0], run[1]
    tot_run = run.sum(axis=1)
    rh["cycle_min"] = tot_run + np.maximum(n["layover_frac"] * tot_run, n["layover_min"])
    rh = rh.reset_index()
    mean_dem = (train.groupby(["route_id", "daytype", "hour"]).ridership.mean().rename("mean_demand").reset_index())
    rh = rh.merge(mean_dem, on=["route_id", "daytype", "hour"], how="left")
    s = summary.assign(load=summary.rho * summary.sigma)
    peak = s.groupby(["route_id", "daytype", "hour"]).load.max().rename("peak_coef").reset_index()
    rh = rh.merge(peak, on=["route_id", "daytype", "hour"], how="left")

    g = build_graph(gtfs, n["transfer_radius_m"])
    validation = validate_network(gtfs, g, cfg)
    tables = {"panel": panel, "route_info": info, "route_hour": rh, "patterns": gtfs.patterns,
              "stop_shares": stop_shares, "profile_summary": summary, "stops": gtfs.stops, "edges": edge_table(g),
              "weather": weather, "direction_map": dmap, "network_validation": validation}
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    meta = {"feed_info": gtfs.feed_info, "reference_dates": gtfs.reference_dates, "n_routes": len(routes),
            "routes_in_instance": info[info.in_instance].route_id.tolist(),
            "gtfs_routes_without_fare_data": sorted(gtfs_routes - set(routes)),
            "fare_routes_not_in_gtfs": sorted(set(fare_tot[fare_tot > 0].index) - gtfs_routes),
            "panel_start": str(panel.ts.min()), "panel_end": str(panel.ts.max()),
            "panel_missing_cells": int(panel.ridership.isna().sum()),
            "outage_days": sorted({str(d.date()) for d in panel.loc[panel.outage, "date"]}),
            "provenance": json.loads((raw / "PROVENANCE.json").read_text())}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return tables


def load_tables(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    out = resolve(cfg["data"]["processed_dir"])
    return {name: pd.read_parquet(out / f"{name}.parquet") for name in TABLES}


def load_meta(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    return json.loads((resolve(cfg["data"]["processed_dir"]) / "meta.json").read_text())


if __name__ == "__main__":
    t = build()
    print(t["network_validation"].to_string())
    print(t["route_info"].round(2).to_string())
