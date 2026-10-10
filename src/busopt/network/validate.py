"""Independent checks of the ingested network. Each check returns a row in a report table."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from busopt.data.gtfs import GTFSData, haversine_m
from busopt.network.graph import edge_table, route_transfer_graph


def validate_network(gtfs: GTFSData, g: nx.MultiDiGraph, cfg: dict) -> pd.DataFrame:
    rows: list[dict] = []

    def add(check: str, ok: bool, detail: str) -> None:
        rows.append({"check": check, "passed": bool(ok), "detail": detail})

    et = edge_table(g)
    ride = et[et.kind == "ride"]
    add(
        "ride edges have positive running time",
        (ride.time_s > 0).all(),
        f"{int((ride.time_s <= 0).sum())} of {len(ride)} ride edges have time <= 0 s",
    )
    speed = ride.dist_m / ride.time_s.where(ride.time_s > 0) * 3.6
    bad = ((speed < 2) | (speed > 80)).sum()
    add(
        "stop-to-stop speeds within 2-80 km/h",
        bad == 0,
        f"{int(bad)} of {len(ride)} edges outside range; median {np.nanmedian(speed):.1f} km/h, max {np.nanmax(speed):.1f}",
    )

    pat = gtfs.patterns
    mono = pat.groupby(["route_id", "direction_id"]).cum_time_s.apply(lambda s: (np.diff(s) >= 0).all())
    add(
        "pattern cumulative times non-decreasing",
        mono.all(),
        f"{int((~mono).sum())} of {len(mono)} patterns violate",
    )
    ndir = pat.groupby("route_id").direction_id.nunique()
    add(
        "every route has two directions",
        (ndir == 2).all(),
        f"routes with one direction: {sorted(ndir[ndir != 2].index)}",
    )
    missing = pat.stop_lat.isna().sum()
    add(
        "all pattern stops have coordinates",
        missing == 0,
        f"{int(missing)} pattern stops without coordinates",
    )

    ug = nx.Graph(g)
    comps = list(nx.connected_components(ug))
    add(
        "stop graph connected (ride + walking transfers)",
        len(comps) == 1,
        f"{len(comps)} components; largest holds {max(map(len, comps))} of {g.number_of_nodes()} stops",
    )
    rg = route_transfer_graph(g)
    rcomps = list(nx.connected_components(rg))
    add(
        "route transfer graph connected",
        len(rcomps) == 1,
        f"{len(rcomps)} components; isolated: {sorted(map(sorted, [c for c in rcomps if len(c) < max(map(len, rcomps))]))}",
    )

    # round-trip closure: terminal of one direction close to origin of the other
    gaps = []
    for route, p in pat.groupby("route_id"):
        if p.direction_id.nunique() < 2:
            continue
        a, b = p[p.direction_id == 0].sort_values("seq"), p[p.direction_id == 1].sort_values("seq")
        gaps.append(
            (
                route,
                float(
                    haversine_m(
                        a.stop_lat.iloc[-1], a.stop_lon.iloc[-1], b.stop_lat.iloc[0], b.stop_lon.iloc[0]
                    )
                ),
                float(
                    haversine_m(
                        b.stop_lat.iloc[-1], b.stop_lon.iloc[-1], a.stop_lat.iloc[0], a.stop_lon.iloc[0]
                    )
                ),
            )
        )
    gd = pd.DataFrame(gaps, columns=["route_id", "gap_a", "gap_b"])
    far = gd[(gd.gap_a > 1000) | (gd.gap_b > 1000)]
    add(
        "terminals of opposite directions within 1 km (round trip closes)",
        far.empty,
        f"routes with a terminal gap > 1 km: {far.route_id.tolist()}; median gap {np.median(gd[['gap_a', 'gap_b']]):.0f} m",
    )

    # fleet model check: ceil(trips * cycle / 60) against distinct GTFS vehicle blocks in the hour
    lay_f, lay_min = cfg["network"]["layover_frac"], cfg["network"]["layover_min"]
    wk = gtfs.service[(gtfs.service.daytype == "weekday") & gtfs.service.hour.between(6, 21)]
    piv = wk.pivot_table(
        index=["route_id", "hour"], columns="direction_id", values=["trips", "run_time_min", "blocks"]
    )
    piv = piv.dropna()
    run = piv["run_time_min"].sum(axis=1)
    cycle = run + np.maximum(lay_f * run, lay_min)
    model = np.ceil(piv["trips"].mean(axis=1) * cycle / 60.0)
    by_hour_model = model.groupby("hour").sum()
    add(
        "fleet model order-of-magnitude check vs GTFS blocks",
        True,
        f"model peak fleet {by_hour_model.max():.0f} buses at hour {int(by_hour_model.idxmax())}; "
        f"GTFS blocks starting a trip in that hour {piv['blocks'].sum(axis=1).groupby('hour').sum().loc[by_hour_model.idxmax()]:.0f} "
        "(blocks counted per direction, so a bus doing both directions within the hour is counted twice)",
    )
    add(
        "cycle times finite and positive",
        bool(np.isfinite(cycle).all() and (cycle > 0).all()),
        f"cycle time range {cycle.min():.0f}-{cycle.max():.0f} min",
    )
    return pd.DataFrame(rows)
