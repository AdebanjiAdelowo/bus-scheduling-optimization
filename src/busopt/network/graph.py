"""Transit network as a directed multigraph.

Nodes are stops. ``ride`` edges join consecutive stops of a route pattern and carry the scheduled
running time and great-circle distance. ``transfer`` edges join stops of *different* routes that lie
within a walking radius, with a walk time at 1.2 m/s.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from busopt.data.gtfs import GTFSData, haversine_m

WALK_SPEED_MPS = 1.2


def build_graph(gtfs: GTFSData, transfer_radius_m: float = 150.0) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph(feed_version=gtfs.feed_info.get("feed_version"))
    stop_routes: dict[str, set[str]] = {}
    for row in gtfs.stops.itertuples():
        g.add_node(row.stop_id, name=row.stop_name, lat=row.stop_lat, lon=row.stop_lon)
    for (route, d), p in gtfs.patterns.groupby(["route_id", "direction_id"]):
        p = p.sort_values("seq")
        ids, t, dist = p.stop_id.to_numpy(), p.cum_time_s.to_numpy(float), p.cum_dist_m.to_numpy(float)
        for i in range(len(p) - 1):
            g.add_edge(
                ids[i],
                ids[i + 1],
                key=f"{route}:{d}",
                kind="ride",
                route_id=route,
                direction_id=int(d),
                time_s=float(t[i + 1] - t[i]),
                dist_m=float(dist[i + 1] - dist[i]),
            )
        for s in ids:
            stop_routes.setdefault(s, set()).add(route)

    s = gtfs.stops.reset_index(drop=True)
    lat, lon = s.stop_lat.to_numpy(), s.stop_lon.to_numpy()
    d = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    ii, jj = np.where((d <= transfer_radius_m) & (d >= 0) & ~np.eye(len(s), dtype=bool))
    for i, j in zip(ii, jj, strict=True):
        a, b = s.stop_id[i], s.stop_id[j]
        if stop_routes.get(a, set()) != stop_routes.get(
            b, set()
        ):  # walking only helps if it reaches another route
            g.add_edge(
                a,
                b,
                key="walk",
                kind="transfer",
                time_s=float(d[i, j] / WALK_SPEED_MPS),
                dist_m=float(d[i, j]),
            )
    return g


def route_transfer_graph(g: nx.MultiDiGraph) -> nx.Graph:
    """Route-level graph: two routes are adjacent if a passenger can change between them."""
    rg = nx.Graph()
    node_routes: dict[str, set[str]] = {}
    for u, v, data in g.edges(data=True):
        if data["kind"] == "ride":
            node_routes.setdefault(u, set()).add(data["route_id"])
            node_routes.setdefault(v, set()).add(data["route_id"])
            rg.add_node(data["route_id"])
    for routes in node_routes.values():  # shared stop
        rl = sorted(routes)
        for i in range(len(rl)):
            for j in range(i + 1, len(rl)):
                rg.add_edge(rl[i], rl[j])
    for u, v, data in g.edges(data=True):  # walking transfer
        if data["kind"] == "transfer":
            for a in node_routes.get(u, ()):
                for b in node_routes.get(v, ()):
                    if a != b:
                        rg.add_edge(a, b)
    return rg


def edge_table(g: nx.MultiDiGraph) -> pd.DataFrame:
    return pd.DataFrame([{"u": u, "v": v, **d} for u, v, d in g.edges(data=True)])
