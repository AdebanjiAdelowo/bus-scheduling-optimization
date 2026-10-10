import networkx as nx
import numpy as np
import pandas as pd

from busopt.data.demand import build_panel, build_profiles
from busopt.data.gtfs import haversine_m, load_gtfs
from busopt.network.graph import build_graph, route_transfer_graph
from busopt.network.validate import validate_network
from busopt.synthetic import toy_gtfs_zip


def _gtfs(tmp_path):
    p = tmp_path / "toy.zip"
    p.write_bytes(toy_gtfs_zip())
    return load_gtfs(p, "M")


def test_gtfs_ingestion(tmp_path):
    g = _gtfs(tmp_path)
    assert set(g.patterns.route_id) == {"M1", "M2"}
    assert g.patterns.groupby(["route_id", "direction_id"]).size().eq(6).all()
    wk = g.service[
        (g.service.daytype == "weekday") & (g.service.route_id == "M1") & (g.service.direction_id == 0)
    ]
    assert wk.set_index("hour").trips[8] == 6 and wk.set_index("hour").trips[10] == 4
    sat = g.service[
        (g.service.daytype == "saturday") & (g.service.route_id == "M1") & (g.service.direction_id == 0)
    ]
    assert sat.set_index("hour").trips[10] == 2
    assert np.isclose(g.patterns.cum_time_s.max(), 900.0)
    assert set(g.reference_dates) == {"weekday", "saturday", "sunday"}


def test_graph_and_validation(tmp_path, cfg):
    g = _gtfs(tmp_path)
    graph = build_graph(g, 150.0)
    ride = [d for _, _, d in graph.edges(data=True) if d["kind"] == "ride"]
    assert len(ride) == 4 * 5 and all(d["time_s"] == 180.0 for d in ride)
    assert any(d["kind"] == "transfer" for _, _, d in graph.edges(data=True))
    assert nx.is_connected(route_transfer_graph(graph))
    report = validate_network(g, graph, cfg)
    assert report.passed.all(), report[~report.passed]


def test_haversine():
    assert abs(haversine_m(40.0, -74.0, 40.0 + 1 / 111.195, -74.0) - 1000.0) < 1.0


def test_panel_marks_outage_days_missing():
    ts = pd.date_range("2025-01-01", periods=72, freq="h")
    fare = pd.DataFrame({"bus_route": "M1", "transit_timestamp": ts, "ridership": 5, "transfers": 1})
    fare.loc[fare.transit_timestamp.dt.day == 2, "ridership"] = 0
    panel = build_panel(fare, ["M1"])
    assert panel[panel.date == "2025-01-02"].ridership.isna().all()
    assert panel[panel.date == "2025-01-01"].ridership.eq(5).all()


def test_profiles_from_apc(tmp_path):
    g = _gtfs(tmp_path)
    rows = []
    for d, compass in ((0, "N"), (1, "S")):
        p = g.patterns[(g.patterns.route_id == "M1") & (g.patterns.direction_id == d)]
        for s, sid in enumerate(p.stop_id):
            for dow in range(7):
                rows.append(
                    {
                        "route_id": "M1",
                        "direction": compass,
                        "stop_id": int(sid),
                        "stop_sequence": s + 1,
                        "dow": dow,
                        "hod": 8,
                        "boardings": 500 if s < 3 else 0,
                        "alightings": 0 if s < 3 else 500,
                        "trips": 10,
                        "n_obs": 5,
                    }
                )
    shares, summ, dmap = build_profiles(pd.DataFrame(rows), g)
    assert dmap.set_index("direction").direction_id.to_dict() == {"N": 0, "S": 1}
    s = summ[(summ.route_id == "M1") & (summ.hour == 8) & (summ.daytype == "weekday")]
    assert np.allclose(s.rho, 1.0)  # everyone boards in the first half and alights in the second
    assert np.allclose(s.sigma, 0.5)
    sh = shares[(shares.route_id == "M1") & (shares.hour == 8)]
    assert np.allclose(sh.groupby(["direction_id", "daytype"]).board_share.sum(), 1.0)
