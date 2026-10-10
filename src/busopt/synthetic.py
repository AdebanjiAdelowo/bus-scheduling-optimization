"""Small synthetic network and demand, used by tests, CI and the offline demo.

Everything produced here is synthetic and is labelled as such wherever it is displayed. It exists so
that the full pipeline (instance -> optimisation -> simulation -> agent tools) can run without any
download; it is never mixed into experiments on observed data.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

DAYTYPES = ("weekday", "saturday", "sunday")


def toy_tables(
    n_routes: int = 4, n_stops: int = 8, hours: tuple[int, int] = (5, 24), seed: int = 0
) -> dict[str, pd.DataFrame]:
    """Processed-table dictionary with the same schema as :func:`busopt.data.build.build`."""
    rng = np.random.default_rng(seed)
    routes = [f"T{i + 1}" for i in range(n_routes)]
    hrs = list(range(24))
    shape = (
        0.25
        + np.exp(-0.5 * ((np.arange(24) - 8) / 1.8) ** 2)
        + 0.9 * np.exp(-0.5 * ((np.arange(24) - 17) / 2.2) ** 2)
    )
    shape[:5] = 0.02
    rh, pat, ss, summ, info = [], [], [], [], []
    for i, r in enumerate(routes):
        level = 250.0 * (1 + i)
        run = 20.0 + 8.0 * i
        info.append(
            {
                "route_id": r,
                "mean_daily_boardings_train": level * shape.sum(),
                "articulated": i == n_routes - 1,
                "capacity_guideline": 85 if i == n_routes - 1 else 54,
                "capacity_crush": 115 if i == n_routes - 1 else 85,
                "length_km": 4.0 + 2 * i,
                "n_stops_mean": n_stops,
                "in_instance": True,
            }
        )
        for d in (0, 1):
            cum = np.linspace(0, run * 60, n_stops)
            for s in range(n_stops):
                pat.append(
                    {
                        "route_id": r,
                        "direction_id": d,
                        "seq": s,
                        "stop_id": f"{r}_{d}_{s}",
                        "cum_time_s": cum[s],
                        "stop_name": f"{r} stop {s}",
                        "stop_lat": 40.7 + 0.01 * i + 0.002 * s * (1 if d == 0 else -1),
                        "stop_lon": -74.0 + 0.004 * s,
                        "cum_dist_m": 400.0 * s,
                        "pattern_share": 1.0,
                    }
                )
            b = np.r_[rng.dirichlet(np.ones(n_stops - 1) * 3), 0.0]
            a = np.r_[0.0, rng.dirichlet(np.ones(n_stops - 1) * 3)]
            rho = float(np.clip(np.cumsum(b - a), 0, None).max())
            for dt in DAYTYPES:
                for h in hrs:
                    summ.append(
                        {
                            "route_id": r,
                            "direction_id": d,
                            "daytype": dt,
                            "hour": h,
                            "rho": rho,
                            "apc_boardings": 1000.0,
                            "fallback": False,
                            "sigma": 0.55 if d == 0 else 0.45,
                        }
                    )
                    for s in range(n_stops):
                        ss.append(
                            {
                                "route_id": r,
                                "direction_id": d,
                                "daytype": dt,
                                "hour": h,
                                "seq": s,
                                "stop_id": f"{r}_{d}_{s}",
                                "board_share": b[s],
                                "alight_share": a[s],
                            }
                        )
        for dt, mult in zip(DAYTYPES, (1.0, 0.7, 0.55), strict=True):
            for h in range(*hours):
                dem = level * shape[h] * mult
                rh.append(
                    {
                        "route_id": r,
                        "daytype": dt,
                        "hour": h,
                        "gtfs_trips_dir0": 0,
                        "gtfs_trips_dir1": 0,
                        "gtfs_freq": float(max(2, round(3 + 5 * shape[h] * mult))),
                        "run_min_dir0": run,
                        "run_min_dir1": run,
                        "cycle_min": 2 * run * 1.1,
                        "mean_demand": dem,
                        "peak_coef": 0.3,
                    }
                )
    return {
        "route_info": pd.DataFrame(info),
        "route_hour": pd.DataFrame(rh),
        "patterns": pd.DataFrame(pat),
        "stop_shares": pd.DataFrame(ss),
        "profile_summary": pd.DataFrame(summ),
    }


def toy_config() -> dict:
    from busopt.config import load_config

    return load_config()


def toy_panel(n_routes: int = 3, n_days: int = 120, seed: int = 0, start: str = "2025-01-01") -> pd.DataFrame:
    """Synthetic hourly ridership panel with weekly seasonality and Poisson noise."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=24 * n_days, freq="h")
    shape = 0.2 + np.exp(-0.5 * ((ts.hour - 8) / 2.0) ** 2) + np.exp(-0.5 * ((ts.hour - 17) / 2.5) ** 2)
    wk = np.where(ts.dayofweek < 5, 1.0, 0.6)
    rows = []
    for i in range(n_routes):
        lam = 100.0 * (i + 1) * shape * wk
        rows.append(
            pd.DataFrame(
                {
                    "route_id": f"T{i + 1}",
                    "ts": ts,
                    "ridership": rng.poisson(lam).astype(float),
                    "transfers": 0.0,
                }
            )
        )
    p = pd.concat(rows, ignore_index=True)
    p["date"], p["hour"] = p.ts.dt.normalize(), p.ts.dt.hour
    p["daytype"] = np.where(
        p.ts.dt.dayofweek < 5, "weekday", np.where(p.ts.dt.dayofweek == 5, "saturday", "sunday")
    )
    p["outage"] = False
    return p


def toy_weather(panel: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.DatetimeIndex(sorted(panel.ts.unique()))
    return pd.DataFrame(
        {
            "time": ts,
            "temperature_2m": 10 + 8 * rng.standard_normal(len(ts)),
            "precipitation": rng.exponential(0.1, len(ts)),
            "snowfall": 0.0,
            "wind_speed_10m": rng.uniform(0, 30, len(ts)),
        }
    )


def toy_gtfs_zip() -> bytes:
    """A minimal valid GTFS feed: two routes, two directions, hourly-varying service on all day types."""
    stops, routes, trips, st = [], [], [], []
    cal = [
        {
            "service_id": s,
            **{
                d: "1" if d in days else "0"
                for d in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            },
            "start_date": "20260101",
            "end_date": "20261231",
        }
        for s, days in (
            ("WK", ["monday", "tuesday", "wednesday", "thursday", "friday"]),
            ("SA", ["saturday"]),
            ("SU", ["sunday"]),
        )
    ]
    for ri, r in enumerate(("M1", "M2")):
        routes.append(
            {"route_id": r, "agency_id": "X", "route_short_name": r, "route_long_name": r, "route_type": "3"}
        )
        for d in (0, 1):
            ids = [f"9{ri}{d}{s:02d}" for s in range(6)]
            for s, sid in enumerate(ids):
                # the two routes cross at stop index 3 (within walking distance)
                lat = 40.70 + 0.003 * s if ri == 0 else 40.709 + 0.0002 * d
                lon = -74.00 + 0.0002 * d if ri == 0 else -74.009 + 0.003 * s
                if d == 1:
                    lat, lon = (40.70 + 0.003 * (5 - s), lon) if ri == 0 else (lat, -74.009 + 0.003 * (5 - s))
                stops.append({"stop_id": sid, "stop_name": f"{r}-{d}-{s}", "stop_lat": lat, "stop_lon": lon})
            for svc, per_hour in (("WK", 4), ("SA", 2), ("SU", 2)):
                for h in range(6, 22):
                    for j in range(per_hour + (2 if svc == "WK" and h in (8, 17) else 0)):
                        tid = f"{r}_{d}_{svc}_{h}_{j}"
                        trips.append(
                            {
                                "route_id": r,
                                "service_id": svc,
                                "trip_id": tid,
                                "direction_id": str(d),
                                "block_id": f"b{tid}",
                            }
                        )
                        t0 = h * 3600 + j * 600
                        for s, sid in enumerate(ids):
                            t = t0 + s * 180
                            hh = f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"
                            st.append(
                                {
                                    "trip_id": tid,
                                    "arrival_time": hh,
                                    "departure_time": hh,
                                    "stop_id": sid,
                                    "stop_sequence": str(s + 1),
                                }
                            )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, rows in (
            ("stops.txt", stops),
            ("routes.txt", routes),
            ("trips.txt", trips),
            ("stop_times.txt", st),
            ("calendar.txt", cal),
        ):
            z.writestr(name, pd.DataFrame(rows).to_csv(index=False))
        z.writestr("calendar_dates.txt", "service_id,date,exception_type\nSU,20260101,1\n")
        z.writestr("feed_info.txt", "feed_publisher_name,feed_version\nsynthetic,toy-1\n")
    return buf.getvalue()
