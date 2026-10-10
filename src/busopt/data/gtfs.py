"""GTFS static ingestion.

Turns the raw feed into the small set of tables the planning model needs:

* ``patterns``: for every (route, direction) the dominant stop pattern with cumulative scheduled
  running time and distance;
* ``service``: for every (route, day type, hour) the scheduled trips per direction, one-way running
  times and the number of distinct vehicle blocks in service;
* ``stops`` and ``routes``.

Day types are resolved through ``calendar.txt``/``calendar_dates.txt`` on three reference dates
(one weekday, one Saturday, one Sunday) inside the feed validity window.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DAYTYPES = ("weekday", "saturday", "sunday")
_DOW_COL = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _secs(t: pd.Series) -> pd.Series:
    parts = t.str.split(":", expand=True).astype(int)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


@dataclass
class GTFSData:
    routes: pd.DataFrame
    stops: pd.DataFrame
    patterns: pd.DataFrame  # route_id, direction_id, seq, stop_id, cum_time_s, cum_dist_m
    service: pd.DataFrame  # route_id, daytype, hour, direction_id, trips, run_time_min, blocks
    feed_info: dict
    reference_dates: dict


def active_services(cal: pd.DataFrame, cal_dates: pd.DataFrame, date: pd.Timestamp) -> set[str]:
    ymd = int(date.strftime("%Y%m%d"))
    col = _DOW_COL[date.dayofweek]
    base = cal[(cal[col] == "1") & (cal.start_date.astype(int) <= ymd) & (cal.end_date.astype(int) >= ymd)]
    services = set(base.service_id)
    ex = cal_dates[cal_dates.date.astype(int) == ymd]
    services |= set(ex[ex.exception_type == "1"].service_id)
    services -= set(ex[ex.exception_type == "2"].service_id)
    return services


def pick_reference_dates(cal: pd.DataFrame, cal_dates: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """First ordinary weekday (Tuesday), Saturday and Sunday with no calendar exception at all."""
    start = pd.Timestamp(str(cal.start_date.astype(int).min()))
    end = pd.Timestamp(str(cal.end_date.astype(int).max()))
    exception_days = set(cal_dates.date.astype(int))
    want = {"weekday": 1, "saturday": 5, "sunday": 6}
    out: dict[str, pd.Timestamp] = {}
    for day in pd.date_range(start + pd.Timedelta(days=7), end):
        if int(day.strftime("%Y%m%d")) in exception_days:
            continue
        for name, dow in want.items():
            if name not in out and day.dayofweek == dow:
                out[name] = day
        if len(out) == 3:
            break
    return out


def load_gtfs(zip_path: str | Path, route_prefix: str = "M") -> GTFSData:
    z = zipfile.ZipFile(zip_path)

    def rd(name: str) -> pd.DataFrame:
        return pd.read_csv(z.open(name), dtype=str)

    routes = rd("routes.txt")
    routes = routes[routes.route_id.str.startswith(route_prefix)].reset_index(drop=True)
    trips = rd("trips.txt")
    trips = trips[trips.route_id.isin(routes.route_id)]
    cal, cal_dates = rd("calendar.txt"), rd("calendar_dates.txt")
    feed_info = rd("feed_info.txt").iloc[0].to_dict() if "feed_info.txt" in z.namelist() else {}
    stops = rd("stops.txt")[["stop_id", "stop_name", "stop_lat", "stop_lon"]]
    stops[["stop_lat", "stop_lon"]] = stops[["stop_lat", "stop_lon"]].astype(float)

    st = pd.read_csv(z.open("stop_times.txt"), dtype=str,
                     usecols=["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"])
    st = st[st.trip_id.isin(trips.trip_id)].copy()
    st["arr"] = _secs(st.arrival_time)
    st["dep"] = _secs(st.departure_time)
    st["stop_sequence"] = st.stop_sequence.astype(int)
    st = st.sort_values(["trip_id", "stop_sequence"])

    ref = pick_reference_dates(cal, cal_dates)
    svc_daytype: dict[str, list[str]] = {dt: sorted(active_services(cal, cal_dates, d)) for dt, d in ref.items()}

    g = st.groupby("trip_id", sort=False)
    trip_summary = pd.DataFrame({
        "start": g.dep.first(), "end": g.arr.last(), "n_stops": g.size(),
        "pattern": g.stop_id.agg(lambda s: hash(tuple(s))),
    }).reset_index().merge(trips[["trip_id", "route_id", "direction_id", "service_id", "block_id"]], on="trip_id")
    trip_summary["run_min"] = (trip_summary.end - trip_summary.start) / 60.0
    trip_summary["hour"] = trip_summary.start // 3600

    # dominant pattern per route-direction, over weekday trips
    wk = trip_summary[trip_summary.service_id.isin(svc_daytype["weekday"])]
    dom = (wk.groupby(["route_id", "direction_id", "pattern"]).size().rename("n").reset_index()
             .sort_values("n", ascending=False).drop_duplicates(["route_id", "direction_id"]))
    pat_rows = []
    for row in dom.itertuples():
        tids = wk[(wk.route_id == row.route_id) & (wk.direction_id == row.direction_id) & (wk.pattern == row.pattern)].trip_id
        sub = st[st.trip_id.isin(tids)].copy()
        sub["seq"] = sub.groupby("trip_id").cumcount()
        sub["rel"] = sub.arr - sub.groupby("trip_id").dep.transform("first")
        med = sub.groupby("seq").agg(stop_id=("stop_id", "first"), cum_time_s=("rel", "median")).reset_index()
        med["route_id"], med["direction_id"] = row.route_id, int(row.direction_id)
        med["pattern_share"] = row.n / len(wk[(wk.route_id == row.route_id) & (wk.direction_id == row.direction_id)])
        pat_rows.append(med)
    patterns = pd.concat(pat_rows, ignore_index=True).merge(stops, on="stop_id", how="left")
    patterns = patterns.sort_values(["route_id", "direction_id", "seq"]).reset_index(drop=True)
    grp = patterns.groupby(["route_id", "direction_id"])
    seg = haversine_m(patterns.stop_lat, patterns.stop_lon, grp.stop_lat.shift(), grp.stop_lon.shift())
    patterns["cum_dist_m"] = pd.Series(seg).fillna(0.0).groupby([patterns.route_id, patterns.direction_id]).cumsum()

    svc_rows = []
    for dt, services in svc_daytype.items():
        sub = trip_summary[trip_summary.service_id.isin(services)]
        agg = sub.groupby(["route_id", "direction_id", "hour"]).agg(
            trips=("trip_id", "size"), run_time_min=("run_min", "median"), blocks=("block_id", "nunique")).reset_index()
        agg["daytype"] = dt
        svc_rows.append(agg)
    service = pd.concat(svc_rows, ignore_index=True)
    service["direction_id"] = service.direction_id.astype(int)
    service["hour"] = service.hour.astype(int)

    return GTFSData(routes=routes, stops=stops[stops.stop_id.isin(patterns.stop_id)].reset_index(drop=True),
                    patterns=patterns, service=service, feed_info=feed_info,
                    reference_dates={k: str(v.date()) for k, v in ref.items()})
