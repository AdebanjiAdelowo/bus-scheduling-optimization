"""Reproducible downloads of every external input.

Nothing downloaded here is committed to Git. Each download writes a provenance record
(URL, query, timestamp, SHA-256, row count) to ``data/raw/PROVENANCE.json``.

Sources
-------
* GTFS static feed, MTA New York City Transit, Manhattan buses.
* MTA Bus Hourly Ridership (fare payments by route and hour), NY Open Data ``gxb3-akrn``.
* MTA Bus Stop Level Ridership (APC boardings/alightings), NY Open Data ``fvdm-uavx``.
* Open-Meteo historical weather archive (ERA5-based reanalysis).

The two ridership tables are aggregated server-side with SoQL so that only route-level and
profile-level tables are transferred (tens of MB instead of tens of GB).
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from busopt.config import load_config, resolve

TIMEOUT = 180


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _record(raw_dir: Path, name: str, info: dict) -> None:
    prov_path = raw_dir / "PROVENANCE.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    info["retrieved_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    prov[name] = info
    prov_path.write_text(json.dumps(prov, indent=2, sort_keys=True))


def _get(url: str, params: dict | None = None, retries: int = 4) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException as e:  # network errors are retried with backoff
            last = e
            time.sleep(2**attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {url}") from last


def _soql_csv(base: str, dataset: str, params: dict) -> pd.DataFrame:
    params = {**params, "$limit": 2_000_000}
    r = _get(f"{base}/{dataset}.csv", params)
    return pd.read_csv(io.StringIO(r.text))


def download_gtfs(cfg: dict, force: bool = False) -> Path:
    raw = resolve(cfg["data"]["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    out = raw / "gtfs_manhattan.zip"
    if out.exists() and not force:
        return out
    r = _get(cfg["data"]["gtfs_url"])
    out.write_bytes(r.content)
    _record(raw, "gtfs", {"url": cfg["data"]["gtfs_url"], "sha256": _sha256(out), "bytes": out.stat().st_size,
                          "last_modified": r.headers.get("Last-Modified")})
    return out


def download_fare_ridership(cfg: dict, force: bool = False) -> Path:
    """Route x hour fare-payment ridership, summed over payment method and fare class."""
    d = cfg["data"]
    raw = resolve(d["raw_dir"])
    out = raw / "fare_route_hour.parquet"
    if out.exists() and not force:
        return out
    months = pd.period_range(d["start"], d["end"], freq="M")
    parts = []
    for m in months:
        lo = max(m.start_time, pd.Timestamp(d["start"]))
        hi = min(m.end_time, pd.Timestamp(d["end"]) + pd.Timedelta(hours=23, minutes=59, seconds=59))
        df = _soql_csv(d["socrata_base"], d["fare_dataset"], {
            "$select": "bus_route,transit_timestamp,sum(ridership) as ridership,sum(transfers) as transfers",
            "$where": f"starts_with(bus_route,'{d['route_prefix']}') and transit_timestamp between "
                      f"'{lo:%Y-%m-%dT%H:%M:%S}' and '{hi:%Y-%m-%dT%H:%M:%S}'",
            "$group": "bus_route,transit_timestamp",
        })
        print(f"fare {m}: {len(df)} rows")
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full["transit_timestamp"] = pd.to_datetime(full["transit_timestamp"])
    full.to_parquet(out, index=False)
    _record(raw, "fare_route_hour", {"dataset": d["fare_dataset"], "rows": len(full), "sha256": _sha256(out),
                                     "start": d["start"], "end": d["end"],
                                     "aggregation": "sum over payment_method, fare_class_category"})
    return out


def download_apc_profiles(cfg: dict, force: bool = False) -> Path:
    """APC boardings/alightings/trips by route, direction, stop, day-of-week and hour of day.

    Only the *training* window is aggregated so that profile shapes never use test-period data.
    """
    d = cfg["data"]
    raw = resolve(d["raw_dir"])
    out = raw / "apc_profile.parquet"
    if out.exists() and not force:
        return out
    lo, hi = d["start"], cfg["forecast"]["train_end"]
    routes = _soql_csv(d["socrata_base"], d["apc_dataset"], {
        "$select": "route_id,count(*) as n",
        "$where": f"starts_with(route_id,'{d['route_prefix']}') and date between '{lo}T00:00:00' and '{lo}T23:59:59'",
        "$group": "route_id",
    })["route_id"].astype(str).tolist()
    # a single day can miss routes; widen with the following week
    wk = _soql_csv(d["socrata_base"], d["apc_dataset"], {
        "$select": "route_id,count(*) as n",
        "$where": f"starts_with(route_id,'{d['route_prefix']}') and date between '2025-03-03T00:00:00' and '2025-03-09T23:59:59'",
        "$group": "route_id",
    })["route_id"].astype(str).tolist()
    routes = sorted(set(routes) | set(wk))
    parts = []
    for r in routes:
        df = _soql_csv(d["socrata_base"], d["apc_dataset"], {
            "$select": "route_id,direction,stop_id,stop_sequence,date_extract_dow(date) as dow,"
                       "date_extract_hh(hour::floating_timestamp) as hod,sum(boardings) as boardings,"
                       "sum(alightings) as alightings,sum(trips) as trips,count(*) as n_obs",
            "$where": f"route_id='{r}' and date between '{lo}T00:00:00' and '{hi}T23:59:59'",
            "$group": "route_id,direction,stop_id,stop_sequence,dow,hod",
        })
        print(f"apc {r}: {len(df)} rows")
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full["route_id"] = full["route_id"].astype(str)
    full.to_parquet(out, index=False)
    _record(raw, "apc_profile", {"dataset": d["apc_dataset"], "rows": len(full), "sha256": _sha256(out),
                                 "start": lo, "end": hi,
                                 "aggregation": "sum by route,direction,stop,dow,hour-of-day"})
    return out


def download_weather(cfg: dict, force: bool = False) -> Path:
    d = cfg["data"]
    raw = resolve(d["raw_dir"])
    out = raw / "weather_hourly.parquet"
    if out.exists() and not force:
        return out
    w = d["weather"]
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": w["latitude"], "longitude": w["longitude"], "start_date": d["start"],
              "end_date": d["end"], "timezone": w["timezone"],
              "hourly": "temperature_2m,precipitation,snowfall,wind_speed_10m"}
    js = _get(url, params).json()
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.to_parquet(out, index=False)
    _record(raw, "weather_hourly", {"url": url, "params": params, "rows": len(df), "sha256": _sha256(out)})
    return out


def main() -> None:
    cfg = load_config()
    for fn in (download_gtfs, download_weather, download_fare_ridership, download_apc_profiles):
        p = fn(cfg)
        print(f"{fn.__name__}: {p} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
