"""Shared experiment plumbing: data access, per-day demand matrices and summary statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from busopt.config import load_config, resolve
from busopt.data.build import load_tables
from busopt.opt.problem import Instance, build_instance
from busopt.opt.stochastic import day_matrix, residual_pool
from busopt.sim.simulator import SimNetwork, build_sim_network

SIM_KEYS = [
    "mean_wait_min",
    "p50_wait_min",
    "p90_wait_min",
    "p95_wait_min",
    "share_wait_le_10min",
    "unmet_capacity",
    "unmet_end_of_service",
    "unmet_share",
    "denied_per_1000_pax",
    "overcrowded_pax_min",
    "share_pax_time_overcrowded",
    "load_utilization",
    "max_load_factor",
    "trips",
    "service_coverage",
]


def daytype_of_date(d: pd.Timestamp) -> str:
    return "weekday" if d.dayofweek < 5 else ("saturday" if d.dayofweek == 5 else "sunday")


@dataclass
class Context:
    cfg: dict
    tables: dict
    inst: dict[str, Instance]  # per day type, demand = training mean
    net: dict[str, SimNetwork]
    actual: dict[pd.Timestamp, np.ndarray]
    forecast: dict[int, dict[pd.Timestamp, np.ndarray]]  # horizon -> date -> [R,T]
    pool: dict[str, np.ndarray]
    test_days: list[pd.Timestamp]


def load_context(cfg: dict | None = None) -> Context:
    cfg = cfg or load_config()
    tables = load_tables(cfg)
    inst = {dt: build_instance(tables, cfg, dt) for dt in ("weekday", "saturday", "sunday")}
    net = {dt: build_sim_network(tables, i) for dt, i in inst.items()}
    routes, hours = inst["weekday"].routes, inst["weekday"].hours
    proc = resolve(cfg["data"]["processed_dir"])
    pred = pd.read_parquet(proc / "forecast_test_predictions.parquet")
    valid = pd.read_parquet(proc / "forecast_valid_predictions.parquet")
    actual = day_matrix(pred[pred.horizon == 1], "y", routes, hours)
    forecast = {
        int(h): day_matrix(pred[pred.horizon == h], "gbt", routes, hours)
        for h in sorted(pred.horizon.unique())
    }
    pool = residual_pool(valid, routes, hours, horizon=1)
    days = [d for d, a in sorted(actual.items()) if not np.isnan(a).any()]
    return Context(cfg, tables, inst, net, actual, forecast, pool, days)


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap 95% interval over days."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_summary(df: pd.DataFrame, metric: str, baseline: str, key: str = "date") -> pd.DataFrame:
    """Per-policy mean of ``metric`` and the paired difference to ``baseline`` with a bootstrap CI."""
    piv = df.pivot_table(index=key, columns="policy", values=metric)
    rows = []
    for pol in piv.columns:
        diff = (piv[pol] - piv[baseline]).dropna().to_numpy()
        m, lo, hi = (
            bootstrap_ci(diff)
            if len(diff) > 1
            else (float(diff.mean()) if len(diff) else np.nan, np.nan, np.nan)
        )
        base = piv[baseline].mean()
        rows.append(
            {
                "policy": pol,
                "metric": metric,
                "mean": piv[pol].mean(),
                f"diff_vs_{baseline}": m,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "pct_vs_baseline": 100 * m / base if base else np.nan,
                "n_days": len(diff),
            }
        )
    return pd.DataFrame(rows)
