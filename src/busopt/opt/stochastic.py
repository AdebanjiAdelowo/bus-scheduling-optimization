"""Demand scenarios and uncertainty-aware planning models.

Scenarios are built from *out-of-sample* forecast errors: the tree model trained on the training
window only is applied to the validation window and the ratio actual / forecast of every validation
day is kept as one [R, T] matrix. A scenario for a target day is its point forecast multiplied by
the ratio matrix of a randomly drawn validation day of the same day type. Resampling whole days
preserves the correlation of errors across routes and hours (weather, events).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from busopt.opt.milp import Solution, solve_frequency_milp
from busopt.opt.problem import Instance

DAYTYPE_CODE = {"weekday": 0, "saturday": 1, "sunday": 2}


def day_matrix(
    df: pd.DataFrame, col: str, routes: list[str], hours: list[int]
) -> dict[pd.Timestamp, np.ndarray]:
    """Pivot a long (ts, route_id, col) frame into one [R, T] matrix per date."""
    df = df[df.route_id.isin(routes) & df.hour.isin(hours)]
    out = {}
    for date, sub in df.groupby(df.ts.dt.normalize()):
        m = sub.pivot(index="route_id", columns="hour", values=col).reindex(index=routes, columns=hours)
        out[date] = m.to_numpy(float)
    return out


def residual_pool(
    valid_pred: pd.DataFrame, routes: list[str], hours: list[int], horizon: int = 1, floor: float = 20.0
) -> dict[str, np.ndarray]:
    """Ratio matrices actual/forecast by day type. Cells with tiny forecasts are damped via ``floor``."""
    v = valid_pred[valid_pred.horizon == horizon]
    act, fc = day_matrix(v, "y", routes, hours), day_matrix(v, "gbt", routes, hours)
    pool: dict[str, list[np.ndarray]] = {k: [] for k in DAYTYPE_CODE}
    for date, a in act.items():
        if np.isnan(a).all():
            continue
        ratio = (np.nan_to_num(a, nan=0.0) + floor) / (np.nan_to_num(fc[date], nan=0.0) + floor)
        ratio = np.where(np.isnan(a), 1.0, np.clip(ratio, 0.2, 5.0))
        dt = "weekday" if date.dayofweek < 5 else ("saturday" if date.dayofweek == 5 else "sunday")
        pool[dt].append(ratio)
    return {k: np.stack(v) for k, v in pool.items() if v}


def make_scenarios(forecast: np.ndarray, pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(pool), size=n, replace=len(pool) < n)
    return forecast[None] * pool[idx]


def solve_saa(inst: Instance, scenarios: np.ndarray, **kw) -> Solution:
    """Two-stage stochastic program by sample-average approximation (expected cost)."""
    sol = solve_frequency_milp(inst, scenarios=scenarios, **kw)
    sol.method = "milp_saa"
    return sol


def solve_cvar(
    inst: Instance, scenarios: np.ndarray, cvar_weight: float = 1.0, cvar_alpha: float = 0.9, **kw
) -> Solution:
    """SAA plus a CVaR penalty on total passengers above the loading standard."""
    sol = solve_frequency_milp(
        inst, scenarios=scenarios, cvar_weight=cvar_weight, cvar_alpha=cvar_alpha, **kw
    )
    sol.method = "milp_saa_cvar"
    return sol


def solve_quantile_robust(inst: Instance, scenarios: np.ndarray, q: float = 0.9, **kw) -> Solution:
    """Deterministic model fed the cell-wise q-quantile of demand (conservative point estimate)."""
    sol = solve_frequency_milp(inst.with_demand(np.quantile(scenarios, q, axis=0)), **kw)
    sol.method = f"milp_quantile_{int(q * 100)}"
    return sol


def solve_slack(inst: Instance, slack: float = 0.10, **kw) -> Solution:
    """Running-time-robust plan: fleet requirements computed with cycle times inflated by ``slack``."""
    sol = solve_frequency_milp(replace(inst, cycle_min=inst.cycle_min * (1.0 + slack)), **kw)
    sol.method = f"milp_slack_{int(slack * 100)}"
    return sol
