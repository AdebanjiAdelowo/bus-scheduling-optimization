"""Synthetic disruption scenarios (none of these are observed events).

A plan allocates n = ceil(x * tau / 60) vehicles to a route-hour. When running times grow or
vehicles are lost, the frequency that those vehicles can actually deliver is

    x_eff = min(x, 60 * n_available / (tau * run_factor)).
"""

from __future__ import annotations

import numpy as np

from busopt.opt.problem import Instance, buses_required


def effective_frequency(
    inst: Instance,
    freq: np.ndarray,
    run_factor: np.ndarray | None = None,
    vehicles_lost: np.ndarray | None = None,
    planned_cycle: np.ndarray | None = None,
) -> np.ndarray:
    """Deliverable frequency. ``planned_cycle`` is the cycle time the plan was sized with (slack plans)."""
    cyc_plan = inst.cycle_min if planned_cycle is None else planned_cycle
    n = np.where(inst.active, buses_required(freq, cyc_plan), 0.0)
    if vehicles_lost is not None:
        n = np.maximum(n - vehicles_lost, 0.0)
    rf = 1.0 if run_factor is None else run_factor
    return np.where(inst.active, np.minimum(freq, 60.0 * n / (inst.cycle_min * rf)), 0.0)


def congestion(inst: Instance, factor: float) -> np.ndarray:
    return np.full(inst.shape, factor)


def breakdowns(
    inst: Instance,
    freq: np.ndarray,
    prob: float,
    rng: np.random.Generator,
    planned_cycle: np.ndarray | None = None,
) -> np.ndarray:
    n = np.where(
        inst.active, buses_required(freq, inst.cycle_min if planned_cycle is None else planned_cycle), 0.0
    )
    return rng.binomial(n.astype(int), prob).astype(float)


def route_disruption(
    inst: Instance,
    rng: np.random.Generator,
    n_routes: int = 5,
    hours: tuple[int, int] = (8, 12),
    factor: float = 1.5,
) -> np.ndarray:
    rf = np.ones(inst.shape)
    routes = rng.choice(len(inst.routes), size=n_routes, replace=False)
    cols = [i for i, h in enumerate(inst.hours) if hours[0] <= h < hours[1]]
    rf[np.ix_(routes, cols)] = factor
    return rf


def demand_spike(
    inst: Instance,
    demand: np.ndarray,
    rng: np.random.Generator,
    n_routes: int = 6,
    hours: tuple[int, int] = (16, 20),
    factor: float = 1.6,
) -> np.ndarray:
    d = demand.copy()
    routes = rng.choice(len(inst.routes), size=n_routes, replace=False)
    cols = [i for i, h in enumerate(inst.hours) if hours[0] <= h < hours[1]]
    d[np.ix_(routes, cols)] *= factor
    return d
