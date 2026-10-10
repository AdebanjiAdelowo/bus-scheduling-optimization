"""The frequency-setting / fleet-allocation planning instance.

Sets: routes r in R, hourly periods t in T.  Data per (r, t): demand d (boardings per hour, both
directions), peak-load coefficient c (peak-direction passengers through the maximum load point per
boarding), round-trip cycle time tau (min), and an ``active`` mask (the route is operated in t).
Decision: x[r, t] = buses per hour per direction, integer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd


@dataclass
class Instance:
    routes: list[str]
    hours: list[int]
    daytype: str
    demand: np.ndarray  # [R, T] boardings / hour
    peak_coef: np.ndarray  # [R, T]
    cycle_min: np.ndarray  # [R, T]
    capacity: np.ndarray  # [R] guideline passengers per bus
    active: np.ndarray  # [R, T] bool
    gtfs_freq: np.ndarray  # [R, T] existing schedule projected on the decision space
    fleet: int
    budget_vh: float
    freq_min: int = 2
    freq_max: int = 30
    unmet_penalty_min: float = 30.0
    headway_cv: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.demand.shape

    def with_demand(self, demand: np.ndarray) -> Instance:
        return replace(self, demand=np.where(self.active, demand, 0.0))

    def scaled(
        self, demand_scale: float = 1.0, fleet_scale: float = 1.0, budget_scale: float | None = None
    ) -> Instance:
        b = fleet_scale if budget_scale is None else budget_scale
        return replace(
            self,
            demand=self.demand * demand_scale,
            fleet=int(np.floor(self.fleet * fleet_scale + 1e-9)),
            budget_vh=self.budget_vh * b,
        )


def buses_required(freq: np.ndarray, cycle_min: np.ndarray) -> np.ndarray:
    """Vehicles needed to hold ``freq`` buses/hour on a route with round-trip time ``cycle_min``."""
    return np.ceil(freq * cycle_min / 60.0 - 1e-9)


def wait_minutes(freq: np.ndarray, headway_cv: float = 0.0) -> np.ndarray:
    """Mean wait of randomly arriving passengers: E[W] = (h/2)(1 + cv^2), h = 60/x."""
    with np.errstate(divide="ignore"):
        return np.where(freq > 0, 30.0 / np.maximum(freq, 1e-12) * (1.0 + headway_cv**2), np.inf)


def evaluate(inst: Instance, freq: np.ndarray, demand: np.ndarray | None = None) -> dict:
    """Closed-form planning metrics of a frequency plan, optionally against a different demand.

    This function is deliberately independent of every solver: it is the feasibility and objective
    referee used to verify solver output.
    """
    d = inst.demand if demand is None else np.where(inst.active, demand, 0.0)
    x = np.where(inst.active, freq, 0)
    n = np.where(inst.active, buses_required(x, inst.cycle_min), 0.0)
    w = np.where(inst.active & (x > 0), wait_minutes(np.maximum(x, 1), inst.headway_cv), 0.0)
    wait_pm = float((d * w).sum())
    cap = x * inst.capacity[:, None]
    over = np.clip(inst.peak_coef * d - cap, 0.0, None)
    fleet_by_t = n.sum(axis=0)
    viol = {
        "fleet": float(max(0.0, fleet_by_t.max() - inst.fleet)),
        "budget": float(max(0.0, n.sum() - inst.budget_vh - 1e-6)),
        "freq_min": int(((x < inst.freq_min) & inst.active).sum()),
        "freq_max": int((x > inst.freq_max).sum()),
        "integrality": float(np.abs(x - np.round(x)).max()),
    }
    tot = float(d.sum())
    return {
        "wait_pax_min": wait_pm,
        "mean_wait_min": wait_pm / tot if tot > 0 else float("nan"),
        "overload_pax": float(over.sum()),
        "overload_share": float(over.sum() / max((inst.peak_coef * d).sum(), 1e-9)),
        "objective": wait_pm + inst.unmet_penalty_min * float(over.sum()),
        "vehicle_hours": float(n.sum()),
        "peak_fleet": float(fleet_by_t.max()),
        "fleet_by_hour": fleet_by_t.tolist(),
        "max_load_factor": float(np.where(cap > 0, inst.peak_coef * d / np.where(cap > 0, cap, 1), 0).max()),
        "mean_load_factor": float((inst.peak_coef * d).sum() / max(cap.sum(), 1e-9)),
        "violations": viol,
        "feasible": bool(
            viol["fleet"] <= 1e-6
            and viol["budget"] <= 1e-6
            and viol["freq_min"] == 0
            and viol["freq_max"] == 0
            and viol["integrality"] <= 1e-6
        ),
    }


def build_instance(
    tables: dict[str, pd.DataFrame],
    cfg: dict,
    daytype: str,
    demand: np.ndarray | None = None,
    routes: list[str] | None = None,
) -> Instance:
    """Assemble an instance for one day type from processed tables.

    ``demand`` defaults to the training-period mean for the day type. Fleet and budget default to what
    the projected existing schedule uses, so every policy is compared at equal resources.
    """
    o, n = cfg["optimization"], cfg["network"]
    hours = list(range(n["service_hours"][0], n["service_hours"][1]))
    rh = tables["route_hour"]
    rh = rh[(rh.daytype == daytype) & rh.hour.isin(hours)]
    routes = routes or sorted(tables["route_info"].query("in_instance").route_id)
    idx = pd.MultiIndex.from_product([routes, hours], names=["route_id", "hour"])
    rh = rh.set_index(["route_id", "hour"]).reindex(idx)
    shp = (len(routes), len(hours))
    gtfs_freq = rh.gtfs_freq.fillna(0).to_numpy().reshape(shp)
    active = gtfs_freq > 0
    info = tables["route_info"].set_index("route_id").loc[routes]
    cap = info.capacity_guideline.to_numpy(float)
    freq_min = int(np.ceil(60.0 / o["max_headway_min"]))
    gtfs_freq = np.where(active, np.clip(gtfs_freq, freq_min, o["freq_max"]), 0).astype(int)
    cycle = rh.cycle_min.to_numpy().reshape(shp)
    cycle = np.where(np.isfinite(cycle), cycle, np.nanmedian(cycle, axis=1, keepdims=True))
    if demand is None:
        demand = rh.mean_demand.fillna(0).to_numpy().reshape(shp)
    need = np.where(active, buses_required(gtfs_freq, cycle), 0.0)
    return Instance(
        routes=routes,
        hours=hours,
        daytype=daytype,
        demand=np.where(active, demand, 0.0),
        peak_coef=rh.peak_coef.fillna(0.5).to_numpy().reshape(shp),
        cycle_min=cycle,
        capacity=cap,
        active=active,
        gtfs_freq=gtfs_freq,
        fleet=int(need.sum(axis=0).max()),
        budget_vh=float(need.sum()),
        freq_min=freq_min,
        freq_max=int(o["freq_max"]),
        unmet_penalty_min=float(o["unmet_penalty_min"]),
        headway_cv=float(o["headway_cv"]),
        meta={"daytype": daytype},
    )
