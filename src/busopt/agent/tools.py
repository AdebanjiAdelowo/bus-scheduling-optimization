"""Typed, read-only planning tools.

Each tool has a Pydantic argument model (the single source of validation for both the deterministic
router and the LLM agent) and a function that runs real forecasting / optimisation / simulation
code against a :class:`PlanningSession`. Tools only compute what-if analyses in memory. No tool
writes a timetable, dispatches a vehicle or touches any external system.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from busopt.opt import heuristics as H
from busopt.opt.milp import Solution, binding_constraints, solve_frequency_milp, verify
from busopt.opt.problem import Instance, build_instance, evaluate
from busopt.opt.stochastic import solve_quantile_robust, solve_saa
from busopt.sim.simulator import build_sim_network, draw_passengers, simulate

DayType = Literal["weekday", "saturday", "sunday"]
PlanName = Literal["existing", "optimized"]


class ToolError(Exception):
    """Raised for invalid arguments or infeasible requests; reported back verbatim, never guessed around."""


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NetworkSummaryArgs(_Args):
    daytype: DayType = "weekday"


class CrowdingArgs(_Args):
    daytype: DayType = "weekday"
    plan: PlanName = "existing"
    top_k: int = Field(5, ge=1, le=50)
    demand_scale: float = Field(1.0, gt=0, le=5)


class OptimizeArgs(_Args):
    daytype: DayType = "weekday"
    fleet_change_pct: float = Field(
        0.0, ge=-60, le=100, description="Change in fleet and vehicle-hour budget, percent"
    )
    demand_scale: float = Field(1.0, gt=0, le=5)


class FleetChangeArgs(_Args):
    daytype: DayType = "weekday"
    fleet_change_pct: float = Field(..., ge=-60, le=100, description="Negative reduces the fleet, e.g. -10")


class FrequencyEffectArgs(_Args):
    daytype: DayType = "weekday"
    frequency_change_pct: float = Field(..., ge=-80, le=300)
    route_id: str | None = Field(None, description="Route to change; omit for all routes")


class RobustnessArgs(_Args):
    daytype: DayType = "weekday"
    demand_scale: float = Field(1.0, gt=0, le=5)
    quantile: float = Field(0.9, gt=0.5, lt=1.0)
    n_scenarios: int = Field(30, ge=5, le=200)


class BindingArgs(_Args):
    daytype: DayType = "weekday"
    plan: PlanName = "optimized"


class SimulateArgs(_Args):
    daytype: DayType = "weekday"
    plan: PlanName = "existing"
    travel_time_cv: float = Field(0.10, ge=0, le=0.5)
    demand_scale: float = Field(1.0, gt=0, le=5)


class DemandForecastArgs(_Args):
    daytype: DayType = "weekday"
    route_id: str | None = None
    top_k: int = Field(5, ge=1, le=50)


@dataclass
class PlanningSession:
    """Everything the tools need. ``demand_source`` states what the demand numbers are."""

    tables: dict
    cfg: dict
    demand_source: str
    residual_pool: dict[str, np.ndarray] | None = None
    seed: int = 0
    _inst: dict[str, Instance] = field(default_factory=dict)
    _opt: dict[tuple, Solution] = field(default_factory=dict)

    def instance(self, daytype: str) -> Instance:
        if daytype not in self._inst:
            self._inst[daytype] = build_instance(self.tables, self.cfg, daytype)
        return self._inst[daytype]

    def solver_kw(self) -> dict:
        o = self.cfg["optimization"]
        return {"solver": o["solver"], "time_limit_s": o["time_limit_s"], "mip_gap": o["mip_gap"]}

    def optimized(
        self, daytype: str, fleet_scale: float = 1.0, demand_scale: float = 1.0
    ) -> tuple[Instance, Solution]:
        inst = self.instance(daytype).scaled(demand_scale=demand_scale, fleet_scale=fleet_scale)
        key = (daytype, round(fleet_scale, 6), round(demand_scale, 6))
        if key not in self._opt:
            self._opt[key] = solve_frequency_milp(inst, **self.solver_kw())
        return inst, self._opt[key]

    def plan(self, daytype: str, plan: str, demand_scale: float = 1.0) -> tuple[Instance, np.ndarray, dict]:
        if plan == "existing":
            inst = self.instance(daytype).scaled(demand_scale=demand_scale)
            return inst, inst.gtfs_freq, {"plan": "existing", "status": "GIVEN"}
        inst, sol = self.optimized(daytype, 1.0, demand_scale)
        if sol.objective is None:
            raise ToolError(f"optimisation returned {sol.status}; no plan available")
        return inst, sol.freq, {"plan": "optimized", **_solver_info(inst, sol)}

    def scenarios(self, inst: Instance, n: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        if self.residual_pool and inst.daytype in self.residual_pool:
            pool = self.residual_pool[inst.daytype]
            return inst.demand[None] * pool[rng.choice(len(pool), n, replace=len(pool) < n)]
        return inst.demand[None] * rng.lognormal(
            -0.02, 0.2, (n, *inst.shape)
        )  # labelled synthetic in the output


def _r(x: float, nd: int = 3) -> float:
    return float(round(float(x), nd))


def _solver_info(inst: Instance, sol: Solution) -> dict:
    v = verify(inst, sol)
    return {
        "status": sol.status,
        "solver": sol.solver,
        "solver_version": sol.solver_version,
        "mip_gap": None if sol.gap is None else _r(sol.gap, 6),
        "runtime_s": _r(sol.runtime_s),
        "independently_verified": bool(v.get("verified", False)),
    }


def _plan_metrics(inst: Instance, freq: np.ndarray) -> dict:
    ev = evaluate(inst, freq)
    return {
        "mean_wait_min": _r(ev["mean_wait_min"]),
        "vehicle_hours": _r(ev["vehicle_hours"], 1),
        "peak_fleet": int(ev["peak_fleet"]),
        "passengers_over_standard": _r(ev["overload_pax"], 1),
        "max_load_factor": _r(ev["max_load_factor"]),
        "feasible": ev["feasible"],
    }


def network_summary(s: PlanningSession, a: NetworkSummaryArgs) -> dict:
    inst = s.instance(a.daytype)
    return {
        "daytype": a.daytype,
        "routes": len(inst.routes),
        "route_ids": inst.routes,
        "service_hours": [inst.hours[0], inst.hours[-1] + 1],
        "fleet": inst.fleet,
        "vehicle_hour_budget": _r(inst.budget_vh, 1),
        "daily_boardings": _r(inst.demand.sum(), 0),
        "existing_schedule": _plan_metrics(inst, inst.gtfs_freq),
        "demand_source": s.demand_source,
    }


def crowding_ranking(s: PlanningSession, a: CrowdingArgs) -> dict:
    inst, freq, info = s.plan(a.daytype, a.plan, a.demand_scale)
    cap = np.where(freq > 0, freq * inst.capacity[:, None], np.nan)
    lf = inst.peak_coef * inst.demand / cap
    peak = np.nanmax(lf, axis=1)
    hour = np.nanargmax(lf, axis=1)
    order = np.argsort(-peak)[: a.top_k]
    return {
        "daytype": a.daytype,
        "demand_scale": a.demand_scale,
        **info,
        "definition": "peak-hour peak-direction load divided by the loading standard times buses per hour",
        "ranking": [
            {
                "route_id": inst.routes[i],
                "peak_load_factor": _r(peak[i]),
                "hour": int(inst.hours[hour[i]]),
                "buses_per_hour": int(freq[i, hour[i]]),
            }
            for i in order
        ],
        "routes_over_standard": int((peak > 1).sum()),
        "demand_source": s.demand_source,
    }


def optimize_schedule(s: PlanningSession, a: OptimizeArgs) -> dict:
    scale = 1.0 + a.fleet_change_pct / 100.0
    inst, sol = s.optimized(a.daytype, scale, a.demand_scale)
    out = {
        "daytype": a.daytype,
        "fleet_change_pct": a.fleet_change_pct,
        "demand_scale": a.demand_scale,
        "fleet_limit": inst.fleet,
        "vehicle_hour_budget": _r(inst.budget_vh, 1),
        **_solver_info(inst, sol),
        "demand_source": s.demand_source,
    }
    if sol.objective is None:
        return {
            **out,
            "feasible": False,
            "message": "No plan satisfies the minimum-frequency policy with these resources.",
        }
    base = s.instance(a.daytype).scaled(demand_scale=a.demand_scale)
    return {
        **out,
        "optimized": _plan_metrics(inst, sol.freq),
        "existing_schedule_same_demand": _plan_metrics(base, base.gtfs_freq),
    }


def compare_fleet_change(s: PlanningSession, a: FleetChangeArgs) -> dict:
    base_inst, base = s.optimized(a.daytype, 1.0)
    new_inst, new = s.optimized(a.daytype, 1.0 + a.fleet_change_pct / 100.0)
    out = {
        "daytype": a.daytype,
        "fleet_change_pct": a.fleet_change_pct,
        "baseline_fleet": base_inst.fleet,
        "new_fleet": new_inst.fleet,
        "baseline": {**_plan_metrics(base_inst, base.freq), **_solver_info(base_inst, base)},
        "note": "fleet and vehicle-hour budget are scaled together; both plans are re-optimised",
        "demand_source": s.demand_source,
    }
    if new.objective is None:
        return {
            **out,
            "new": {"feasible": False, **_solver_info(new_inst, new)},
            "message": "Infeasible: the minimum-frequency policy cannot be met with the reduced fleet.",
        }
    nm, bm = _plan_metrics(new_inst, new.freq), _plan_metrics(base_inst, base.freq)
    return {
        **out,
        "new": {**nm, **_solver_info(new_inst, new)},
        "change": {
            "mean_wait_min": _r(nm["mean_wait_min"] - bm["mean_wait_min"]),
            "mean_wait_pct": _r(100 * (nm["mean_wait_min"] / bm["mean_wait_min"] - 1), 2),
            "vehicle_hours": _r(nm["vehicle_hours"] - bm["vehicle_hours"], 1),
        },
    }


def frequency_effect(s: PlanningSession, a: FrequencyEffectArgs) -> dict:
    inst = s.instance(a.daytype)
    if a.route_id is not None and a.route_id not in inst.routes:
        raise ToolError(f"unknown route_id {a.route_id!r}; valid routes: {', '.join(inst.routes)}")
    rows = slice(None) if a.route_id is None else [inst.routes.index(a.route_id)]
    freq = inst.gtfs_freq.astype(float).copy()
    freq[rows] = np.where(
        inst.active[rows], np.clip(np.round(freq[rows] * (1 + a.frequency_change_pct / 100.0)), 1, None), 0
    )
    before, after = evaluate(inst, inst.gtfs_freq), evaluate(inst, freq)
    dem = inst.demand[rows].sum()
    wait_b = (inst.demand[rows] * 30.0 / np.maximum(inst.gtfs_freq[rows], 1)).sum() / dem
    wait_a = (inst.demand[rows] * 30.0 / np.maximum(freq[rows], 1)).sum() / dem
    return {
        "daytype": a.daytype,
        "route_id": a.route_id or "ALL",
        "frequency_change_pct": a.frequency_change_pct,
        "starting_point": "existing schedule",
        "affected_mean_wait_min": {"before": _r(wait_b), "after": _r(wait_a), "change": _r(wait_a - wait_b)},
        "network_mean_wait_min": {"before": _r(before["mean_wait_min"]), "after": _r(after["mean_wait_min"])},
        "vehicle_hours": {
            "before": _r(before["vehicle_hours"], 1),
            "after": _r(after["vehicle_hours"], 1),
            "change": _r(after["vehicle_hours"] - before["vehicle_hours"], 1),
        },
        "peak_fleet": {
            "before": int(before["peak_fleet"]),
            "after": int(after["peak_fleet"]),
            "available": inst.fleet,
        },
        "fits_current_fleet": bool(after["violations"]["fleet"] == 0),
        "operating_cost_change_usd_per_day": _r(
            (after["vehicle_hours"] - before["vehicle_hours"])
            * s.cfg["optimization"]["cost_per_vehicle_hour"],
            0,
        ),
        "cost_rate_note": "cost per vehicle-hour is an assumed parameter",
        "demand_source": s.demand_source,
    }


def robustness_cost(s: PlanningSession, a: RobustnessArgs) -> dict:
    inst = s.instance(a.daytype).scaled(demand_scale=a.demand_scale)
    scen = s.scenarios(inst, a.n_scenarios)
    det = solve_frequency_milp(inst, **s.solver_kw())
    rob = solve_quantile_robust(inst, scen, a.quantile, **s.solver_kw())
    saa = solve_saa(inst, scen, **s.solver_kw())
    if det.objective is None or rob.objective is None or saa.objective is None:
        raise ToolError("optimisation infeasible for these settings")

    def stats(sol: Solution) -> dict:
        ev = [evaluate(inst, sol.freq, d) for d in scen]
        nominal = evaluate(inst, sol.freq)
        return {
            "nominal_mean_wait_min": _r(nominal["mean_wait_min"]),
            "vehicle_hours": _r(nominal["vehicle_hours"], 1),
            "expected_passengers_over_standard": _r(np.mean([e["overload_pax"] for e in ev]), 1),
            "worst_scenario_passengers_over_standard": _r(np.max([e["overload_pax"] for e in ev]), 1),
            "expected_mean_wait_min": _r(np.mean([e["mean_wait_min"] for e in ev])),
            "status": sol.status,
        }

    d, r, e = stats(det), stats(rob), stats(saa)
    return {
        "daytype": a.daytype,
        "demand_scale": a.demand_scale,
        "n_scenarios": a.n_scenarios,
        "quantile": a.quantile,
        "deterministic": d,
        "quantile_robust": r,
        "stochastic_saa": e,
        "cost_of_robustness_wait_min": _r(r["nominal_mean_wait_min"] - d["nominal_mean_wait_min"], 4),
        "benefit_expected_passengers_over_standard": _r(
            d["expected_passengers_over_standard"] - r["expected_passengers_over_standard"], 1
        ),
        "scenario_source": "out-of-sample forecast residuals"
        if s.residual_pool
        else "synthetic lognormal (sigma 0.2)",
        "note": "all three plans use the same fleet and budget, so robustness costs waiting time, not vehicle-hours",
        "demand_source": s.demand_source,
    }


def binding(s: PlanningSession, a: BindingArgs) -> dict:
    inst, freq, info = s.plan(a.daytype, a.plan)
    b = binding_constraints(inst, freq)
    lp = solve_frequency_milp(inst, relax=True)
    return {
        "daytype": a.daytype,
        **info,
        "budget": {k: (_r(v, 1) if isinstance(v, float) else v) for k, v in b["budget"].items()},
        "fleet_binding_hours": [f["hour"] for f in b["fleet"] if f["binding"]],
        "fleet_slack_by_hour": {int(f["hour"]): _r(f["slack"], 1) for f in b["fleet"]},
        "cells_at_min_frequency": b["min_frequency_cells"],
        "cells_at_max_frequency": b["max_frequency_cells"],
        "cells_over_loading_standard": b["capacity_cells_over_standard"],
        "active_cells": b["active_cells"],
        "lp_shadow_prices": {
            "budget_pax_min_per_vehicle_hour": _r(-lp.info["budget_dual"], 2),
            "fleet_pax_min_per_bus_by_hour": {
                int(h): _r(-d, 2)
                for h, d in zip(inst.hours, lp.info["fleet_dual"], strict=True)
                if abs(d) > 1e-9
            },
        },
        "shadow_price_note": "from the LP relaxation; the integer model has no exact duals",
        "demand_source": s.demand_source,
    }


def simulate_plan(s: PlanningSession, a: SimulateArgs) -> dict:
    inst, freq, info = s.plan(a.daytype, a.plan, a.demand_scale)
    net = build_sim_network(s.tables, inst)
    draw = draw_passengers(net, inst.demand, np.random.default_rng(s.seed))
    sim = simulate(net, freq, draw, seed=s.seed, travel_time_cv=a.travel_time_cv)
    keys = [
        "passengers",
        "mean_wait_min",
        "p90_wait_min",
        "p95_wait_min",
        "unmet_share",
        "denied_per_1000_pax",
        "overcrowded_pax_min",
        "load_utilization",
        "service_coverage",
        "trips",
    ]
    return {
        "daytype": a.daytype,
        "travel_time_cv": a.travel_time_cv,
        "demand_scale": a.demand_scale,
        **info,
        "simulation": {k: (_r(sim[k]) if isinstance(sim[k], float) else sim[k]) for k in keys},
        "model_mean_wait_min": _r(evaluate(inst, freq)["mean_wait_min"]),
        "note": "passenger arrival times, running-time noise and dwell are simulated",
        "demand_source": s.demand_source,
    }


def demand_forecast(s: PlanningSession, a: DemandForecastArgs) -> dict:
    inst = s.instance(a.daytype)
    if a.route_id is not None:
        if a.route_id not in inst.routes:
            raise ToolError(f"unknown route_id {a.route_id!r}; valid routes: {', '.join(inst.routes)}")
        i = inst.routes.index(a.route_id)
        return {
            "daytype": a.daytype,
            "route_id": a.route_id,
            "expected_boardings_by_hour": {
                int(h): _r(v, 1) for h, v in zip(inst.hours, inst.demand[i], strict=True)
            },
            "daily_total": _r(inst.demand[i].sum(), 0),
            "peak_hour": int(inst.hours[int(inst.demand[i].argmax())]),
            "demand_source": s.demand_source,
        }
    tot = inst.demand.sum(axis=1)
    order = np.argsort(-tot)[: a.top_k]
    return {
        "daytype": a.daytype,
        "network_daily_total": _r(tot.sum(), 0),
        "network_peak_hour": int(inst.hours[int(inst.demand.sum(axis=0).argmax())]),
        "top_routes": [{"route_id": inst.routes[i], "daily_boardings": _r(tot[i], 0)} for i in order],
        "demand_source": s.demand_source,
    }


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: type[_Args]
    fn: Callable[[PlanningSession, Any], dict]


TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in [
        ToolSpec(
            "network_summary",
            "Size of the network and planning instance, fleet, budget and metrics of the existing schedule.",
            NetworkSummaryArgs,
            network_summary,
        ),
        ToolSpec(
            "demand_forecast",
            "Expected boardings by route (ranking) or by hour for one route.",
            DemandForecastArgs,
            demand_forecast,
        ),
        ToolSpec(
            "crowding_ranking",
            "Rank routes by expected crowding (peak load factor against the loading standard).",
            CrowdingArgs,
            crowding_ranking,
        ),
        ToolSpec(
            "optimize_schedule",
            "Solve the frequency-setting MILP and compare it with the existing schedule. Optional fleet/budget change and demand scaling.",
            OptimizeArgs,
            optimize_schedule,
        ),
        ToolSpec(
            "compare_fleet_change",
            "What happens to waiting time if the number of available buses changes by a percentage (both plans re-optimised).",
            FleetChangeArgs,
            compare_fleet_change,
        ),
        ToolSpec(
            "frequency_effect",
            "Effect on expected waiting time, vehicle-hours and cost of changing frequency on one route or all routes, starting from the existing schedule.",
            FrequencyEffectArgs,
            frequency_effect,
        ),
        ToolSpec(
            "robustness_cost",
            "Cost and benefit of a demand-robust schedule versus the deterministic optimum under demand scenarios.",
            RobustnessArgs,
            robustness_cost,
        ),
        ToolSpec(
            "binding_constraints",
            "Which constraints are binding in a plan (fleet by hour, budget, frequency bounds, loading standard) with LP shadow prices.",
            BindingArgs,
            binding,
        ),
        ToolSpec(
            "simulate_plan",
            "Passenger-level simulation of a plan: waiting-time percentiles, unmet demand and overcrowding.",
            SimulateArgs,
            simulate_plan,
        ),
    ]
}


def call_tool(session: PlanningSession, name: str, raw_args: dict | None) -> dict:
    """Validate and execute. Returns {"ok": bool, "tool", "args", "result" | "error"}; never raises."""
    if name not in TOOLS:
        return {
            "ok": False,
            "tool": name,
            "args": raw_args,
            "error": f"unknown tool {name!r}",
            "error_type": "unknown_tool",
        }
    spec = TOOLS[name]
    try:
        args = spec.args.model_validate(raw_args or {})
    except ValidationError as e:
        return {
            "ok": False,
            "tool": name,
            "args": raw_args,
            "error_type": "invalid_arguments",
            "error": "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()),
        }
    try:
        return {"ok": True, "tool": name, "args": args.model_dump(), "result": spec.fn(session, args)}
    except ToolError as e:
        return {
            "ok": False,
            "tool": name,
            "args": args.model_dump(),
            "error": str(e),
            "error_type": "tool_error",
        }


def anthropic_tool_definitions() -> list[dict]:
    """JSON-schema tool definitions for the Messages API, generated from the Pydantic models."""
    defs = []
    for spec in TOOLS.values():
        schema = spec.args.model_json_schema()
        schema.pop("title", None)
        defs.append(
            {
                "name": spec.name,
                "description": spec.description + " Read-only what-if analysis.",
                "input_schema": schema,
            }
        )
    return defs


def demo_session(cfg: dict | None = None, prefer_real: bool = True) -> PlanningSession:
    """Session on processed MTA data when present, otherwise on the synthetic toy network."""
    from busopt.config import load_config, resolve
    from busopt.synthetic import toy_tables

    cfg = cfg or load_config()
    proc = resolve(cfg["data"]["processed_dir"])
    if prefer_real and (proc / "route_hour.parquet").exists():
        import pandas as pd

        from busopt.data.build import load_tables
        from busopt.opt.stochastic import residual_pool

        tables = load_tables(cfg)
        pool = None
        vp = proc / "forecast_valid_predictions.parquet"
        if vp.exists():
            inst = build_instance(tables, cfg, "weekday")
            pool = residual_pool(pd.read_parquet(vp), inst.routes, inst.hours)
        return PlanningSession(
            tables,
            cfg,
            "observed MTA fare-payment ridership, training-window mean by day type (excludes unpaid boardings)",
            pool,
        )
    return PlanningSession(toy_tables(), cfg, "SYNTHETIC toy network (not observed data)")


__all__ = [
    "TOOLS",
    "PlanningSession",
    "ToolError",
    "call_tool",
    "anthropic_tool_definitions",
    "demo_session",
    "H",
]
