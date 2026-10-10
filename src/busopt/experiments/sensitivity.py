"""E3-E5: cost/service trade-off, fleet-size sensitivity and demand scaling (RQ3, RQ5)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from busopt.config import resolve
from busopt.experiments.common import SIM_KEYS, Context, load_context
from busopt.opt import heuristics as H
from busopt.opt.milp import solve_frequency_milp, verify
from busopt.opt.problem import evaluate
from busopt.opt.stochastic import make_scenarios, solve_cvar, solve_quantile_robust, solve_saa
from busopt.sim.simulator import draw_passengers, simulate
from busopt.tracking import Run


def _kw(cfg: dict) -> dict:
    o = cfg["optimization"]
    return {"solver": o["solver"], "time_limit_s": o["time_limit_s"], "mip_gap": o["mip_gap"]}


def _row(inst, sol, label: dict, cost_per_vh: float) -> dict:
    base = {
        **label,
        "policy": sol.method,
        "status": sol.status,
        "gap": sol.gap,
        "runtime_s": sol.runtime_s,
        "fleet_limit": inst.fleet,
        "budget_vh": inst.budget_vh,
    }
    if sol.objective is None:
        return {**base, "feasible": False}
    ev = evaluate(inst, sol.freq)
    return {
        **base,
        "feasible": ev["feasible"],
        "mean_wait_min": ev["mean_wait_min"],
        "wait_pax_min": ev["wait_pax_min"],
        "overload_pax": ev["overload_pax"],
        "vehicle_hours": ev["vehicle_hours"],
        "peak_fleet": ev["peak_fleet"],
        "operating_cost_usd": ev["vehicle_hours"] * cost_per_vh,
        "fleet_utilization": ev["vehicle_hours"] / max(inst.fleet * len(inst.hours), 1),
        "max_load_factor": ev["max_load_factor"],
    }


def resource_frontier(ctx: Context) -> pd.DataFrame:
    """Budget sweep (epsilon-constraint), fleet-only sweep and weighted-sum sweep on the weekday instance."""
    kw, cpv = _kw(ctx.cfg), ctx.cfg["optimization"]["cost_per_vehicle_hour"]
    base = ctx.inst["weekday"]
    rows = [_row(base, H.existing_schedule(base), {"experiment": "reference", "scale": 1.0}, cpv)]
    for sc in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4):
        inst = base.scaled(fleet_scale=sc)  # fleet and budget together
        for sol in (
            solve_frequency_milp(inst, **kw),
            H.sqrt_rule(inst),
            H.greedy(inst),
            H.proportional(inst),
        ):
            rows.append(_row(inst, sol, {"experiment": "budget_and_fleet", "scale": sc}, cpv))
        inst = base.scaled(fleet_scale=sc, budget_scale=1.0)  # peak fleet only
        for sol in (solve_frequency_milp(inst, **kw), H.sqrt_rule(inst), H.greedy(inst)):
            rows.append(_row(inst, sol, {"experiment": "fleet_only", "scale": sc}, cpv))
    loose = base.scaled(fleet_scale=1.5)
    for lam in (0, 25, 50, 75, 100, 150, 200, 300, 500, 1000):
        sol = solve_frequency_milp(loose, cost_weight=float(lam), **kw)
        r = _row(loose, sol, {"experiment": "weighted_sum", "scale": lam}, cpv)
        r["verified"] = verify(loose, sol, cost_weight=float(lam))["verified"]
        rows.append(r)
    return pd.DataFrame(rows)


def demand_scaling(ctx: Context, n_days: int = 15, scales=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0)) -> pd.DataFrame:
    """Deterministic vs uncertainty-aware plans as demand grows and the loading standard starts to bind."""
    cfg, kw, s = ctx.cfg, _kw(ctx.cfg), ctx.cfg["simulation"]
    days = [d for d in ctx.test_days if d.dayofweek < 5][:n_days]
    inst0, net = ctx.inst["weekday"], ctx.net["weekday"]
    rows = []
    for sc in scales:
        hist = replace(inst0, demand=inst0.demand * sc)
        static = {
            "existing_gtfs": H.existing_schedule(hist),
            "sqrt_rule_hist": H.sqrt_rule(hist),
            "milp_historical": solve_frequency_milp(hist, **kw),
        }
        for di, date in enumerate(days):
            rng = np.random.default_rng([s["seed"], 100 + di])
            actual, f1 = ctx.actual[date] * sc, ctx.forecast[1][date] * sc
            fi = inst0.with_demand(f1)
            scen = make_scenarios(f1, ctx.pool["weekday"], cfg["uncertainty"]["n_scenarios"], rng)
            plans = {
                **static,
                "milp_forecast": solve_frequency_milp(fi, **kw),
                "milp_saa": solve_saa(fi, scen, **kw),
                "milp_quantile_90": solve_quantile_robust(
                    fi, scen, cfg["uncertainty"]["robust_quantile"], **kw
                ),
                "milp_saa_cvar": solve_cvar(fi, scen, 1.0, cfg["uncertainty"]["cvar_alpha"], **kw),
                "milp_oracle": solve_frequency_milp(inst0.with_demand(actual), **kw),
            }
            draw = draw_passengers(net, actual, rng)
            for name, sol in plans.items():
                ev = evaluate(inst0, sol.freq, actual)
                sim = simulate(
                    net,
                    sol.freq,
                    draw,
                    seed=s["seed"] + di,
                    travel_time_cv=s["travel_time_cv"],
                    dwell_per_pax_s=s["dwell_per_pax_s"],
                )
                rows.append(
                    {
                        "demand_scale": sc,
                        "date": date,
                        "policy": name,
                        "status": sol.status,
                        "gap": sol.gap,
                        "runtime_s": sol.runtime_s,
                        "model_mean_wait_min": ev["mean_wait_min"],
                        "model_overload_pax": ev["overload_pax"],
                        "model_overload_share": ev["overload_share"],
                        "model_objective": ev["objective"],
                        "vehicle_hours": ev["vehicle_hours"],
                        **{f"sim_{k}": sim[k] for k in SIM_KEYS},
                    }
                )
        print(f"demand scale {sc} done", flush=True)
    return pd.DataFrame(rows)


def fleet_reduction_on_days(ctx: Context, n_days: int = 15, scales=(0.7, 0.8, 0.9, 1.0, 1.1)) -> pd.DataFrame:
    """Simulated effect of fleet and budget changes with forecast-informed re-optimisation."""
    kw, s = _kw(ctx.cfg), ctx.cfg["simulation"]
    days = [d for d in ctx.test_days if d.dayofweek < 5][:n_days]
    inst0, net = ctx.inst["weekday"], ctx.net["weekday"]
    rows = []
    for sc in scales:
        inst = inst0.scaled(fleet_scale=sc)
        for di, date in enumerate(days):
            rng = np.random.default_rng([s["seed"], 200 + di])
            draw = draw_passengers(net, ctx.actual[date], rng)
            plans = {
                "milp_forecast": solve_frequency_milp(inst.with_demand(ctx.forecast[1][date]), **kw),
                "sqrt_rule_forecast": H.sqrt_rule(inst.with_demand(ctx.forecast[1][date])),
            }
            for name, sol in plans.items():
                if sol.objective is None:
                    rows.append({"fleet_scale": sc, "date": date, "policy": name, "status": sol.status})
                    continue
                ev = evaluate(inst, sol.freq, ctx.actual[date])
                sim = simulate(
                    net,
                    sol.freq,
                    draw,
                    seed=s["seed"] + di,
                    travel_time_cv=s["travel_time_cv"],
                    dwell_per_pax_s=s["dwell_per_pax_s"],
                )
                rows.append(
                    {
                        "fleet_scale": sc,
                        "date": date,
                        "policy": name,
                        "status": sol.status,
                        "model_mean_wait_min": ev["mean_wait_min"],
                        "vehicle_hours": ev["vehicle_hours"],
                        "peak_fleet": ev["peak_fleet"],
                        **{f"sim_{k}": sim[k] for k in SIM_KEYS},
                    }
                )
    return pd.DataFrame(rows)


def run(cfg: dict | None = None) -> None:
    ctx = load_context(cfg)
    tracker = Run("sensitivity", ctx.cfg)
    res = resolve("results")
    fr = resource_frontier(ctx)
    fr.to_csv(res / "resource_frontier.csv", index=False)
    print(
        fr[fr.policy == "milp"][
            ["experiment", "scale", "status", "mean_wait_min", "vehicle_hours", "peak_fleet"]
        ]
        .round(3)
        .to_string()
    )
    fl = fleet_reduction_on_days(ctx)
    fl.to_csv(res / "fleet_sensitivity_days.csv", index=False)
    ds = demand_scaling(ctx)
    ds.to_csv(res / "demand_scaling_days.csv", index=False)
    tracker.log(infeasible_frontier_rows=int((~fr.feasible).sum()), frontier_rows=len(fr))
    tracker.finish()


if __name__ == "__main__":
    run()
