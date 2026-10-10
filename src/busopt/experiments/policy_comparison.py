"""E2: schedule policies on held-out test days (RQ2, RQ3, RQ5).

Every policy receives the same fleet and vehicle-hour budget (those of the projected existing
schedule for the day type) and is scored against the same realised demand: analytically with the
planning formula and by passenger simulation with common random numbers.

Static policies are fixed per day type and built from training-window demand only. Daily policies
are re-optimised for each test day from information available the evening before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from busopt.config import resolve
from busopt.experiments.common import SIM_KEYS, Context, daytype_of_date, load_context, paired_summary
from busopt.opt import heuristics as H
from busopt.opt.milp import Solution, solve_frequency_milp, verify
from busopt.opt.problem import evaluate
from busopt.opt.stochastic import make_scenarios, solve_quantile_robust, solve_saa
from busopt.sim.simulator import draw_passengers, simulate
from busopt.tracking import Run


def static_plans(ctx: Context, kw: dict) -> dict[str, dict[str, Solution]]:
    plans = {}
    for dt, inst in ctx.inst.items():
        plans[dt] = {
            "existing_gtfs": H.existing_schedule(inst),
            "fixed_frequency": H.flat_frequency(inst),
            "proportional_load_hist": H.proportional(inst),
            "sqrt_rule_hist": H.sqrt_rule(inst),
            "greedy_hist": H.greedy(inst),
            "milp_historical": solve_frequency_milp(inst, **kw),
        }
    return plans


def daily_plans(
    ctx: Context,
    date: pd.Timestamp,
    kw: dict,
    rng: np.random.Generator,
    demand_scale: float = 1.0,
    which: tuple[str, ...] = (
        "sqrt_rule_forecast",
        "milp_forecast",
        "milp_forecast_h7",
        "milp_saa",
        "milp_quantile_90",
        "milp_oracle",
    ),
) -> tuple[dict[str, Solution], dict]:
    dt = daytype_of_date(date)
    inst = ctx.inst[dt]
    f1 = ctx.forecast[1][date] * demand_scale
    out: dict[str, Solution] = {}
    extra: dict = {}
    scen = make_scenarios(f1, ctx.pool[dt], ctx.cfg["uncertainty"]["n_scenarios"], rng)
    if "sqrt_rule_forecast" in which:
        out["sqrt_rule_forecast"] = H.sqrt_rule(inst.with_demand(f1))
    if "milp_forecast" in which:
        out["milp_forecast"] = solve_frequency_milp(inst.with_demand(f1), **kw)
    if "milp_forecast_h7" in which and 7 in ctx.forecast:
        out["milp_forecast_h7"] = solve_frequency_milp(
            inst.with_demand(ctx.forecast[7][date] * demand_scale), **kw
        )
    if "milp_saa" in which:
        out["milp_saa"] = solve_saa(inst.with_demand(f1), scen, **kw)
        extra["saa_verify"] = verify(inst.with_demand(f1), out["milp_saa"], scenarios=scen)["verified"]
    if "milp_quantile_90" in which:
        out["milp_quantile_90"] = solve_quantile_robust(
            inst.with_demand(f1), scen, ctx.cfg["uncertainty"]["robust_quantile"], **kw
        )
    if "milp_oracle" in which:
        out["milp_oracle"] = solve_frequency_milp(inst.with_demand(ctx.actual[date] * demand_scale), **kw)
    return out, extra


def run(
    cfg: dict | None = None, max_days: int | None = None, out_name: str = "policy_comparison"
) -> pd.DataFrame:
    ctx = load_context(cfg)
    cfg = ctx.cfg
    o, s = cfg["optimization"], cfg["simulation"]
    kw = {"solver": o["solver"], "time_limit_s": o["time_limit_s"], "mip_gap": o["mip_gap"]}
    tracker = Run(out_name, cfg)
    static = static_plans(ctx, kw)
    days = ctx.test_days[:max_days] if max_days else ctx.test_days
    rows, hists = [], {}
    for di, date in enumerate(days):
        dt = daytype_of_date(date)
        inst, net = ctx.inst[dt], ctx.net[dt]
        actual = ctx.actual[date]
        rng = np.random.default_rng([s["seed"], di])
        daily, extra = daily_plans(ctx, date, kw, rng)
        draw = draw_passengers(net, actual, rng)
        for name, sol in {**static[dt], **daily}.items():
            ev = evaluate(inst, sol.freq, actual)
            sim = simulate(
                net,
                sol.freq,
                draw,
                seed=s["seed"] + di,
                travel_time_cv=s["travel_time_cv"],
                dwell_per_pax_s=s["dwell_per_pax_s"],
            )
            hists[name] = hists.get(name, 0) + sim["wait_hist"]
            rows.append(
                {
                    "date": date,
                    "daytype": dt,
                    "policy": name,
                    "status": sol.status,
                    "gap": sol.gap,
                    "runtime_s": sol.runtime_s,
                    "plan_feasible": ev["feasible"],
                    "model_mean_wait_min": ev["mean_wait_min"],
                    "model_overload_pax": ev["overload_pax"],
                    "model_objective": ev["objective"],
                    "vehicle_hours": ev["vehicle_hours"],
                    "peak_fleet": ev["peak_fleet"],
                    "fleet_limit": inst.fleet,
                    "budget_vh": inst.budget_vh,
                    "fleet_utilization": ev["vehicle_hours"] / (inst.fleet * len(inst.hours)),
                    "operating_cost_usd": ev["vehicle_hours"] * o["cost_per_vehicle_hour"],
                    "model_max_load_factor": ev["max_load_factor"],
                    **{f"sim_{k}": sim[k] for k in SIM_KEYS},
                    **extra,
                }
            )
        if (di + 1) % 10 == 0:
            print(f"{di + 1}/{len(days)} days", flush=True)
    df = pd.DataFrame(rows)
    res = resolve("results")
    df.to_csv(res / f"{out_name}_days.csv", index=False)
    np.savez_compressed(res / f"{out_name}_wait_hist.npz", **hists)
    summ = pd.concat(
        [
            paired_summary(df, m, "existing_gtfs")
            for m in (
                "model_mean_wait_min",
                "sim_mean_wait_min",
                "sim_p90_wait_min",
                "sim_p95_wait_min",
                "sim_overcrowded_pax_min",
                "sim_unmet_share",
                "vehicle_hours",
                "sim_load_utilization",
            )
        ]
    )
    summ.to_csv(res / f"{out_name}_summary.csv", index=False)
    vs_hist = pd.concat(
        [paired_summary(df, m, "milp_historical") for m in ("model_mean_wait_min", "sim_mean_wait_min")]
    )
    vs_hist.to_csv(res / f"{out_name}_vs_milp_historical.csv", index=False)
    tracker.log(
        n_days=len(days),
        all_plans_feasible=bool(df.plan_feasible.all()),
        statuses=df.groupby("policy").status.agg(lambda x: x.value_counts().to_dict()).to_dict(),
        max_gap=float(df.gap.max()),
        mean_runtime_s=df.groupby("policy").runtime_s.mean().to_dict(),
    )
    tracker.finish()
    return df


if __name__ == "__main__":
    d = run()
    pd.set_option("display.width", 220)
    print(
        d.groupby("policy")[
            [
                "model_mean_wait_min",
                "sim_mean_wait_min",
                "sim_p90_wait_min",
                "sim_p95_wait_min",
                "vehicle_hours",
                "sim_overcrowded_pax_min",
                "runtime_s",
            ]
        ]
        .mean()
        .round(3)
    )
