"""E6: robustness of plans to demand, running-time and vehicle uncertainty (RQ4, RQ5).

All stress scenarios are synthetic and applied to observed test-day demand. Each plan is built
without knowledge of the scenario; the same passengers and the same scenario draw are used for every
plan (common random numbers).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from busopt.config import resolve
from busopt.experiments.common import SIM_KEYS, Context, load_context
from busopt.opt import heuristics as H
from busopt.opt.milp import solve_frequency_milp
from busopt.opt.problem import evaluate
from busopt.opt.stochastic import make_scenarios, solve_quantile_robust, solve_saa, solve_slack
from busopt.sim import disruptions as D
from busopt.sim.simulator import draw_passengers, simulate
from busopt.tracking import Run

SCENARIOS = [
    "nominal",
    "tt_cv_0.25",
    "congestion_+15%",
    "congestion_+30%",
    "breakdowns_5%",
    "breakdowns_10%",
    "route_disruption",
    "demand_spike",
    "demand_x1.5",
    "demand_x2",
]


def run(cfg: dict | None = None, n_days: int = 15, slack: float = 0.10) -> pd.DataFrame:
    ctx: Context = load_context(cfg)
    cfg = ctx.cfg
    o, s = cfg["optimization"], cfg["simulation"]
    kw = {"solver": o["solver"], "time_limit_s": o["time_limit_s"], "mip_gap": o["mip_gap"]}
    tracker = Run("stress", cfg)
    inst, net = ctx.inst["weekday"], ctx.net["weekday"]
    days = [d for d in ctx.test_days if d.dayofweek < 5][:n_days]
    static = {
        "existing_gtfs": (H.existing_schedule(inst), None),
        "sqrt_rule_hist": (H.sqrt_rule(inst), None),
        "milp_historical": (solve_frequency_milp(inst, **kw), None),
    }
    rows = []
    for di, date in enumerate(days):
        rng = np.random.default_rng([s["seed"], 300 + di])
        f1, actual = ctx.forecast[1][date], ctx.actual[date]
        fi = inst.with_demand(f1)
        scen = make_scenarios(f1, ctx.pool["weekday"], cfg["uncertainty"]["n_scenarios"], rng)
        slack_cycle = inst.cycle_min * (1 + slack)
        plans = {
            **static,
            "milp_forecast": (solve_frequency_milp(fi, **kw), None),
            "milp_saa": (solve_saa(fi, scen, **kw), None),
            "milp_quantile_90": (
                solve_quantile_robust(fi, scen, cfg["uncertainty"]["robust_quantile"], **kw),
                None,
            ),
            f"milp_slack_{int(slack * 100)}": (solve_slack(fi, slack, **kw), slack_cycle),
        }
        # scenario draws shared by all plans
        spike = D.demand_spike(inst, actual, np.random.default_rng([s["seed"], 400 + di]))
        disrupt = D.route_disruption(inst, np.random.default_rng([s["seed"], 500 + di]))
        demands = {
            "nominal": actual,
            "demand_spike": spike,
            "demand_x1.5": actual * 1.5,
            "demand_x2": actual * 2.0,
        }
        draws = {
            k: draw_passengers(net, d, np.random.default_rng([s["seed"], 600 + di]))
            for k, d in demands.items()
        }
        for name, (sol, pc) in plans.items():
            nominal_ev = evaluate(replace(inst, cycle_min=pc) if pc is not None else inst, sol.freq, actual)
            for sc in SCENARIOS:
                freq, rf, cv, dkey = sol.freq.astype(float), None, s["travel_time_cv"], "nominal"
                if sc == "tt_cv_0.25":
                    cv = 0.25
                elif sc.startswith("congestion"):
                    f = 1.15 if "15" in sc else 1.30
                    rf = D.congestion(inst, f)
                    freq = D.effective_frequency(inst, freq, rf, planned_cycle=pc)
                elif sc.startswith("breakdowns"):
                    p = 0.05 if "5%" in sc else 0.10
                    lost = D.breakdowns(
                        inst, sol.freq, p, np.random.default_rng([s["seed"], 700 + di]), planned_cycle=pc
                    )
                    freq = D.effective_frequency(inst, freq, vehicles_lost=lost, planned_cycle=pc)
                elif sc == "route_disruption":
                    rf = disrupt
                    freq = D.effective_frequency(inst, freq, rf, planned_cycle=pc)
                elif sc in demands:
                    dkey = sc
                sim = simulate(
                    net,
                    freq,
                    draws[dkey],
                    seed=s["seed"] + di,
                    travel_time_cv=cv,
                    dwell_per_pax_s=s["dwell_per_pax_s"],
                    run_time_factor=rf,
                )
                rows.append(
                    {
                        "date": date,
                        "policy": name,
                        "scenario": sc,
                        "status": sol.status,
                        "planned_vehicle_hours": nominal_ev["vehicle_hours"],
                        "planned_trips_per_hour_total": float(sol.freq.sum()),
                        "delivered_trips_share": float(freq.sum() / max(sol.freq.sum(), 1)),
                        **{f"sim_{k}": sim[k] for k in SIM_KEYS},
                    }
                )
        print(f"stress day {di + 1}/{len(days)}", flush=True)
    df = pd.DataFrame(rows)
    res = resolve("results")
    df.to_csv(res / "stress_days.csv", index=False)
    summ = (
        df.groupby(["scenario", "policy"])[
            [
                "sim_mean_wait_min",
                "sim_p95_wait_min",
                "sim_unmet_share",
                "sim_overcrowded_pax_min",
                "sim_denied_per_1000_pax",
                "delivered_trips_share",
                "planned_vehicle_hours",
            ]
        ]
        .mean()
        .reset_index()
    )
    summ.to_csv(res / "stress_summary.csv", index=False)
    tracker.log(n_days=len(days), scenarios=SCENARIOS)
    tracker.finish()
    return df


if __name__ == "__main__":
    d = run()
    pd.set_option("display.width", 250)
    print(d.pivot_table(index="policy", columns="scenario", values="sim_mean_wait_min").round(3).to_string())
