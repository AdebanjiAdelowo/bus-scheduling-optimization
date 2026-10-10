"""E7: solver scaling and heuristic-vs-exact gaps (RQ6)."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd

from busopt.config import resolve
from busopt.experiments.common import load_context
from busopt.opt import heuristics as H
from busopt.opt.milp import cost_tensor, solve_frequency_milp, verify
from busopt.opt.problem import Instance, buses_required
from busopt.opt.stochastic import make_scenarios, solve_cvar, solve_saa
from busopt.tracking import Run


def subset(inst: Instance, n_routes: int) -> Instance:
    idx = np.arange(n_routes)
    sub = replace(
        inst,
        routes=[inst.routes[i] for i in idx],
        demand=inst.demand[idx],
        peak_coef=inst.peak_coef[idx],
        cycle_min=inst.cycle_min[idx],
        capacity=inst.capacity[idx],
        active=inst.active[idx],
        gtfs_freq=inst.gtfs_freq[idx],
    )
    need = np.where(sub.active, buses_required(sub.gtfs_freq, sub.cycle_min), 0.0)
    return replace(sub, fleet=int(need.sum(axis=0).max()), budget_vh=float(need.sum()))


def replicate(inst: Instance, k: int, rng: np.random.Generator) -> Instance:
    """Synthetic larger network: k perturbed copies of the real instance (demand and cycle times jittered)."""
    dem = np.concatenate([inst.demand * rng.lognormal(0, 0.3, inst.shape) for _ in range(k)])
    cyc = np.concatenate([inst.cycle_min * rng.uniform(0.8, 1.25, (inst.shape[0], 1)) for _ in range(k)])
    rep = replace(
        inst,
        routes=[f"{r}#{j}" for j in range(k) for r in inst.routes],
        demand=dem,
        peak_coef=np.tile(inst.peak_coef, (k, 1)),
        cycle_min=cyc,
        capacity=np.tile(inst.capacity, k),
        active=np.tile(inst.active, (k, 1)),
        gtfs_freq=np.tile(inst.gtfs_freq, (k, 1)),
    )
    need = np.where(rep.active, buses_required(rep.gtfs_freq, rep.cycle_min), 0.0)
    return replace(rep, fleet=int(need.sum(axis=0).max()), budget_vh=float(need.sum()))


def run(
    cfg: dict | None = None, solvers=("highs", "scip", "cbc", "cpsat"), time_limit_s: float = 60.0
) -> pd.DataFrame:
    ctx = load_context(cfg)
    tracker = Run("scalability", ctx.cfg)
    base = ctx.inst["weekday"]
    rng = np.random.default_rng(ctx.cfg["simulation"]["seed"])
    cases = [(f"real_{n}_routes", subset(base, n)) for n in (5, 10, 20)] + [("real_31_routes", base)]
    cases += [(f"synthetic_x{k}", replicate(base, k, rng)) for k in (2, 4, 8, 16)]
    rows = []
    for label, inst in cases:
        lp = solve_frequency_milp(inst, relax=True)
        best_bound = lp.objective
        exact = []
        for sv in solvers:
            sol = solve_frequency_milp(
                inst, solver=sv, time_limit_s=time_limit_s, mip_gap=ctx.cfg["optimization"]["mip_gap"]
            )
            v = verify(inst, sol)
            exact.append(sol)
            rows.append(
                {
                    "case": label,
                    "routes": len(inst.routes),
                    "method": sv,
                    "solver_version": sol.solver_version,
                    "status": sol.status,
                    "objective": sol.objective,
                    "bound": sol.bound,
                    "gap": sol.gap,
                    "runtime_s": sol.runtime_s,
                    "n_vars": sol.n_vars,
                    "n_cons": sol.n_cons,
                    "verified": v.get("verified"),
                    "time_limit_s": time_limit_s,
                }
            )
            if sol.bound is not None:
                best_bound = max(best_bound, sol.bound)
        best_obj = min((s.objective for s in exact if s.objective is not None), default=np.nan)
        heur = [H.existing_schedule(inst), H.proportional(inst), H.sqrt_rule(inst), H.greedy(inst)]
        g = heur[-1]
        if g.objective is not None and len(inst.routes) <= 62:
            heur.append(H.local_search(inst, g.freq))
        for sol in heur:
            rows.append(
                {
                    "case": label,
                    "routes": len(inst.routes),
                    "method": sol.method,
                    "status": sol.status,
                    "objective": sol.objective,
                    "bound": best_bound,
                    "runtime_s": sol.runtime_s,
                    "gap": None if sol.objective is None else (sol.objective - best_bound) / sol.objective,
                    "gap_to_best_exact": None
                    if sol.objective is None
                    else (sol.objective - best_obj) / best_obj,
                }
            )
        rows.append(
            {
                "case": label,
                "routes": len(inst.routes),
                "method": "lp_relaxation",
                "status": lp.status,
                "objective": lp.objective,
                "runtime_s": lp.runtime_s,
                "n_vars": lp.n_vars,
                "n_cons": lp.n_cons,
            }
        )
        print(label, "done", flush=True)
    # scenario count scaling on the real instance
    f = base.demand
    for n_s in (10, 30, 100, 300):
        scen = make_scenarios(f, ctx.pool["weekday"], n_s, rng)
        t0 = time.perf_counter()
        cost_tensor(base, scen)
        prep = time.perf_counter() - t0
        sol = solve_saa(base, scen, time_limit_s=time_limit_s)
        rows.append(
            {
                "case": f"saa_S{n_s}",
                "routes": 31,
                "method": "milp_saa",
                "status": sol.status,
                "objective": sol.objective,
                "bound": sol.bound,
                "gap": sol.gap,
                "runtime_s": sol.runtime_s,
                "n_vars": sol.n_vars,
                "n_cons": sol.n_cons,
                "cost_tensor_s": prep,
                "verified": verify(base, sol, scenarios=scen)["verified"],
            }
        )
        if n_s <= 100:
            hi = replace(
                base, demand=base.demand * 2.5
            )  # recourse variables only exist where capacity can bind
            scen_hi = scen * 2.5
            sol = solve_cvar(hi, scen_hi, 1.0, 0.9, time_limit_s=time_limit_s)
            rows.append(
                {
                    "case": f"cvar_S{n_s}_demand_x2.5",
                    "routes": 31,
                    "method": "milp_saa_cvar",
                    "status": sol.status,
                    "objective": sol.objective,
                    "bound": sol.bound,
                    "gap": sol.gap,
                    "runtime_s": sol.runtime_s,
                    "n_vars": sol.n_vars,
                    "n_cons": sol.n_cons,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(resolve("results") / "scalability.csv", index=False)
    tracker.log(cases=[c for c, _ in cases])
    tracker.finish()
    return df


if __name__ == "__main__":
    d = run()
    pd.set_option("display.width", 250)
    print(d[["case", "method", "status", "objective", "gap", "runtime_s", "n_vars"]].to_string())
