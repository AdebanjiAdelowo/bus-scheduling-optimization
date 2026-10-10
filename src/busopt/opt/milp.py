"""Exact frequency setting as a multiple-choice mixed-integer linear program.

For every active (route r, hour t) exactly one frequency k in {k_min, ..., k_max} is selected by a
binary y[r,t,k]. Because the frequency is discrete, the non-linear waiting term d/k and the ceiling
in the fleet requirement ceil(k * tau / 60) become *data*, so the model is an exact MILP rather than
a piecewise-linear approximation.

    min   sum_{r,t,k} c[r,t,k] y[r,t,k]
    s.t.  sum_k y[r,t,k] = 1                              for all active (r,t)        (assignment)
          sum_r sum_k n[r,t,k] y[r,t,k] <= F              for all t                   (fleet)
          sum_{r,t,k} n[r,t,k] y[r,t,k] <= B                                          (vehicle-hours)
          y binary

    c[r,t,k] = dbar[r,t] * 30/k * (1+cv^2)  +  p * mean_s max(0, coef[r,t] d_s[r,t] - C_r k)
    n[r,t,k] = ceil(k * tau[r,t] / 60)

With one scenario this is the deterministic model. With S scenarios it is the sample-average
approximation of the two-stage stochastic program whose recourse (passengers above the loading
standard) is separable by (r,t), so the expectation is folded exactly into c. The optional CVaR term
couples scenarios and needs explicit recourse variables.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from ortools.linear_solver import pywraplp

from busopt.opt.problem import Instance, buses_required, evaluate

SOLVER_NAMES = {"highs": "HIGHS", "scip": "SCIP", "cbc": "CBC", "cpsat": "SAT"}


@dataclass
class Solution:
    freq: np.ndarray
    status: str
    objective: float | None
    bound: float | None
    gap: float | None
    runtime_s: float
    solver: str
    solver_version: str
    n_vars: int
    n_cons: int
    method: str = "milp"
    info: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            k: getattr(self, k)
            for k in (
                "status",
                "objective",
                "bound",
                "gap",
                "runtime_s",
                "solver",
                "solver_version",
                "n_vars",
                "n_cons",
                "method",
            )
        }


def cost_tensor(
    inst: Instance, scenarios: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ks, c[R,T,K], n[R,T,K]). ``scenarios`` has shape [S,R,T]; default is ``inst.demand``."""
    ks = np.arange(inst.freq_min, inst.freq_max + 1)
    scen = inst.demand[None] if scenarios is None else np.where(inst.active[None], scenarios, 0.0)
    dbar = scen.mean(axis=0)
    wait = dbar[:, :, None] * 30.0 / ks[None, None, :] * (1.0 + inst.headway_cv**2)
    load = inst.peak_coef[None] * scen  # [S,R,T]
    cap = inst.capacity[:, None, None] * ks[None, None, :]  # [R,1,K]
    over = np.clip(load[:, :, :, None] - cap[None], 0.0, None).mean(axis=0)
    n = buses_required(ks[None, None, :], inst.cycle_min[:, :, None])
    return ks, wait + inst.unmet_penalty_min * over, n


def solve_frequency_milp(
    inst: Instance,
    scenarios: np.ndarray | None = None,
    solver: str = "highs",
    time_limit_s: float = 60.0,
    mip_gap: float = 1e-4,
    cost_weight: float = 0.0,
    cvar_weight: float = 0.0,
    cvar_alpha: float = 0.9,
    relax: bool = False,
    threads: int = 2,
) -> Solution:
    """Solve the model. ``cost_weight`` (passenger-minutes per vehicle-hour) adds operating cost to the
    objective; ``relax`` solves the LP relaxation (used for bounds and shadow prices)."""
    ks, c, n = cost_tensor(inst, scenarios)
    R, T = inst.shape
    K = len(ks)
    name = "GLOP" if relax else SOLVER_NAMES[solver]
    s = pywraplp.Solver.CreateSolver(name)
    if s is None:
        raise RuntimeError(f"solver {name} is not available in this OR-Tools build")
    s.SetNumThreads(threads) if name in ("SAT", "HIGHS") else None
    inf = s.infinity()
    y: dict[tuple[int, int, int], pywraplp.Variable] = {}
    obj = s.Objective()
    fleet_c = [s.Constraint(-inf, float(inst.fleet), f"fleet_{t}") for t in range(T)]
    budget_c = s.Constraint(-inf, float(inst.budget_vh) + 1e-6, "budget")
    for r in range(R):
        for t in range(T):
            if not inst.active[r, t]:
                continue
            assign = s.Constraint(1.0, 1.0, f"assign_{r}_{t}")
            for j in range(K):
                v = s.NumVar(0.0, 1.0, "") if relax else s.BoolVar("")
                y[r, t, j] = v
                assign.SetCoefficient(v, 1.0)
                fleet_c[t].SetCoefficient(v, float(n[r, t, j]))
                budget_c.SetCoefficient(v, float(n[r, t, j]))
                obj.SetCoefficient(v, float(c[r, t, j] + cost_weight * n[r, t, j]))
    if cvar_weight > 0 and scenarios is not None:
        # recourse o[s,r,t] >= load - C_r * x[r,t]; CVaR_alpha of the total via Rockafellar-Uryasev
        S = scenarios.shape[0]
        eta = s.NumVar(0.0, inf, "eta")
        obj.SetCoefficient(eta, cvar_weight * inst.unmet_penalty_min)
        load = inst.peak_coef[None] * np.where(inst.active[None], scenarios, 0.0)
        for sc in range(S):
            z = s.NumVar(0.0, inf, "")
            obj.SetCoefficient(z, cvar_weight * inst.unmet_penalty_min / ((1.0 - cvar_alpha) * S))
            tail = s.Constraint(-inf, 0.0, "")  # sum o - eta - z <= 0
            tail.SetCoefficient(eta, -1.0)
            tail.SetCoefficient(z, -1.0)
            for r in range(R):
                for t in range(T):
                    if not inst.active[r, t] or load[sc, r, t] <= inst.capacity[r] * inst.freq_min:
                        continue
                    o = s.NumVar(0.0, inf, "")
                    tail.SetCoefficient(o, 1.0)
                    con = s.Constraint(float(load[sc, r, t]), inf, "")  # o + C sum k y >= load
                    con.SetCoefficient(o, 1.0)
                    for j in range(K):
                        con.SetCoefficient(y[r, t, j], float(inst.capacity[r] * ks[j]))
    obj.SetMinimization()
    if not relax:
        s.SetTimeLimit(int(time_limit_s * 1000))
        params = pywraplp.MPSolverParameters()
        params.SetDoubleParam(pywraplp.MPSolverParameters.RELATIVE_MIP_GAP, mip_gap)
    t0 = time.perf_counter()
    status = s.Solve() if relax else s.Solve(params)
    runtime = time.perf_counter() - t0
    names = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    st = names.get(status, str(status))
    freq = np.zeros((R, T))
    objective = bound = gap = None
    info: dict = {}
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        for (r, t, j), v in y.items():
            freq[r, t] += ks[j] * v.solution_value()
        objective = float(obj.Value())
        if relax:
            bound, gap = objective, 0.0
            info["fleet_dual"] = [float(cst.dual_value()) for cst in fleet_c]
            info["budget_dual"] = float(budget_c.dual_value())
        else:
            freq = np.round(freq).astype(int)
            bound = float(obj.BestBound())
            gap = abs(objective - bound) / max(abs(objective), 1e-9)
            if st == "OPTIMAL" and gap > mip_gap * 1.01:
                st = "OPTIMAL_WITHIN_TOL"
    version = s.SolverVersion()
    if name == "HIGHS":
        version = f"HiGHS (bundled with OR-Tools {__import__('ortools').__version__})"
    return Solution(
        freq=freq,
        status=st,
        objective=objective,
        bound=bound,
        gap=gap,
        runtime_s=runtime,
        solver=name,
        solver_version=version,
        n_vars=s.NumVariables(),
        n_cons=s.NumConstraints(),
        method="lp_relaxation" if relax else "milp",
        info=info,
    )


def verify(
    inst: Instance,
    sol: Solution,
    scenarios: np.ndarray | None = None,
    cost_weight: float = 0.0,
    tol: float = 1e-4,
) -> dict:
    """Recompute feasibility and the objective without the solver and compare."""
    if sol.objective is None:
        return {"verified": False, "reason": sol.status}
    ev = evaluate(inst, sol.freq)
    scen = inst.demand[None] if scenarios is None else scenarios
    objs = [evaluate(inst, sol.freq, d)["objective"] for d in scen]
    recomputed = float(np.mean(objs)) + cost_weight * ev["vehicle_hours"]
    rel = abs(recomputed - sol.objective) / max(abs(sol.objective), 1e-9)
    return {
        "verified": bool(ev["feasible"] and rel <= tol),
        "feasible": ev["feasible"],
        "violations": ev["violations"],
        "solver_objective": sol.objective,
        "recomputed_objective": recomputed,
        "relative_difference": rel,
    }


def binding_constraints(inst: Instance, freq: np.ndarray, tol: float = 1e-6) -> dict:
    """Slack of every resource and policy constraint at a plan."""
    x = np.where(inst.active, freq, 0)
    n = np.where(inst.active, buses_required(x, inst.cycle_min), 0.0)
    fleet_slack = inst.fleet - n.sum(axis=0)
    at_min = (x <= inst.freq_min) & inst.active
    at_max = (x >= inst.freq_max) & inst.active
    load_ratio = np.where(
        x > 0, inst.peak_coef * inst.demand / np.maximum(x * inst.capacity[:, None], 1e-9), 0.0
    )
    return {
        "fleet": [
            {
                "hour": int(h),
                "used": float(u),
                "limit": int(inst.fleet),
                "slack": float(sl),
                "binding": bool(sl < 1 - tol),
            }
            for h, u, sl in zip(inst.hours, n.sum(axis=0), fleet_slack, strict=True)
        ],
        "budget": {
            "used": float(n.sum()),
            "limit": float(inst.budget_vh),
            "slack": float(inst.budget_vh - n.sum()),
            "binding": bool(inst.budget_vh - n.sum() < 1 - tol),
        },
        "min_frequency_cells": int(at_min.sum()),
        "max_frequency_cells": int(at_max.sum()),
        "capacity_cells_over_standard": int((load_ratio > 1 + tol).sum()),
        "capacity_cells_at_or_above_90pct": int((load_ratio >= 0.9).sum()),
        "active_cells": int(inst.active.sum()),
    }
