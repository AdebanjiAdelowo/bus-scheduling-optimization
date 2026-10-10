import itertools
from dataclasses import replace

import numpy as np
import pytest

from busopt.opt import heuristics as H
from busopt.opt.milp import binding_constraints, cost_tensor, solve_frequency_milp, verify
from busopt.opt.problem import buses_required, evaluate
from busopt.opt.stochastic import make_scenarios, solve_cvar, solve_quantile_robust, solve_saa, solve_slack


def tiny(inst, scale=1.0):
    """2 routes x 2 hours with frequencies 2..6: small enough to enumerate (5^4 plans)."""
    sub = replace(
        inst,
        routes=inst.routes[:2],
        hours=inst.hours[3:5],
        demand=inst.demand[:2, 3:5] * scale,
        peak_coef=inst.peak_coef[:2, 3:5],
        cycle_min=inst.cycle_min[:2, 3:5],
        capacity=inst.capacity[:2],
        active=inst.active[:2, 3:5],
        gtfs_freq=np.full((2, 2), 4),
        freq_max=6,
    )
    need = buses_required(sub.gtfs_freq, sub.cycle_min)
    return replace(sub, fleet=int(need.sum(axis=0).max()), budget_vh=float(need.sum()))


def brute_force(inst, scenarios=None):
    best, best_x = np.inf, None
    scen = [inst.demand] if scenarios is None else scenarios
    for combo in itertools.product(range(inst.freq_min, inst.freq_max + 1), repeat=4):
        x = np.array(combo).reshape(2, 2)
        if not evaluate(inst, x)["feasible"]:
            continue
        obj = np.mean([evaluate(inst, x, d)["objective"] for d in scen])
        if obj < best - 1e-9:
            best, best_x = obj, x
    return best, best_x


@pytest.mark.parametrize("solver", ["highs", "cpsat", "scip", "cbc"])
@pytest.mark.parametrize("scale", [1.0, 40.0])
def test_milp_matches_enumeration(inst, solver, scale):
    t = tiny(inst, scale)
    sol = solve_frequency_milp(t, solver=solver, time_limit_s=20)
    best, _ = brute_force(t)
    assert sol.status.startswith("OPTIMAL")
    assert np.isclose(sol.objective, best, rtol=1e-6)
    assert verify(t, sol)["verified"]


def test_saa_matches_enumeration(inst):
    t = tiny(inst, 40.0)
    rng = np.random.default_rng(0)
    scen = t.demand[None] * rng.lognormal(0, 0.4, (8, 2, 2))
    sol = solve_saa(t, scen, time_limit_s=20)
    best, _ = brute_force(t, scen)
    assert np.isclose(sol.objective, best, rtol=1e-6)
    assert verify(t, sol, scenarios=scen)["verified"]


def test_infeasible_when_fleet_too_small(inst):
    sol = solve_frequency_milp(replace(inst, fleet=1), time_limit_s=10)
    assert sol.status == "INFEASIBLE" and sol.objective is None
    assert not verify(inst, sol)["verified"]
    assert H.greedy(replace(inst, fleet=1)).status == "INFEASIBLE"


def test_lp_bound_and_heuristics_bracket_optimum(inst):
    opt = solve_frequency_milp(inst, time_limit_s=30)
    lp = solve_frequency_milp(inst, relax=True)
    assert lp.objective <= opt.objective + 1e-6
    for h in (
        H.existing_schedule(inst),
        H.flat_frequency(inst),
        H.proportional(inst),
        H.sqrt_rule(inst),
        H.greedy(inst),
    ):
        assert evaluate(inst, h.freq)["feasible"], h.method
        assert h.objective >= opt.objective - 1e-6, h.method
    ls = H.local_search(inst, H.greedy(inst).freq)
    assert ls.objective <= H.greedy(inst).objective + 1e-9


def test_more_resources_never_hurt(inst):
    a = solve_frequency_milp(inst, time_limit_s=30).objective
    b = solve_frequency_milp(inst.scaled(fleet_scale=1.2), time_limit_s=30).objective
    assert b <= a + 1e-6


def test_cost_weight_reduces_vehicle_hours(inst):
    loose = inst.scaled(fleet_scale=1.5)
    free = solve_frequency_milp(loose, time_limit_s=30)
    priced = solve_frequency_milp(loose, cost_weight=500.0, time_limit_s=30)
    assert evaluate(loose, priced.freq)["vehicle_hours"] < evaluate(loose, free.freq)["vehicle_hours"]
    assert verify(loose, priced, cost_weight=500.0)["verified"]


def test_uncertainty_aware_models_are_feasible_and_ordered(inst):
    hi = inst.with_demand(inst.demand * 12)
    rng = np.random.default_rng(1)
    pool = rng.lognormal(0, 0.3, (20, *inst.shape))
    scen = make_scenarios(hi.demand, pool, 12, rng)
    assert scen.shape == (12, *inst.shape)
    det, saa = solve_frequency_milp(hi, time_limit_s=30), solve_saa(hi, scen, time_limit_s=30)
    cvar = solve_cvar(hi, scen, 1.0, 0.9, time_limit_s=30)
    rob = solve_quantile_robust(hi, scen, 0.9, time_limit_s=30)
    slack = solve_slack(hi, 0.1, time_limit_s=30)
    for s in (det, saa, cvar, rob, slack):
        assert evaluate(hi, s.freq)["feasible"], s.method

    def exp_cost(x):
        return np.mean([evaluate(hi, x, d)["objective"] for d in scen])

    assert exp_cost(saa.freq) <= exp_cost(det.freq) + 1e-6  # SAA is optimal for its own scenario set
    # the slack plan fits the fleet even when every cycle time is 10% longer
    assert evaluate(replace(hi, cycle_min=hi.cycle_min * 1.1), slack.freq)["violations"]["fleet"] == 0


def test_binding_constraints_report(inst):
    sol = solve_frequency_milp(inst, time_limit_s=30)
    b = binding_constraints(inst, sol.freq)
    assert len(b["fleet"]) == len(inst.hours)
    assert b["budget"]["used"] <= b["budget"]["limit"] + 1e-6
    assert all(f["slack"] >= -1e-6 for f in b["fleet"])


def test_cost_tensor_is_convex_decreasing_in_wait(inst):
    ks, c, n = cost_tensor(inst)
    assert (
        np.diff(c, axis=2)[inst.active] <= 1e-9
    ).all()  # no overload at toy demand: cost falls with frequency
    assert (np.diff(n, axis=2) >= 0).all()
