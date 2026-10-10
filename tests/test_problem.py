import numpy as np

from busopt.opt.problem import buses_required, evaluate, wait_minutes


def test_wait_formula():
    assert wait_minutes(np.array([6.0]))[0] == 5.0  # 10-minute headway -> 5-minute mean wait
    assert np.isclose(
        wait_minutes(np.array([6.0]), headway_cv=1.0)[0], 10.0
    )  # exponential headways double it


def test_buses_required_is_ceiling():
    assert buses_required(np.array([6.0]), np.array([60.0]))[0] == 6
    assert buses_required(np.array([6.0]), np.array([61.0]))[0] == 7
    assert buses_required(np.array([0.0]), np.array([61.0]))[0] == 0


def test_existing_schedule_defines_resources(inst):
    ev = evaluate(inst, inst.gtfs_freq)
    assert ev["feasible"]
    assert ev["vehicle_hours"] == inst.budget_vh
    assert ev["peak_fleet"] == inst.fleet


def test_evaluate_flags_violations(inst):
    ev = evaluate(inst, inst.gtfs_freq + 5)
    assert not ev["feasible"] and ev["violations"]["fleet"] > 0 and ev["violations"]["budget"] > 0
    low = np.where(inst.active, 1, 0)
    assert evaluate(inst, low)["violations"]["freq_min"] == int(inst.active.sum())


def test_overload_counts_passengers_above_standard(inst):
    big = inst.with_demand(inst.demand * 50)
    ev = evaluate(big, inst.gtfs_freq)
    expected = np.clip(big.peak_coef * big.demand - inst.gtfs_freq * big.capacity[:, None], 0, None).sum()
    assert np.isclose(ev["overload_pax"], expected) and expected > 0
