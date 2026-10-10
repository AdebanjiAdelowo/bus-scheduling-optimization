import numpy as np

from busopt.opt.problem import evaluate
from busopt.sim import disruptions as D
from busopt.sim.simulator import build_sim_network, draw_passengers, simulate


def _setup(tables, inst, scale=1.0, seed=0):
    net = build_sim_network(tables, inst)
    draw = draw_passengers(net, inst.demand * scale, np.random.default_rng(seed))
    return net, draw


def test_noise_free_simulation_matches_wait_formula(tables, inst):
    net, draw = _setup(tables, inst, scale=2.0)
    sim = simulate(
        net, inst.gtfs_freq, draw, seed=1, travel_time_cv=0.0, dwell_per_pax_s=0.0, dispatch_jitter_s=0.0
    )
    model = evaluate(inst, inst.gtfs_freq, inst.demand * 2.0)["mean_wait_min"]
    assert sim["unmet_capacity"] == 0  # the formula assumes nobody is left behind
    assert sim["passengers"] > 20_000  # enough passengers for a tight Monte Carlo error
    assert abs(sim["mean_wait_min"] - model) / model < 0.03


def test_passenger_conservation_and_capacity(tables, inst):
    net, draw = _setup(tables, inst, scale=60.0)
    sim = simulate(net, inst.gtfs_freq, draw, seed=2)
    assert sim["boarded"] + sim["unmet_capacity"] + sim["unmet_end_of_service"] == sim["passengers"]
    assert sim["max_load_factor"] <= (net.crush / inst.capacity).max() + 1e-9
    assert sim["unmet_capacity"] > 0 and sim["denied_boarding_events"] > 0  # x60 demand must overflow


def test_common_random_numbers_are_deterministic(tables, inst):
    net, draw = _setup(tables, inst)
    a = simulate(net, inst.gtfs_freq, draw, seed=3)
    b = simulate(net, inst.gtfs_freq, draw, seed=3)
    assert a["mean_wait_min"] == b["mean_wait_min"] and (a["wait_hist"] == b["wait_hist"]).all()


def test_more_service_means_less_waiting(tables, inst):
    net, draw = _setup(tables, inst, scale=5.0)
    lo = simulate(net, inst.gtfs_freq, draw, seed=4)
    hi = simulate(net, inst.gtfs_freq * 2, draw, seed=4)
    assert hi["mean_wait_min"] < lo["mean_wait_min"]
    assert hi["p95_wait_min"] <= lo["p95_wait_min"]


def test_effective_frequency_under_disruption(inst):
    x = inst.gtfs_freq.astype(float)
    assert np.allclose(D.effective_frequency(inst, x), np.where(inst.active, x, 0))
    slow = D.effective_frequency(inst, x, run_factor=D.congestion(inst, 1.3))
    assert (slow <= x + 1e-9).all() and slow.sum() < x.sum()
    lost = np.ones(inst.shape)
    assert D.effective_frequency(inst, x, vehicles_lost=lost).sum() < x.sum()
    spike = D.demand_spike(inst, inst.demand, np.random.default_rng(0), n_routes=2)
    assert spike.sum() > inst.demand.sum() and (spike >= inst.demand).all()
