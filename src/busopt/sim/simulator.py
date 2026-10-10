"""Passenger-level simulation of one service day on every route of an instance.

The simulator is the evaluation referee: it shares no code with the optimizer and does not assume
the optimizer's waiting-time formula. Individual passengers arrive at stops, buses run stop to stop
with random running times and load-dependent dwell, boarding is first-come-first-served up to a
physical (crush) capacity and passengers left behind wait for the next bus.

What is observed and what is simulated
--------------------------------------
* Route-hour demand totals passed in are observed fare-payment counts (or scaled versions of them).
* Their split by direction and stop uses observed APC shares from the training window.
* Passenger arrival instants (Poisson within the hour), running-time noise, dwell times and vehicle
  capacities are *simulated/assumed*. No observed travel times are used anywhere.

Time convention: demand of hour t at stop s is placed in the window in which buses dispatched during
hour t pass stop s (the hour window shifted by the scheduled running time to s). This removes the
start- and end-of-day boundary artefacts that would otherwise penalise long routes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from busopt.opt.problem import Instance

BIN_S = 10.0
N_BINS = 2160  # waits up to 6 h


@njit(cache=True)
def _run_direction(
    dep, hour_idx, factor, seg, seg_cv, offsets, arr, alight_prob, crush, guide, dwell_s, hist, seed
):
    np.random.seed(seed)
    n_trips = dep.shape[0]
    n_stops = seg.shape[1] + 1
    ptr = offsets[:-1].copy()
    prev = np.full(n_stops, -1e18)
    wait_sum = 0.0
    boarded = 0
    denied = 0
    pax_sec = 0.0
    over_sec = 0.0
    over_pax_sec = 0.0
    cap_sec = 0.0
    max_load = 0
    last_pass = np.zeros(n_stops)
    for k in range(n_trips):
        t = dep[k]
        h = hour_idx[k]
        load = 0
        for s in range(n_stops):
            a = t
            if a < prev[s] + 5.0:  # no overtaking: a bus cannot pass the one in front
                a = prev[s] + 5.0
            al = 0
            if load > 0:
                p = alight_prob[h, s]
                if p >= 1.0:
                    al = load
                elif p > 0.0:
                    al = np.random.binomial(load, p)
            load -= al
            end = offsets[s + 1]
            i = ptr[s]
            nb = 0
            while i < end and arr[i] <= a and load < crush:
                w = a - arr[i]
                b = int(w / BIN_S)
                if b >= N_BINS:
                    b = N_BINS - 1
                hist[b] += 1
                wait_sum += w
                load += 1
                i += 1
                nb += 1
            j = i
            while j < end and arr[j] <= a:
                j += 1
            denied += j - i
            ptr[s] = i
            boarded += nb
            depart = a + dwell_s * (nb + al)
            prev[s] = depart
            last_pass[s] = a
            if load > max_load:
                max_load = load
            if s < n_stops - 1:
                st = seg[h, s] * factor[k]
                if seg_cv > 0.0:
                    st *= max(0.3, 1.0 + seg_cv * np.random.standard_normal())
                pax_sec += load * st
                cap_sec += guide * st
                if load > guide:
                    over_sec += (load - guide) * st
                    over_pax_sec += load * st
                t = depart + st
    left_capacity = 0
    left_end = 0
    for s in range(n_stops):
        for i in range(ptr[s], offsets[s + 1]):
            if arr[i] <= last_pass[s]:
                left_capacity += 1
            else:
                left_end += 1
    return (
        wait_sum,
        boarded,
        denied,
        pax_sec,
        over_sec,
        over_pax_sec,
        cap_sec,
        max_load,
        left_capacity,
        left_end,
    )


@dataclass
class DirectionModel:
    route_idx: int
    direction_id: int
    cum_s: np.ndarray  # [T, S] scheduled cumulative running time by dispatch hour
    board_share: np.ndarray  # [T, S]
    alight_prob: np.ndarray  # [T, S]
    sigma: np.ndarray  # [T]


@dataclass
class SimNetwork:
    inst: Instance
    directions: list[DirectionModel]
    crush: np.ndarray  # [R]


def build_sim_network(tables: dict[str, pd.DataFrame], inst: Instance) -> SimNetwork:
    dt, hours = inst.daytype, inst.hours
    pat = tables["patterns"]
    ss = tables["stop_shares"]
    ss = ss[(ss.daytype == dt) & ss.hour.isin(hours)]
    summ = tables["profile_summary"]
    summ = summ[(summ.daytype == dt) & summ.hour.isin(hours)].set_index(["route_id", "direction_id", "hour"])
    rh = tables["route_hour"]
    rh = rh[(rh.daytype == dt)].set_index(["route_id", "hour"])
    info = tables["route_info"].set_index("route_id")
    dirs = []
    for ri, route in enumerate(inst.routes):
        for d in (0, 1):
            p = pat[(pat.route_id == route) & (pat.direction_id == d)].sort_values("seq")
            base = p.cum_time_s.to_numpy(float)
            n = len(p)
            run = np.array([rh[f"run_min_dir{d}"].get((route, h), np.nan) for h in hours]) * 60.0
            run = np.where(np.isfinite(run) & (run > 0), run, base[-1])
            cum = base[None, :] * (run[:, None] / base[-1])
            sub = ss[(ss.route_id == route) & (ss.direction_id == d)]
            b = (
                sub.pivot(index="hour", columns="seq", values="board_share")
                .reindex(index=hours, columns=range(n))
                .to_numpy(float)
            )
            a = (
                sub.pivot(index="hour", columns="seq", values="alight_share")
                .reindex(index=hours, columns=range(n))
                .to_numpy(float)
            )
            load_before = np.cumsum(b - a, axis=1) - (b - a)  # expected load arriving at stop s, per boarding
            with np.errstate(divide="ignore", invalid="ignore"):
                ap = np.where(load_before > 1e-9, a / np.maximum(load_before, 1e-9), 0.0)
            ap = np.clip(ap, 0.0, 1.0)
            ap[:, -1] = 1.0
            sigma = np.array([summ.sigma.get((route, d, h), 0.5) for h in hours], dtype=float)
            dirs.append(DirectionModel(ri, d, cum, b, ap, sigma))
    return SimNetwork(inst=inst, directions=dirs, crush=info.loc[inst.routes].capacity_crush.to_numpy(float))


@dataclass
class DemandDraw:
    """Passenger arrival instants for one day: identical for every policy (common random numbers)."""

    arr: list[np.ndarray]
    offsets: list[np.ndarray]
    generated: int


def draw_passengers(net: SimNetwork, demand: np.ndarray, rng: np.random.Generator) -> DemandDraw:
    h0 = net.inst.hours[0]
    arrs, offs, total = [], [], 0
    for dm in net.directions:
        lam = demand[dm.route_idx][:, None] * dm.sigma[:, None] * dm.board_share  # [T, S]
        lam = np.where(net.inst.active[dm.route_idx][:, None], lam, 0.0)
        counts = rng.poisson(lam)
        n_stops = counts.shape[1]
        times, off = [], [0]
        for s in range(n_stops):
            c = counts[:, s]
            tt = np.repeat(np.arange(len(c)), c)
            x = (h0 + tt + rng.random(tt.size)) * 3600.0 + dm.cum_s[tt, s]
            x.sort()
            times.append(x)
            off.append(off[-1] + x.size)
        arrs.append(np.concatenate(times) if times else np.zeros(0))
        offs.append(np.array(off, dtype=np.int64))
        total += off[-1]
    return DemandDraw(arrs, offs, total)


def _departures(
    freq_row: np.ndarray, hours: list[int], rng: np.random.Generator, jitter_s: float, cancel_prob: float
):
    start, end = hours[0] * 3600.0, (hours[-1] + 1) * 3600.0
    t, dep, hidx = None, [], []
    for i, f in enumerate(freq_row):
        if f <= 0:
            t = None
            continue
        hw = 3600.0 / f
        lo, hi = start + i * 3600.0, start + (i + 1) * 3600.0
        if t is None or t < lo:
            t = lo + rng.random() * min(hw, 3600.0) if t is None else max(t, lo)
        while t < hi and t < end:
            dep.append(t)
            hidx.append(i)
            t += hw
    dep, hidx = np.array(dep), np.array(hidx, dtype=np.int64)
    if cancel_prob > 0 and dep.size:
        keep = rng.random(dep.size) >= cancel_prob
        dep, hidx = dep[keep], hidx[keep]
    if jitter_s > 0 and dep.size:
        dep = dep + rng.normal(0.0, jitter_s, dep.size)
        order = np.argsort(dep, kind="stable")
        dep, hidx = dep[order], hidx[order]
    return dep, hidx


def simulate(
    net: SimNetwork,
    freq: np.ndarray,
    draw: DemandDraw,
    seed: int = 0,
    travel_time_cv: float = 0.10,
    dwell_per_pax_s: float = 2.5,
    dispatch_jitter_s: float = 30.0,
    cancel_prob: float = 0.0,
    run_time_factor: np.ndarray | None = None,
) -> dict:
    """Simulate one day. ``freq`` [R,T] may be fractional (effective frequency after disruptions).

    ``run_time_factor`` [R,T] multiplies scheduled running times (congestion / detour scenarios).
    """
    inst = net.inst
    rng = np.random.default_rng(seed)
    hist = np.zeros(N_BINS, dtype=np.int64)
    tot = dict(
        wait=0.0,
        boarded=0,
        denied=0,
        pax_sec=0.0,
        over_sec=0.0,
        over_pax_sec=0.0,
        cap_sec=0.0,
        left_cap=0,
        left_end=0,
        trips=0,
    )
    max_load_factor = 0.0
    route_wait = np.zeros(len(inst.routes))
    route_board = np.zeros(len(inst.routes))
    route_left = np.zeros(len(inst.routes))
    for di, dm in enumerate(net.directions):
        r = dm.route_idx
        dep, hidx = _departures(
            np.where(inst.active[r], freq[r], 0.0), inst.hours, rng, dispatch_jitter_s, cancel_prob
        )
        if dep.size == 0:
            left = int(draw.offsets[di][-1])
            tot["left_end"] += left
            route_left[r] += left
            continue
        seg = np.diff(dm.cum_s, axis=1)
        if run_time_factor is not None:
            seg = seg * run_time_factor[r][:, None]
        # trip-level factor: lognormal with unit mean; segment-level noise handled inside the kernel
        sig = np.sqrt(np.log(1.0 + travel_time_cv**2)) if travel_time_cv > 0 else 0.0
        factor = np.exp(rng.normal(-0.5 * sig**2, sig, dep.size)) if sig > 0 else np.ones(dep.size)
        out = _run_direction(
            dep,
            hidx,
            factor,
            np.ascontiguousarray(seg),
            travel_time_cv,
            draw.offsets[di],
            draw.arr[di],
            np.ascontiguousarray(dm.alight_prob),
            int(net.crush[r]),
            float(inst.capacity[r]),
            dwell_per_pax_s,
            hist,
            int(rng.integers(0, 2**31 - 1)),
        )
        w, b, dn, ps, osec, ops, cs, ml, lc, le = out
        tot["wait"] += w
        tot["boarded"] += b
        tot["denied"] += dn
        tot["pax_sec"] += ps
        tot["over_sec"] += osec
        tot["over_pax_sec"] += ops
        tot["cap_sec"] += cs
        tot["left_cap"] += lc
        tot["left_end"] += le
        tot["trips"] += dep.size
        route_wait[r] += w
        route_board[r] += b
        route_left[r] += lc + le
        max_load_factor = max(max_load_factor, ml / inst.capacity[r])
    cdf = np.cumsum(hist)
    n = max(int(cdf[-1]), 1)

    def pct(q: float) -> float:
        return float((np.searchsorted(cdf, q * n) + 0.5) * BIN_S / 60.0)

    gen = max(draw.generated, 1)
    return {
        "passengers": int(draw.generated),
        "boarded": int(tot["boarded"]),
        "mean_wait_min": tot["wait"] / max(tot["boarded"], 1) / 60.0,
        "p50_wait_min": pct(0.5),
        "p90_wait_min": pct(0.9),
        "p95_wait_min": pct(0.95),
        "share_wait_le_10min": float(cdf[int(600 / BIN_S) - 1] / gen),
        "unmet_capacity": int(tot["left_cap"]),
        "unmet_end_of_service": int(tot["left_end"]),
        "unmet_share": (tot["left_cap"] + tot["left_end"]) / gen,
        "denied_boarding_events": int(tot["denied"]),
        "denied_per_1000_pax": 1000.0 * tot["denied"] / gen,
        "overcrowded_pax_min": tot["over_sec"] / 60.0,
        "share_pax_time_overcrowded": tot["over_pax_sec"] / max(tot["pax_sec"], 1e-9),
        "load_utilization": tot["pax_sec"] / max(tot["cap_sec"], 1e-9),
        "max_load_factor": float(max_load_factor),
        "trips": int(tot["trips"]),
        "service_coverage": tot["boarded"] / gen,
        "route_mean_wait_min": (route_wait / np.maximum(route_board, 1) / 60.0).tolist(),
        "route_unmet": route_left.tolist(),
        "wait_hist": hist,
    }
