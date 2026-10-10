"""Non-exact policies: the existing schedule, rule-based allocations and a greedy heuristic.

All rule-based policies are scaled to use the same fleet and vehicle-hour budget as the instance,
so no baseline is handicapped by a smaller resource envelope.
"""

from __future__ import annotations

import heapq
import time

import numpy as np

from busopt.opt.milp import Solution, cost_tensor
from busopt.opt.problem import Instance, buses_required, evaluate


def _wrap(inst: Instance, freq: np.ndarray, method: str, t0: float, feasible: bool = True) -> Solution:
    ev = evaluate(inst, freq)
    ok = feasible and ev["feasible"]
    return Solution(
        freq=freq.astype(int),
        status="HEURISTIC" if ok else "INFEASIBLE",
        objective=ev["objective"] if ok else None,
        bound=None,
        gap=None,
        runtime_s=time.perf_counter() - t0,
        solver="none",
        solver_version="-",
        n_vars=0,
        n_cons=0,
        method=method,
    )


def _feasible(inst: Instance, x: np.ndarray) -> bool:
    n = np.where(inst.active, buses_required(x, inst.cycle_min), 0.0)
    return bool(n.sum(axis=0).max() <= inst.fleet + 1e-9 and n.sum() <= inst.budget_vh + 1e-6)


def existing_schedule(inst: Instance) -> Solution:
    """The GTFS timetable projected onto the decision space (symmetric integer buses/hour)."""
    t0 = time.perf_counter()
    return _wrap(inst, inst.gtfs_freq.copy(), "existing_gtfs", t0)


def scale_weights(inst: Instance, w: np.ndarray, method: str) -> Solution:
    """Largest alpha such that x = clip(floor(alpha * w), k_min, k_max) fits fleet and budget."""
    t0 = time.perf_counter()

    def plan(alpha: float) -> np.ndarray:
        return np.where(inst.active, np.clip(np.floor(alpha * w + 1e-9), inst.freq_min, inst.freq_max), 0)

    if not _feasible(inst, plan(0.0)):
        return _wrap(inst, plan(0.0), method, t0, feasible=False)
    lo, hi = 0.0, 1.0
    while _feasible(inst, plan(hi)) and hi < 1e9:
        if (plan(hi)[inst.active] >= inst.freq_max).all():
            break
        lo, hi = hi, hi * 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _feasible(inst, plan(mid)):
            lo = mid
        else:
            hi = mid
    x = plan(lo)
    # top-up: spend what the integer rounding left over, in the order the rule itself would
    order = sorted(
        ((w[r, t] / max(x[r, t], 1), r, t) for r, t in zip(*np.nonzero(inst.active), strict=True)),
        reverse=True,
    )
    progress = True
    while progress:
        progress = False
        for _, r, t in order:
            if x[r, t] < inst.freq_max:
                x[r, t] += 1
                if _feasible(inst, x):
                    progress = True
                else:
                    x[r, t] -= 1
        order = sorted(((w[r, t] / max(x[r, t], 1), r, t) for _, r, t in order), reverse=True)
    return _wrap(inst, x, method, t0)


def flat_frequency(inst: Instance) -> Solution:
    """Fixed-frequency schedule: one headway per route all day (the existing day-average level)."""
    act = np.where(inst.active, inst.gtfs_freq, np.nan)
    w = np.where(inst.active, np.nanmean(act, axis=1, keepdims=True), 0.0)
    return scale_weights(inst, w, "fixed_frequency")


def proportional(inst: Instance) -> Solution:
    """Load-proportional rule (equal load factor): frequency proportional to peak load / capacity."""
    w = inst.peak_coef * inst.demand / inst.capacity[:, None]
    return scale_weights(inst, w, "proportional_load")


def sqrt_rule(inst: Instance) -> Solution:
    """Square-root rule: x proportional to sqrt(d / tau), the continuous optimum under one budget."""
    w = np.sqrt(np.maximum(inst.demand, 0.0) / inst.cycle_min)
    return scale_weights(inst, w, "sqrt_rule")


def greedy(inst: Instance, scenarios: np.ndarray | None = None) -> Solution:
    """Start at minimum frequency; repeatedly add the bus/hour with the best cost drop per vehicle-hour."""
    t0 = time.perf_counter()
    ks, c, n = cost_tensor(inst, scenarios)
    R, T = inst.shape
    j = np.zeros((R, T), dtype=int)  # index into ks
    fleet_used = np.where(inst.active, n[:, :, 0], 0.0).sum(axis=0)
    budget_used = float(fleet_used.sum())
    x0 = np.where(inst.active, ks[0], 0)
    if fleet_used.max() > inst.fleet + 1e-9 or budget_used > inst.budget_vh + 1e-6:
        return _wrap(inst, x0, "greedy", t0, feasible=False)

    def key(r: int, t: int) -> float:
        dn = n[r, t, j[r, t] + 1] - n[r, t, j[r, t]]
        gain = c[r, t, j[r, t]] - c[r, t, j[r, t] + 1]
        return -(gain / dn if dn > 0 else gain * 1e6)

    heap = [(key(r, t), r, t) for r in range(R) for t in range(T) if inst.active[r, t] and len(ks) > 1]
    heapq.heapify(heap)
    while heap:
        k, r, t = heapq.heappop(heap)
        if k >= 0:  # no further improvement possible from this cell
            continue
        dn = n[r, t, j[r, t] + 1] - n[r, t, j[r, t]]
        if fleet_used[t] + dn > inst.fleet + 1e-9 or budget_used + dn > inst.budget_vh + 1e-6:
            continue  # this cell is blocked for good: resources only shrink
        j[r, t] += 1
        fleet_used[t] += dn
        budget_used += dn
        if j[r, t] + 1 < len(ks):
            heapq.heappush(heap, (key(r, t), r, t))
    return _wrap(inst, np.where(inst.active, ks[j], 0), "greedy", t0)


def local_search(
    inst: Instance, freq: np.ndarray, scenarios: np.ndarray | None = None, max_passes: int = 20
) -> Solution:
    """Pairwise exchange improvement: move one bus/hour from one cell to another when it lowers cost."""
    t0 = time.perf_counter()
    ks, c, n = cost_tensor(inst, scenarios)
    j = np.where(inst.active, freq - inst.freq_min, 0).astype(int)
    R, T = inst.shape
    cells = [(r, t) for r in range(R) for t in range(T) if inst.active[r, t]]
    idx = np.array(cells)
    for _ in range(max_passes):
        jj = j[idx[:, 0], idx[:, 1]]
        up_ok, dn_ok = jj + 1 < len(ks), jj > 0
        up_gain = np.where(
            up_ok,
            c[idx[:, 0], idx[:, 1], jj] - c[idx[:, 0], idx[:, 1], np.minimum(jj + 1, len(ks) - 1)],
            -np.inf,
        )
        dn_loss = np.where(
            dn_ok, c[idx[:, 0], idx[:, 1], np.maximum(jj - 1, 0)] - c[idx[:, 0], idx[:, 1], jj], np.inf
        )
        improved = False
        for a in np.argsort(-up_gain)[:200]:
            if not up_ok[a]:
                continue
            ra, ta = idx[a]
            for b in np.argsort(dn_loss)[:200]:
                if a == b or not dn_ok[b] or up_gain[a] - dn_loss[b] <= 1e-9:
                    continue
                rb, tb = idx[b]
                trial = j.copy()
                trial[ra, ta] += 1
                trial[rb, tb] -= 1
                x = np.where(inst.active, ks[trial], 0)
                if _feasible(inst, x):
                    j, improved = trial, True
                    break
            if improved:
                break
        if not improved:
            break
    return _wrap(inst, np.where(inst.active, ks[j], 0), "greedy_local_search", t0)
