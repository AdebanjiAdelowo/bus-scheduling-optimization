"""Static figures for the documentation. Run: ``python -m busopt.viz.plots``.

Every figure is drawn from files in ``results/`` or ``data/processed/``; a figure whose inputs are
missing is skipped and reported, never drawn from placeholder numbers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from busopt.config import load_config, resolve  # noqa: E402

# Categorical slots in fixed order (validated for colour-vision-deficiency separation on the light surface).
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f0efec", "#cde2fb", "#86b6ef", "#2a78d6", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f5b59a", "#f0efec", "#86b6ef", "#2a78d6"])

POLICY_LABEL = {
    "existing_gtfs": "Existing schedule",
    "fixed_frequency": "Fixed frequency",
    "proportional_load_hist": "Load-proportional rule",
    "sqrt_rule_hist": "Square-root rule",
    "greedy_hist": "Greedy",
    "milp_historical": "MILP, historical demand",
    "sqrt_rule_forecast": "Square-root rule, forecast",
    "milp_forecast": "MILP, day-ahead forecast",
    "milp_forecast_h7": "MILP, 7-day-ahead forecast",
    "milp_saa": "Stochastic MILP (SAA)",
    "milp_quantile_90": "Robust MILP (90% quantile)",
    "milp_oracle": "MILP, perfect foresight",
    "milp_saa_cvar": "Stochastic MILP + CVaR",
    "milp_slack_10": "MILP, 10% running-time slack",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def _save(fig, out: Path, name: str) -> str:
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=160)
    plt.close(fig)
    return name


def network_map(t, out):
    pat, info = t["patterns"], t["route_info"].set_index("route_id")
    fig, ax = plt.subplots(figsize=(5.2, 8))
    dem = info.mean_daily_boardings_train
    for (r, d), p in pat.groupby(["route_id", "direction_id"]):
        if d != 0:
            continue
        v = float(np.clip(dem.get(r, 0) / dem.max(), 0.05, 1))
        ax.plot(
            p.stop_lon,
            p.stop_lat,
            color=SEQ(0.35 + 0.65 * v),
            lw=0.8 + 2.4 * v,
            solid_capstyle="round",
            zorder=2 + v,
        )
    top = dem.sort_values(ascending=False).index[:4]
    for r in top:
        p = pat[(pat.route_id == r) & (pat.direction_id == 0)]
        mid = p.iloc[len(p) // 2]
        ax.annotate(
            r, (mid.stop_lon, mid.stop_lat), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK2
        )
    ax.set_aspect(1 / np.cos(np.radians(40.78)))
    ax.set(title="Manhattan bus network (GTFS dominant patterns)", xlabel="Longitude", ylabel="Latitude")
    ax.text(
        0.02,
        0.02,
        "Line weight and shade: mean daily fare-paying boardings\n(training window). Four busiest routes labelled.",
        transform=ax.transAxes,
        fontsize=8,
        color=INK2,
    )
    return _save(fig, out, "network_map")


def demand_figures(t, out):
    rh = t["route_hour"]
    wk = rh[(rh.daytype == "weekday") & rh.hour.between(5, 23)].pivot(
        index="route_id", columns="hour", values="mean_demand"
    )
    wk = wk.loc[wk.sum(axis=1).sort_values(ascending=False).index[:31]]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(wk.to_numpy(), aspect="auto", cmap=SEQ)
    ax.set_xticks(range(len(wk.columns)), wk.columns)
    ax.set_yticks(range(len(wk.index)), wk.index, fontsize=7)
    ax.grid(False)
    ax.set(
        title="Mean weekday fare-paying boardings per hour, by route (observed, training window)",
        xlabel="Hour of day",
    )
    fig.colorbar(im, ax=ax, shrink=0.6, label="Boardings per hour")
    _save(fig, out, "demand_heatmap")

    prof = rh[rh.hour.between(5, 23)].groupby(["daytype", "hour"]).mean_demand.sum().unstack(0)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, dt in enumerate(["weekday", "saturday", "sunday"]):
        ax.plot(prof.index, prof[dt], color=C[i], label=dt.capitalize())
        ax.annotate(
            dt.capitalize(),
            (prof.index[-1], prof[dt].iloc[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK2,
            va="center",
        )
    ax.set(
        title="Network demand by hour and day type (observed fare-paying boardings)",
        xlabel="Hour of day",
        ylabel="Boardings per hour",
        xlim=(5, 25.5),
    )
    ax.legend(loc="upper left", ncols=3)
    return _save(fig, out, "demand_profile")


def forecast_figure(res, out):
    m = pd.read_csv(res / "forecast_metrics.csv")
    m = (
        m[m.slice == "service_hours"]
        .pivot(index="model", columns="horizon_days", values="mae")
        .sort_values(1, ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(m))
    for i, h in enumerate(m.columns):
        ax.barh(y + (i - 1) * 0.27, m[h], height=0.24, color=C[i], label=f"{h}-day horizon")
    for yy, v in zip(y, m[1], strict=True):
        ax.text(v + 0.4, yy - 0.27, f"{v:.1f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [s.replace("_", " ") for s in m.index])
    ax.grid(axis="y", visible=False)
    ax.set(
        title="Forecast error on the held-out test period (service hours)",
        xlabel="Mean absolute error, boardings per route-hour",
    )
    ax.legend(loc="lower right")
    _save(fig, out, "forecast_mae")
    bh = pd.read_csv(res / "forecast_error_by_hour.csv")
    bh = bh[(bh.horizon == 1) & bh.hour.between(5, 23)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(bh.hour, bh.historical_average_mae, color=C[1], label="Historical average")
    ax.plot(bh.hour, bh.gbt_mae, color=C[0], label="Gradient-boosted trees")
    ax.set(
        title="Day-ahead forecast error by hour of day (test period)",
        xlabel="Hour of day",
        ylabel="MAE, boardings per route-hour",
    )
    ax.legend(loc="upper right")
    return _save(fig, out, "forecast_error_by_hour")


def schedule_figures(cfg, t, out):
    from busopt.opt.milp import solve_frequency_milp
    from busopt.opt.problem import build_instance, buses_required

    inst = build_instance(t, cfg, "weekday")
    sol = solve_frequency_milp(inst, time_limit_s=cfg["optimization"]["time_limit_s"])
    diff = np.where(inst.active, sol.freq - inst.gtfs_freq, np.nan)
    lim = np.nanmax(np.abs(diff))
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(diff, aspect="auto", cmap=DIV, vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(inst.hours)), inst.hours)
    ax.set_yticks(range(len(inst.routes)), inst.routes, fontsize=7)
    ax.grid(False)
    ax.set(
        title="Optimised minus existing frequency, weekday (buses per hour per direction)",
        xlabel="Hour of day",
    )
    fig.colorbar(im, ax=ax, shrink=0.6, label="Blue: more service than today. Orange: less.")
    _save(fig, out, "schedule_difference")

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, (name, f) in enumerate(
        (("Existing schedule", inst.gtfs_freq), ("MILP, historical demand", sol.freq))
    ):
        n = np.where(inst.active, buses_required(f, inst.cycle_min), 0).sum(axis=0)
        ax.plot(inst.hours, n, color=C[i], label=name, marker="o", ms=3.5)
    ax.axhline(inst.fleet, color=MUTED, lw=1, ls="--")
    ax.text(inst.hours[0], inst.fleet + 6, f"Fleet limit: {inst.fleet} buses", fontsize=8, color=INK2)
    ax.set(
        title="Fleet in use by hour, weekday",
        xlabel="Hour of day",
        ylabel="Buses required",
        ylim=(0, inst.fleet * 1.1),
    )
    ax.legend(loc="lower center", ncols=2)
    return _save(fig, out, "fleet_by_hour")


def policy_figures(res, out):
    s = pd.read_csv(res / "policy_comparison_summary.csv")
    s = s[(s.metric == "sim_mean_wait_min") & (s.policy != "existing_gtfs")].sort_values("pct_vs_baseline")
    base = pd.read_csv(res / "policy_comparison_days.csv")
    base_wait = base[base.policy == "existing_gtfs"].sim_mean_wait_min.mean()
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    y = np.arange(len(s))
    pct, lo, hi = s.pct_vs_baseline, 100 * s.ci95_lo / base_wait, 100 * s.ci95_hi / base_wait
    ax.hlines(y, lo, hi, color=C[0], lw=2)
    ax.plot(pct, y, "o", color=C[0], ms=7, mec=SURFACE, mew=1.5)
    for yy, v in zip(y, pct, strict=True):
        ax.text(v, yy + 0.32, f"{v:+.1f}%", ha="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, [POLICY_LABEL.get(p, p) for p in s.policy])
    ax.grid(axis="y", visible=False)
    ax.set(
        title=f"Simulated mean wait relative to the existing schedule ({int(s.n_days.max())} held-out days)",
        xlabel="Change in mean passenger wait, % (bars: bootstrap 95% interval over days)",
    )
    _save(fig, out, "policy_comparison")

    h = np.load(res / "policy_comparison_wait_hist.npz")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = (np.arange(len(h["existing_gtfs"])) + 1) * 10 / 60
    for i, p in enumerate(["existing_gtfs", "sqrt_rule_hist", "milp_historical", "milp_forecast"]):
        cdf = np.cumsum(h[p]) / h[p].sum()
        ax.plot(x, 100 * cdf, color=C[i], label=POLICY_LABEL[p], lw=1.8)
    ax.set(
        title="Waiting-time distribution, all held-out days (simulation)",
        xlabel="Wait, minutes",
        ylabel="Passengers served within wait, %",
        xlim=(0, 25),
        ylim=(0, 100),
    )
    ax.legend(loc="lower right")
    return _save(fig, out, "wait_distribution")


def frontier_figure(res, out):
    f = pd.read_csv(res / "resource_frontier.csv")
    b = f[(f.experiment == "budget_and_fleet") & f.feasible.astype(bool)]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (pol, lab) in enumerate(
        (
            ("milp", "MILP (exact)"),
            ("sqrt_rule", "Square-root rule"),
            ("proportional_load", "Load-proportional rule"),
        )
    ):
        d = b[b.policy == pol].sort_values("vehicle_hours")
        ax.plot(d.vehicle_hours, d.mean_wait_min, color=C[i], marker="o", ms=4, label=lab)
    ref = f[f.experiment == "reference"].iloc[0]
    ax.plot(ref.vehicle_hours, ref.mean_wait_min, "s", color=INK, ms=7)
    ax.annotate(
        "Existing schedule",
        (ref.vehicle_hours, ref.mean_wait_min),
        xytext=(8, 6),
        textcoords="offset points",
        fontsize=8,
        color=INK2,
    )
    ax.set(
        title="Cost-service trade-off, weekday (planning model, historical demand)",
        xlabel="Vehicle-hours per day",
        ylabel="Mean passenger wait, minutes",
    )
    ax.legend(loc="upper right")
    return _save(fig, out, "cost_service_frontier")


def demand_scaling_figure(res, out):
    d = pd.read_csv(res / "demand_scaling_days.csv")
    g = (
        d.groupby(["demand_scale", "policy"])[["sim_overcrowded_pax_min", "sim_mean_wait_min"]]
        .mean()
        .reset_index()
    )
    pols = ["existing_gtfs", "milp_historical", "milp_forecast", "milp_saa", "milp_quantile_90"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, col, lab in zip(
        axes,
        ("sim_mean_wait_min", "sim_overcrowded_pax_min"),
        ("Mean wait, minutes", "Overcrowded passenger-minutes per day"),
        strict=True,
    ):
        for i, p in enumerate(pols):
            s = g[g.policy == p]
            ax.plot(s.demand_scale, s[col], color=C[i], marker="o", ms=3.5, label=POLICY_LABEL[p], lw=1.8)
        ax.set(xlabel="Demand multiplier (1 = observed fare-paying demand)", ylabel=lab)
    axes[0].set_title("Demand scaling: waits and overcrowding by policy (simulation)")
    axes[1].legend(loc="upper left", fontsize=8)
    return _save(fig, out, "demand_scaling")


def stress_figure(res, out):
    s = pd.read_csv(res / "stress_summary.csv")
    piv = s.pivot(index="policy", columns="scenario", values="sim_mean_wait_min")
    order = [
        c
        for c in [
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
        if c in piv.columns
    ]
    piv = piv[order]
    rel = 100 * (piv / piv.loc["existing_gtfs"] - 1)
    lim = float(np.nanmax(np.abs(rel.to_numpy())))
    fig, ax = plt.subplots(figsize=(10, 3.9))
    im = ax.imshow(-rel.to_numpy(), aspect="auto", cmap=DIV, vmin=-lim, vmax=lim)
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            ax.text(j, i, f"{rel.iat[i, j]:+.1f}", ha="center", va="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(order)), [o.replace("_", " ") for o in order], rotation=30, ha="right")
    ax.set_yticks(range(len(rel.index)), [POLICY_LABEL.get(p, p) for p in rel.index], fontsize=8)
    ax.grid(False)
    ax.set_title(
        "Simulated mean wait vs existing schedule under synthetic stress scenarios, % (blue: shorter waits)"
    )
    return _save(fig, out, "stress_heatmap")


def scalability_figure(res, out):
    d = pd.read_csv(res / "scalability.csv")
    d = d[d.method.isin(["highs", "scip", "cbc", "cpsat"]) & d.case.str.contains("routes|synthetic")]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, sv in enumerate(["highs", "scip", "cbc", "cpsat"]):
        s = d[d.method == sv].sort_values("n_vars")
        ax.plot(
            s.n_vars,
            s.runtime_s,
            color=C[i],
            marker="o",
            ms=4,
            label={"highs": "HiGHS", "scip": "SCIP", "cbc": "CBC", "cpsat": "CP-SAT"}[sv],
        )
    ax.set(
        xscale="log",
        yscale="log",
        title="Solver runtime against model size (time limit 60 s)",
        xlabel="Binary variables",
        ylabel="Wall-clock seconds",
    )
    ax.legend(loc="upper left")
    return _save(fig, out, "solver_scaling")


def main() -> None:
    _style()
    cfg = load_config()
    res, out = resolve("results"), resolve("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    from busopt.data.build import load_tables

    t = load_tables(cfg)
    jobs = [
        ("network", lambda: network_map(t, out)),
        ("demand", lambda: demand_figures(t, out)),
        ("forecast", lambda: forecast_figure(res, out)),
        ("schedule", lambda: schedule_figures(cfg, t, out)),
        ("policy", lambda: policy_figures(res, out)),
        ("frontier", lambda: frontier_figure(res, out)),
        ("demand_scaling", lambda: demand_scaling_figure(res, out)),
        ("stress", lambda: stress_figure(res, out)),
        ("scalability", lambda: scalability_figure(res, out)),
    ]
    for name, job in jobs:
        try:
            print("wrote", job())
        except FileNotFoundError as e:
            print(f"SKIPPED {name}: missing input {e.filename}")


if __name__ == "__main__":
    main()
