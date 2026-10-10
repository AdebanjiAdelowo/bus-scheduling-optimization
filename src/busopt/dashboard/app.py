"""Streamlit dashboard. Run: ``streamlit run src/busopt/dashboard/app.py``.

Shows recorded experiment results and runs live what-if analyses through the same typed tools the
API and the agents use. Analysis only: nothing here changes a real schedule.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from busopt.agent import router
from busopt.agent.tools import call_tool, demo_session
from busopt.config import resolve

st.set_page_config(page_title="Bus scheduling optimisation", layout="wide")
RES, FIG = resolve("results"), resolve("docs/figures")


@st.cache_resource
def session():
    return demo_session(prefer_real=os.environ.get("BUSOPT_SYNTHETIC", "0") != "1")


def table(name: str) -> pd.DataFrame | None:
    p = RES / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def figure(name: str, caption: str) -> None:
    p = FIG / f"{name}.png"
    if p.exists():
        st.image(str(p), caption=caption)
    else:
        st.info(f"Figure '{name}' has not been generated. Run `busopt figures`.")


s = session()
st.title("Bus scheduling and route optimisation")
st.caption(
    f"Demand basis: {s.demand_source}. All analyses are what-if computations; nothing is applied to live operations."
)
tabs = st.tabs(
    [
        "Network and demand",
        "Forecasting",
        "Optimise",
        "Policy results",
        "Uncertainty",
        "Solver scaling",
        "Assistant",
    ]
)

with tabs[0]:
    summ = call_tool(s, "network_summary", {})["result"]
    c = st.columns(4)
    c[0].metric("Routes in instance", summ["routes"])
    c[1].metric("Weekday fleet", summ["fleet"])
    c[2].metric("Vehicle-hours per weekday", f"{summ['vehicle_hour_budget']:,.0f}")
    c[3].metric("Weekday boardings", f"{summ['daily_boardings']:,.0f}")
    a, b = st.columns([1, 1.6])
    with a:
        figure("network_map", "Route patterns from GTFS; weight shows observed boardings.")
    with b:
        figure("demand_profile", "Observed fare-paying boardings by hour.")
        figure("demand_heatmap", "Observed weekday boardings by route and hour.")

with tabs[1]:
    m = table("forecast_metrics")
    if m is None:
        st.info("Run `busopt forecast` to generate forecast metrics.")
    else:
        st.subheader("Test-period error, service hours")
        st.dataframe(
            m[m.slice == "service_hours"].pivot(index="model", columns="horizon_days", values="mae").round(2),
            width="stretch",
        )
        figure("forecast_mae", "Mean absolute error by model and horizon.")
        figure("forecast_error_by_hour", "Day-ahead error by hour of day.")

with tabs[2]:
    st.subheader("Live what-if: re-optimise frequencies")
    c1, c2, c3 = st.columns(3)
    daytype = c1.selectbox("Day type", ["weekday", "saturday", "sunday"])
    fleet = c2.slider("Fleet and budget change, %", -40, 40, 0, 5)
    scale = c3.slider("Demand multiplier", 0.5, 3.0, 1.0, 0.25)
    if st.button("Solve"):
        out = call_tool(
            s, "optimize_schedule", {"daytype": daytype, "fleet_change_pct": fleet, "demand_scale": scale}
        )
        if not out["ok"]:
            st.error(out["error"])
        elif "optimized" not in out["result"]:
            st.warning(out["result"]["message"])
        else:
            r = out["result"]
            k = st.columns(4)
            k[0].metric(
                "Optimised mean wait, min",
                r["optimized"]["mean_wait_min"],
                delta=round(
                    r["optimized"]["mean_wait_min"] - r["existing_schedule_same_demand"]["mean_wait_min"], 3
                ),
                delta_color="inverse",
            )
            k[1].metric("Vehicle-hours", r["optimized"]["vehicle_hours"])
            k[2].metric("Solver status", r["status"])
            k[3].metric("Independently verified", "yes" if r["independently_verified"] else "no")
            st.json(r, expanded=False)
        st.subheader("Binding constraints")
        st.json(call_tool(s, "binding_constraints", {"daytype": daytype})["result"], expanded=False)
    figure("schedule_difference", "Where the optimiser moves service relative to the existing timetable.")
    figure("fleet_by_hour", "Fleet in use by hour.")

with tabs[3]:
    summ_t = table("policy_comparison_summary")
    if summ_t is None:
        st.info("Run `busopt policies` to generate the policy comparison.")
    else:
        st.dataframe(summ_t[summ_t.metric == "sim_mean_wait_min"].round(3), width="stretch")
        figure("policy_comparison", "Paired difference to the existing schedule on held-out days.")
        figure("wait_distribution", "Simulated waiting-time distribution.")
        figure("cost_service_frontier", "Cost-service trade-off.")

with tabs[4]:
    st.caption("Stress scenarios are synthetic; demand totals are observed.")
    figure("stress_heatmap", "Mean wait relative to the existing schedule by scenario.")
    figure("demand_scaling", "Deterministic and uncertainty-aware plans as demand grows.")
    t = table("stress_summary")
    if t is not None:
        st.dataframe(t.round(3), width="stretch")

with tabs[5]:
    figure("solver_scaling", "Runtime against model size.")
    t = table("scalability")
    if t is not None:
        st.dataframe(t.round(4), width="stretch")

with tabs[6]:
    st.caption(
        "Deterministic router by default. The LLM agent needs Anthropic credentials and the 'agent' extra."
    )
    mode = st.radio("Agent", ["router", "llm"], horizontal=True)
    q = st.text_input("Question", "What happens if available buses are reduced by 10%?")
    if st.button("Ask"):
        try:
            if mode == "llm":
                from busopt.agent.llm import LLMAgent

                out = LLMAgent(s).answer(q)
            else:
                out = router.answer(q, s)
            st.write(out["answer"])
            for call in out["calls"]:
                with st.expander(f"Tool call: {call['tool']} ({'ok' if call['ok'] else 'error'})"):
                    st.json(call)
        except Exception as e:
            st.error(f"Agent unavailable: {type(e).__name__}")
