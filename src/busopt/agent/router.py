"""Deterministic intent router: keyword rules pick a tool, regexes extract its arguments.

It is the baseline the LLM agent is compared against. Its answers are templated directly from tool
output, so every number it states is a solver or simulator output by construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from busopt.agent.tools import PlanningSession, call_tool

OUT_OF_SCOPE = "out_of_scope"

# Ordered: the first rule whose pattern matches wins.
RULES: list[tuple[str, str]] = [
    (
        OUT_OF_SCOPE,
        r"\b(apply|deploy|push|publish|implement|dispatch|send|activate|cancel|reroute)\b.*\b(live|now|production|real|drivers?|depot|operators?|timetable to)\b",
    ),
    (
        "binding_constraints",
        r"\b(binding|bottleneck|shadow price|slack|tight constraint|limiting constraint|which constraints?)\b",
    ),
    ("robustness_cost", r"\b(robust|robustness|stochastic|uncertain|uncertainty|hedg\w*|scenario-based)\b"),
    (
        "compare_fleet_change",
        r"\b(buses|bus fleet|fleet|vehicles)\b.*\b(reduc\w*|cut|fewer|less|lose|lost|increase\w*|more|add\w*|extra|by \d)|\b(reduc\w*|cut|fewer|lose|add\w*|more|extra)\b.*\b(buses|fleet|vehicles)\b",
    ),
    ("frequency_effect", r"\b(frequenc\w*|headways?|buses per hour|more often|run more)\b"),
    (
        "crowding_ranking",
        r"\b(crowd\w*|overcrowd\w*|load factor|fullest|packed|busiest buses|over capacity)\b",
    ),
    ("simulate_plan", r"\b(simulat\w*|percentile|p9[05]|95th|90th|denied boarding|left behind|unmet)\b"),
    (
        "optimize_schedule",
        r"\b(optimi[sz]\w*|best schedule|optimal|re-?plan|improve the schedule|minimi[sz]e wait\w*)\b",
    ),
    (
        "demand_forecast",
        r"\b(demand|ridership|boardings|passengers|forecast\w*|busiest routes?|highest ridership)\b",
    ),
    ("network_summary", r"\b(how many routes|network|summary|overview|fleet size|how many buses|budget)\b"),
]

_PCT = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent|per cent)")
_ROUTE = re.compile(r"\b(M\d{1,3}[A-D]?\+?|T\d{1,2})(?![\w+])|\b(M\d{1,3}[A-D]?) ?(?:SBS|select bus)", re.I)
_TOPK = re.compile(r"\btop (\d+)\b|\b(\d+) (?:most|worst|busiest)\b")
_SCALE = re.compile(
    r"demand (?:is |were |was |grows? |increases? |rises? )?(?:by |to |x|×)?\s*(\d+(?:\.\d+)?)\s*(x|×|times|%|percent)?",
    re.I,
)
_DECREASE = re.compile(
    r"\b(reduc\w*|cut\w*|fewer|less|lose|lost|decreas\w*|lower\w*|drop\w*|remov\w*|halv\w*)\b", re.I
)


@dataclass
class RoutedCall:
    tool: str
    args: dict


def _daytype(q: str) -> str | None:
    ql = q.lower()
    if "saturday" in ql:
        return "saturday"
    if "sunday" in ql:
        return "sunday"
    if "weekend" in ql:
        return "saturday"
    if (
        "weekday" in ql
        or "monday" in ql
        or "tuesday" in ql
        or "wednesday" in ql
        or "thursday" in ql
        or "friday" in ql
    ):
        return "weekday"
    return None


def _route(q: str, session: PlanningSession) -> str | None:
    known = {r.upper(): r for r in session.instance("weekday").routes}
    for m in _ROUTE.finditer(q):
        token = (m.group(1) or m.group(2)).upper()
        if m.group(2) and token + "+" in known:
            return known[token + "+"]
        if token in known:
            return known[token]
        return token  # unknown route: passed through so validation reports it instead of silently dropping it
    return None


def route(question: str, session: PlanningSession) -> RoutedCall | None:
    q = question.lower()
    tool = next((name for name, pat in RULES if re.search(pat, q)), None)
    if tool is None or tool == OUT_OF_SCOPE:
        return None
    args: dict = {}
    if (dt := _daytype(question)) is not None:
        args["daytype"] = dt
    pct = _PCT.search(q)
    if tool == "compare_fleet_change":
        if pct is None:
            return RoutedCall(tool, args)  # missing required argument -> validation error, not a guess
        v = abs(float(pct.group(1)))
        args["fleet_change_pct"] = -v if _DECREASE.search(q) else v
    elif tool == "frequency_effect":
        v = abs(float(pct.group(1))) if pct else (100.0 if re.search(r"\bdoubl\w*", q) else None)
        if v is not None:
            args["frequency_change_pct"] = -v if _DECREASE.search(q) else v
        if (r := _route(question, session)) is not None:
            args["route_id"] = r
    elif tool in ("crowding_ranking", "demand_forecast"):
        if (m := _TOPK.search(q)) is not None:
            args["top_k"] = int(m.group(1) or m.group(2))
        if tool == "demand_forecast" and (r := _route(question, session)) is not None:
            args["route_id"] = r
        if tool == "crowding_ranking" and re.search(r"\b(optimi[sz]ed|after optimi[sz]ation)\b", q):
            args["plan"] = "optimized"
    elif tool in ("binding_constraints", "simulate_plan"):
        if re.search(r"\b(existing|current timetable|today'?s schedule|gtfs)\b", q):
            args["plan"] = "existing"
        elif re.search(r"\b(optimi[sz]ed|optimal)\b", q):
            args["plan"] = "optimized"
    if tool in ("crowding_ranking", "optimize_schedule", "robustness_cost", "simulate_plan") and (
        m := _SCALE.search(q)
    ):
        val, unit = float(m.group(1)), (m.group(2) or "").lower()
        args["demand_scale"] = 1 + val / 100 if unit in ("%", "percent") else val
    return RoutedCall(tool, args)


def _fmt(result: dict) -> str:
    """Template an answer from tool output only."""
    t, r = result["tool"], result.get("result", {})
    if not result["ok"]:
        return f"I could not run {t}: {result['error']}"
    src = f" (Demand basis: {r.get('demand_source', 'n/a')}.)"
    if t == "crowding_ranking":
        rows = "; ".join(
            f"{x['route_id']} load factor {x['peak_load_factor']} at {x['hour']}:00" for x in r["ranking"]
        )
        return (
            f"Highest expected crowding on the {r['plan']} {r['daytype']} schedule: {rows}. {r['routes_over_standard']} routes exceed the loading standard."
            + src
        )
    if t == "compare_fleet_change":
        if "change" not in r:
            return (
                f"A {r['fleet_change_pct']}% fleet change ({r['baseline_fleet']} to {r['new_fleet']} buses) is infeasible: {r['message']}"
                + src
            )
        return (
            f"Changing the fleet by {r['fleet_change_pct']}% ({r['baseline_fleet']} to {r['new_fleet']} buses) moves the optimised mean wait from "
            f"{r['baseline']['mean_wait_min']} to {r['new']['mean_wait_min']} min ({r['change']['mean_wait_pct']}%). Solver status: {r['new']['status']}."
            + src
        )
    if t == "frequency_effect":
        a, v = r["affected_mean_wait_min"], r["vehicle_hours"]
        return (
            f"A {r['frequency_change_pct']}% frequency change on {r['route_id']} moves mean wait for affected passengers from {a['before']} to {a['after']} min "
            f"and vehicle-hours from {v['before']} to {v['after']}. Fits the current fleet: {r['fits_current_fleet']}."
            + src
        )
    if t == "robustness_cost":
        return (
            f"The {r['quantile']}-quantile robust plan raises nominal mean wait by {r['cost_of_robustness_wait_min']} min versus the deterministic optimum and "
            f"lowers expected passengers above the loading standard by {r['benefit_expected_passengers_over_standard']} ({r['n_scenarios']} scenarios, {r['scenario_source']})."
            + src
        )
    if t == "binding_constraints":
        return (
            f"On the {r['plan']} plan the budget is {'binding' if r['budget']['binding'] else 'not binding'} (slack {r['budget']['slack']} vehicle-hours); "
            f"fleet binds at hours {r['fleet_binding_hours']}; {r['cells_at_min_frequency']} of {r['active_cells']} route-hours sit at the minimum frequency; "
            f"{r['cells_over_loading_standard']} exceed the loading standard." + src
        )
    if t == "optimize_schedule":
        if "optimized" not in r:
            return r["message"] + src
        return (
            f"Optimised mean wait {r['optimized']['mean_wait_min']} min versus {r['existing_schedule_same_demand']['mean_wait_min']} min for the existing schedule, "
            f"using {r['optimized']['vehicle_hours']} vehicle-hours. Status {r['status']}, gap {r['mip_gap']}, verified {r['independently_verified']}."
            + src
        )
    if t == "simulate_plan":
        s = r["simulation"]
        return (
            f"Simulated {r['plan']} plan: mean wait {s['mean_wait_min']} min, 90th percentile {s['p90_wait_min']}, 95th percentile {s['p95_wait_min']}, "
            f"unmet share {s['unmet_share']}." + src
        )
    if t == "demand_forecast":
        if "top_routes" in r:
            return (
                "Highest expected boardings: "
                + "; ".join(f"{x['route_id']} {x['daily_boardings']}" for x in r["top_routes"])
                + "."
                + src
            )
        return (
            f"{r['route_id']} expects {r['daily_total']} boardings per {r['daytype']}, peaking at {r['peak_hour']}:00."
            + src
        )
    if t == "network_summary":
        return (
            f"{r['routes']} routes, fleet {r['fleet']}, {r['vehicle_hour_budget']} vehicle-hours, {r['daily_boardings']} daily boardings."
            + src
        )
    return str(r)


def answer(question: str, session: PlanningSession) -> dict:
    """Route, execute and template. Same return shape as the LLM agent."""
    call = route(question, session)
    if call is None:
        return {
            "answer": "This request is outside what the planning tools can do. They only run read-only what-if analyses; "
            "they cannot change a live timetable or operations.",
            "calls": [],
            "agent": "router",
        }
    result = call_tool(session, call.tool, call.args)
    return {"answer": _fmt(result), "calls": [result], "agent": "router"}
