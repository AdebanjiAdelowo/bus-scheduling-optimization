"""FastAPI service exposing the typed planning tools and both agents.

The service is analysis-only: every endpoint computes a what-if result in memory. Nothing here can
alter a real timetable. Run: ``uvicorn busopt.api.app:app``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from busopt.agent import router
from busopt.agent.tools import TOOLS, PlanningSession, call_tool, demo_session
from busopt.config import resolve
from busopt.tracking import git_commit

app = FastAPI(
    title="busopt planning API",
    version="0.1.0",
    description="Forecast-informed bus frequency setting: read-only what-if analysis tools.",
)

RESULT_FILES = {
    "forecast_metrics",
    "policy_comparison_summary",
    "policy_comparison_vs_milp_historical",
    "resource_frontier",
    "stress_summary",
    "scalability",
    "agent_eval_summary",
}


@lru_cache(maxsize=1)
def get_session() -> PlanningSession:
    return demo_session(prefer_real=os.environ.get("BUSOPT_SYNTHETIC", "0") != "1")


class AgentQuery(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    agent: Literal["router", "llm"] = "router"


@app.get("/health")
def health() -> dict:
    s = get_session()
    return {
        "status": "ok",
        "git_commit": git_commit(),
        "demand_source": s.demand_source,
        "tools": sorted(TOOLS),
    }


@app.get("/tools")
def list_tools() -> list[dict]:
    return [
        {"name": t.name, "description": t.description, "arguments": t.args.model_json_schema()}
        for t in TOOLS.values()
    ]


@app.post("/tools/{name}")
def run_tool(name: str, args: dict | None = None) -> dict:
    if name not in TOOLS:
        raise HTTPException(404, f"unknown tool {name!r}")
    out = call_tool(get_session(), name, args or {})
    if not out["ok"]:
        raise HTTPException(422 if out["error_type"] == "invalid_arguments" else 400, out["error"])
    return out


@app.post("/agent/query")
def agent_query(q: AgentQuery) -> dict:
    if q.agent == "router":
        return router.answer(q.question, get_session())
    try:
        from busopt.agent.llm import LLMAgent

        return LLMAgent(get_session()).answer(q.question)
    except ImportError as e:
        raise HTTPException(503, "LLM agent unavailable: install the 'agent' extra") from e
    except Exception as e:  # missing credentials or an upstream API error
        raise HTTPException(503, f"LLM agent unavailable: {type(e).__name__}") from e


@app.get("/results/{name}")
def results(name: str) -> list[dict]:
    if name not in RESULT_FILES:
        raise HTTPException(404, f"unknown result table; available: {sorted(RESULT_FILES)}")
    path = resolve("results") / f"{name}.csv"
    if not path.exists():
        raise HTTPException(404, f"{name} has not been generated yet")
    return pd.read_csv(path).astype(object).where(lambda d: d.notna(), None).to_dict(orient="records")
