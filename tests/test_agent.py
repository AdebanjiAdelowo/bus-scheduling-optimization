import json
from types import SimpleNamespace

import pytest

from busopt.agent import router
from busopt.agent.eval import EVAL_SET, numeric_consistency, run_agent, score, summarise
from busopt.agent.llm import LLMAgent
from busopt.agent.tools import TOOLS, PlanningSession, anthropic_tool_definitions, call_tool


@pytest.fixture(scope="module")
def session(tables, cfg):
    return PlanningSession(tables, cfg, "SYNTHETIC toy network (not observed data)")


def test_every_tool_runs_with_defaults_or_minimal_args(session):
    minimal = {
        "compare_fleet_change": {"fleet_change_pct": -10},
        "frequency_effect": {"frequency_change_pct": 20},
    }
    for name in TOOLS:
        out = call_tool(session, name, minimal.get(name, {}))
        assert out["ok"], (name, out.get("error"))
        assert "demand_source" in out["result"] and "SYNTHETIC" in out["result"]["demand_source"]
        json.dumps(out["result"])  # serialisable


def test_argument_validation_rejects_bad_input(session):
    assert call_tool(session, "compare_fleet_change", {})["error_type"] == "invalid_arguments"
    assert (
        call_tool(session, "compare_fleet_change", {"fleet_change_pct": -95})["error_type"]
        == "invalid_arguments"
    )
    assert call_tool(session, "crowding_ranking", {"daytype": "holiday"})["error_type"] == "invalid_arguments"
    assert (
        call_tool(session, "crowding_ranking", {"top_k": 3, "surprise": 1})["error_type"]
        == "invalid_arguments"
    )
    assert (
        call_tool(session, "frequency_effect", {"frequency_change_pct": 10, "route_id": "M999"})["error_type"]
        == "tool_error"
    )
    assert call_tool(session, "set_live_timetable", {})["error_type"] == "unknown_tool"


def test_no_tool_has_side_effects_by_name():
    """The registry exposes analysis only: nothing that applies, publishes or dispatches."""
    banned = ("apply", "deploy", "publish", "dispatch", "write", "update", "delete", "send")
    assert not [n for n in TOOLS if any(b in n for b in banned)]
    assert all("Read-only" in d["description"] for d in anthropic_tool_definitions())


def test_fleet_reduction_cannot_improve_waits(session):
    out = call_tool(session, "compare_fleet_change", {"fleet_change_pct": -10})["result"]
    assert out["new"]["mean_wait_min"] >= out["baseline"]["mean_wait_min"] - 1e-9
    assert out["baseline"]["independently_verified"]


def test_infeasible_request_is_reported_not_invented(tables, cfg):
    tight = {
        **tables,
        "route_hour": tables["route_hour"].assign(gtfs_freq=2.0),
    }  # fleet exactly covers the minimum policy
    session = PlanningSession(tight, cfg, "SYNTHETIC toy network (not observed data)")
    out = call_tool(session, "compare_fleet_change", {"fleet_change_pct": -20})
    assert out["ok"] and out["result"]["new"]["feasible"] is False and "change" not in out["result"]


def test_router_canonical_questions(session):
    cases = {
        "Which routes experience the greatest expected crowding?": ("crowding_ranking", {}),
        "What happens if available buses are reduced by 10%?": (
            "compare_fleet_change",
            {"fleet_change_pct": -10.0},
        ),
        "How does increasing frequency by 20% affect expected waiting time?": (
            "frequency_effect",
            {"frequency_change_pct": 20.0},
        ),
        "What is the cost of a more robust schedule?": ("robustness_cost", {}),
        "Which constraints are binding in the current solution?": ("binding_constraints", {}),
    }
    for q, (tool, args) in cases.items():
        call = router.route(q, session)
        assert call.tool == tool and call.args == args, q
    assert router.route("Apply the optimized schedule to the live timetable now.", session) is None
    assert router.route("What is the subway fare?", session) is None


def test_router_answers_only_quote_tool_numbers(session):
    out = router.answer("What happens if available buses are reduced by 10%?", session)
    total, bad = numeric_consistency(
        out["answer"], "What happens if available buses are reduced by 10%?", out["calls"]
    )
    assert total > 0 and bad == 0


def test_numeric_consistency_flags_invented_numbers():
    calls = [{"ok": True, "result": {"mean_wait_min": 4.219, "share": 0.25}, "args": {}}]
    assert numeric_consistency("Mean wait is 4.22 min (25%).", "q", calls) == (2, 0)
    assert numeric_consistency("Mean wait is 3.9 min at the 95th percentile.", "q", calls) == (1, 1)


def test_scoring_and_summary(session):
    items = json.loads(EVAL_SET.read_text())
    df = run_agent(
        "router",
        session,
        [i for i in items if i["split"] in ("canonical", "out_of_scope") and "M15" not in i["question"]],
    )
    s = summarise(df, "router").set_index("split")
    assert s.loc["out_of_scope", "task_completion"] == 1.0
    assert s.loc["canonical", "tool_selection_accuracy"] == 1.0
    bad = score(items[0], {"answer": "Route X has load 9.99", "calls": []})
    assert not bad["task_completed"] and bad["unsupported_answer"]


class _FakeMessages:
    """Scripted stand-in for the Messages API: one tool call, then a final answer."""

    def __init__(self, script):
        self.script, self.requests = list(script), []

    def create(self, **kw):
        self.requests.append(kw)
        return self.script.pop(0)


def _resp(stop, *blocks):
    return SimpleNamespace(
        stop_reason=stop, content=list(blocks), usage=SimpleNamespace(input_tokens=10, output_tokens=5)
    )


def test_llm_agent_loop_validates_and_returns_tool_results(session):
    script = [
        _resp(
            "tool_use",
            SimpleNamespace(
                type="tool_use", id="t1", name="compare_fleet_change", input={"fleet_change_pct": -10}
            ),
            SimpleNamespace(
                type="tool_use", id="t2", name="compare_fleet_change", input={"fleet_change_pct": "lots"}
            ),
        ),
        _resp("end_turn", SimpleNamespace(type="text", text="Done.")),
    ]
    fake = _FakeMessages(script)
    client = SimpleNamespace(messages=fake, beta=SimpleNamespace(messages=fake))
    out = LLMAgent(session, client=client, use_fallbacks=False).answer("What if buses are reduced by 10%?")
    assert out["answer"] == "Done." and [c["ok"] for c in out["calls"]] == [True, False]
    results = fake.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"] and results[1]["is_error"] is True
    assert (
        "mean_wait_min" in results[0]["content"]
    )  # the model is handed solver output, not asked to imagine it
    assert fake.requests[0]["model"] == "claude-opus-5-5" and fake.requests[0]["output_config"] == {
        "effort": "medium"
    }
