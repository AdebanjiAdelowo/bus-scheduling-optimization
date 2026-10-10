import os

os.environ["BUSOPT_SYNTHETIC"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from busopt.api.app import app  # noqa: E402

client = TestClient(app)


def test_health_and_tools():
    h = client.get("/health").json()
    assert h["status"] == "ok" and "SYNTHETIC" in h["demand_source"]
    names = [t["name"] for t in client.get("/tools").json()]
    assert "optimize_schedule" in names and "compare_fleet_change" in names


def test_tool_endpoint_validates():
    ok = client.post("/tools/compare_fleet_change", json={"fleet_change_pct": -10})
    assert ok.status_code == 200 and ok.json()["result"]["new_fleet"] < ok.json()["result"]["baseline_fleet"]
    assert client.post("/tools/compare_fleet_change", json={"fleet_change_pct": -99}).status_code == 422
    assert (
        client.post(
            "/tools/frequency_effect", json={"frequency_change_pct": 10, "route_id": "NOPE"}
        ).status_code
        == 400
    )
    assert client.post("/tools/apply_schedule", json={}).status_code == 404


def test_agent_router_endpoint():
    r = client.post(
        "/agent/query", json={"question": "Which constraints are binding in the current solution?"}
    ).json()
    assert r["agent"] == "router" and r["calls"][0]["tool"] == "binding_constraints"
    r = client.post(
        "/agent/query", json={"question": "Apply the optimized schedule to the live timetable now."}
    ).json()
    assert r["calls"] == []


def test_results_whitelist():
    assert client.get("/results/../../pyproject").status_code in (404, 422)
    assert client.get("/results/not_a_table").status_code == 404
