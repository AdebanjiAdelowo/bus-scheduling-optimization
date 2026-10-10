"""Agent evaluation (RQ7): deterministic router vs LLM tool selection on a labelled request set.

Run: ``python -m busopt.agent.eval --agent router`` (offline) or ``--agent llm`` (needs Anthropic
credentials). Without credentials the LLM arm is recorded as PENDING; no results are invented.

Splits: ``canonical`` (the phrasings the router rules were written for), ``paraphrase`` (written
afterwards and not used to tune the rules), ``missing_argument`` (a correct answer must not guess:
either ask, or surface the validation error) and ``out_of_scope`` (no tool should be called).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from busopt.agent import router
from busopt.agent.tools import PlanningSession, demo_session
from busopt.config import resolve

EVAL_SET = Path(__file__).with_name("eval_set.json")
# numbers, excluding ordinals ("95th percentile") and clock times ("8:00"), which are labels, not results
_NUM = re.compile(r"(?<![\w.:])-?\d+(?:,\d{3})*(?:\.\d+)?(?!\d|th\b|st\b|nd\b|rd\b|:\d)")


def _numbers(obj) -> list[float]:
    """Every number in a nested tool result."""
    if isinstance(obj, bool) or obj is None:
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        return [float(m.replace(",", "")) for m in _NUM.findall(obj)]
    if isinstance(obj, dict):
        return [n for k, v in obj.items() for n in (_numbers(k) + _numbers(v))]
    if isinstance(obj, (list, tuple)):
        return [n for v in obj for n in _numbers(v)]
    return []


def numeric_consistency(answer: str, question: str, calls: list[dict]) -> tuple[int, int]:
    """(numbers in the answer, numbers not traceable to a tool result or the question).

    A number is traceable if it equals a source number after rounding the source to the precision
    shown in the answer; sign and percent/fraction conversion (x100) are allowed.
    """
    source = _numbers(question) + [
        n
        for c in calls
        for n in _numbers(c.get("result", {})) + _numbers(c.get("args", {})) + _numbers(c.get("error", ""))
    ]
    total = bad = 0
    for tok in _NUM.findall(answer):
        clean = tok.replace(",", "")
        val = float(clean)
        nd = len(clean.split(".")[1]) if "." in clean else 0
        total += 1
        ok = any(
            abs(round(abs(s) * k, nd) - abs(val)) <= 0.5 * 10 ** (-nd) + 1e-12
            for s in source
            for k in (1.0, 100.0)
        )
        bad += not ok
    return total, bad


def _args_match(gold: dict, got: dict) -> bool:
    for k, v in gold.items():
        if k not in got:
            return False
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if abs(float(got[k]) - float(v)) > 1e-6 * max(1.0, abs(float(v))):
                return False
        elif got[k] != v:
            return False
    return True


def score(item: dict, out: dict) -> dict:
    calls = out["calls"]
    gold_tool, gold_args, split = item["tool"], item["args"], item["split"]
    called = [c["tool"] for c in calls]
    valid = [c for c in calls if c["ok"] or c.get("error_type") == "tool_error"]
    n_num, n_bad = numeric_consistency(out["answer"], item["question"], calls)
    successful = [c for c in calls if c["ok"]]
    if split == "out_of_scope":
        selection = len(calls) == 0
        completion = selection
        args_ok = None
    elif split == "missing_argument":
        # correct behaviour: no successful call that required inventing the missing/invalid value
        selection = (len(calls) == 0) or (gold_tool in called)
        completion = not any(c["ok"] and c["tool"] == gold_tool for c in calls)
        args_ok = None
    else:
        selection = gold_tool in called
        hit = [c for c in calls if c["tool"] == gold_tool]
        args_ok = bool(hit) and any(_args_match(gold_args, c["args"] or {}) for c in hit)
        completion = bool(hit) and any(c["ok"] and _args_match(gold_args, c["args"] or {}) for c in hit)
    needed = 0 if gold_tool is None or split == "missing_argument" else 1
    return {
        "id": item["id"],
        "split": split,
        "question": item["question"],
        "gold_tool": gold_tool,
        "called": "|".join(called),
        "tool_selection_correct": bool(selection),
        "n_calls": len(calls),
        "n_valid_calls": len(valid),
        "arguments_correct": args_ok,
        "task_completed": bool(completion),
        "unnecessary_calls": max(0, len(calls) - needed)
        if split != "missing_argument"
        else max(0, len(calls) - 1),
        "numbers_in_answer": n_num,
        "untraceable_numbers": n_bad,
        "unsupported_answer": bool(
            n_bad > 0 or (n_num > 0 and not successful and _numbers(item["question"]) == [])
        ),
        "answer": out["answer"],
    }


def summarise(df: pd.DataFrame, agent: str) -> pd.DataFrame:
    rows = []
    for split, g in list(df.groupby("split")) + [("ALL", df)]:
        calls = g.n_calls.sum()
        rows.append(
            {
                "agent": agent,
                "split": split,
                "n": len(g),
                "tool_selection_accuracy": g.tool_selection_correct.mean(),
                "argument_validity": g.n_valid_calls.sum() / calls if calls else float("nan"),
                "argument_correctness": g.arguments_correct.dropna().astype(float).mean()
                if g.arguments_correct.notna().any()
                else float("nan"),
                "task_completion": g.task_completed.mean(),
                "unnecessary_calls_per_request": g.unnecessary_calls.mean(),
                "numeric_consistency": 1 - g.untraceable_numbers.sum() / max(g.numbers_in_answer.sum(), 1),
                "unsupported_answer_rate": g.unsupported_answer.mean(),
            }
        )
    return pd.DataFrame(rows)


def run_agent(agent: str, session: PlanningSession, items: list[dict], llm=None) -> pd.DataFrame:
    rows = []
    for item in items:
        out = router.answer(item["question"], session) if agent == "router" else llm.answer(item["question"])
        rows.append(score(item, out))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["router", "llm", "both"], default="router")
    ap.add_argument(
        "--synthetic", action="store_true", help="use the synthetic toy network instead of processed MTA data"
    )
    args = ap.parse_args()
    items = json.loads(EVAL_SET.read_text())
    session = demo_session(prefer_real=not args.synthetic)
    res = resolve("results")
    res.mkdir(exist_ok=True)
    summaries = []
    if args.agent in ("router", "both"):
        df = run_agent("router", session, items)
        df.to_csv(res / "agent_eval_router.csv", index=False)
        summaries.append(summarise(df, "router"))
    if args.agent in ("llm", "both"):
        status = {
            "agent": "llm",
            "status": "PENDING",
            "command": "uv run --extra agent python -m busopt.agent.eval --agent llm",
        }
        try:
            from busopt.agent.llm import MODEL, LLMAgent

            llm = LLMAgent(session)
            df = run_agent("llm", session, items, llm)
            df.to_csv(res / "agent_eval_llm.csv", index=False)
            summaries.append(summarise(df, "llm"))
            status = {"agent": "llm", "status": "COMPLETED", "model": MODEL, "n": len(df)}
        except Exception as e:  # no credentials, no network, or an API error: record it, never fabricate
            status["reason"] = f"{type(e).__name__}: {e}"[:400]
        (res / "agent_eval_llm_status.json").write_text(json.dumps(status, indent=2))
        print(status)
    if summaries:
        out = pd.concat(summaries, ignore_index=True)
        prev = res / "agent_eval_summary.csv"
        if prev.exists():
            old = pd.read_csv(prev)
            out = pd.concat([old[~old.agent.isin(out.agent.unique())], out], ignore_index=True)
        out.to_csv(prev, index=False)
        pd.set_option("display.width", 220)
        print(out.round(3).to_string())


if __name__ == "__main__":
    main()
