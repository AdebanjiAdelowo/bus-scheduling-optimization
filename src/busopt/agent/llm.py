"""LLM planning agent: Claude selects and calls the typed tools; it never computes results itself.

A manual tool loop is used (rather than the SDK tool runner) because the evaluation needs the exact
sequence of tool calls and raw arguments, including invalid ones, and because every call must pass
through :func:`busopt.agent.tools.call_tool` for validation.
"""

from __future__ import annotations

import json

from busopt.agent.tools import PlanningSession, anthropic_tool_definitions, call_tool

MODEL = "claude-opus-5-5"
MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """You are a planning analyst for a public bus network. Planners ask you what-if questions \
about demand, crowding, schedules, fleet size and robustness.

You answer by calling the provided tools, which run the actual forecasting, optimisation and \
simulation code. Every number in your answer must come from a tool result in this conversation: the \
tools are the only source of quantitative facts you have about this network, so if a tool has not \
returned a figure, you do not have it. If a tool reports an error or an infeasible instance, say so \
plainly and do not estimate what the result would have been.

The tools are read-only what-if analyses. Nothing you do changes a real timetable, vehicle or depot. \
If someone asks you to apply, publish or dispatch a schedule, explain that you can only analyse \
options and that a human planner has to make and implement the decision.

Call the fewest tools that answer the question; one is usually enough. If a required detail is \
missing and no default is sensible (for example a percentage for a fleet change), ask for it instead \
of guessing. Questions unrelated to this bus network need no tool call.

Keep answers short and concrete for a transport planner: the direct answer first, then the figures \
that support it, then any caveat the tool output states (demand basis, assumed parameters, simulated \
quantities). Explain only what the tool output supports; do not add causes or recommendations that \
no tool result backs up."""


class LLMAgent:
    def __init__(
        self,
        session: PlanningSession,
        client=None,
        model: str = MODEL,
        effort: str = "medium",
        use_fallbacks: bool = True,
    ):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.session, self.client, self.model, self.effort, self.use_fallbacks = (
            session,
            client,
            model,
            effort,
            use_fallbacks,
        )
        self.tools = anthropic_tool_definitions()

    def _create(self, messages: list[dict]):
        kwargs = dict(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=self.tools,
            messages=messages,
            output_config={"effort": self.effort},
        )
        if self.use_fallbacks:
            # A safety-classifier decline is re-run server-side on a fallback model instead of failing the request.
            return self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        return self.client.messages.create(**kwargs)

    def answer(self, question: str) -> dict:
        messages: list[dict] = [{"role": "user", "content": question}]
        calls: list[dict] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        stop_reason, text = None, ""
        for _ in range(MAX_TOOL_ROUNDS + 1):
            response = self._create(messages)
            stop_reason = response.stop_reason
            usage["input_tokens"] += response.usage.input_tokens
            usage["output_tokens"] += response.usage.output_tokens
            text = "".join(b.text for b in response.content if b.type == "text")
            if stop_reason == "refusal":
                text = text or "The model declined this request."
                break
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if stop_reason != "tool_use" or not tool_uses:
                break
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for tu in tool_uses:  # all results go back in one user message
                out = call_tool(self.session, tu.name, tu.input if isinstance(tu.input, dict) else None)
                calls.append(out)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "is_error": not out["ok"],
                        "content": json.dumps(
                            out["result"] if out["ok"] else {"error": out["error"]}, default=str
                        ),
                    }
                )
            messages.append({"role": "user", "content": results})
        else:
            text = text or "Stopped: the tool-call limit was reached before an answer was produced."
        return {
            "answer": text,
            "calls": calls,
            "agent": "llm",
            "model": self.model,
            "stop_reason": stop_reason,
            "usage": usage,
        }
