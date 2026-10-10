"""Command-line entry point: ``busopt <command>``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="busopt", description="Forecast-informed bus scheduling experiments")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_ in (
        ("download", "download GTFS, ridership, APC profiles and weather"),
        ("build", "build processed tables and validate the network"),
        ("forecast", "run the forecasting comparison"),
        ("policies", "compare schedule policies on held-out days"),
        ("sensitivity", "cost/service frontier, fleet and demand scaling"),
        ("stress", "uncertainty and disruption stress tests"),
        ("scalability", "solver scaling and heuristic gaps"),
        ("figures", "render documentation figures"),
        ("all", "run every stage above in order"),
    ):
        sub.add_parser(name, help=help_)
    ae = sub.add_parser("agent-eval", help="evaluate the router and/or the LLM agent")
    ae.add_argument("--agent", choices=["router", "llm", "both"], default="router")
    ask = sub.add_parser("ask", help="ask the planning assistant a question")
    ask.add_argument("question")
    ask.add_argument("--agent", choices=["router", "llm"], default="router")
    args = p.parse_args(argv)

    def run(cmd: str) -> None:
        if cmd == "download":
            from busopt.data.download import main as m

            m()
        elif cmd == "build":
            from busopt.data.build import build

            print(build()["network_validation"].to_string())
        elif cmd == "forecast":
            from busopt.forecast.evaluate import run as r

            r()
        elif cmd == "policies":
            from busopt.experiments.policy_comparison import run as r

            r()
        elif cmd == "sensitivity":
            from busopt.experiments.sensitivity import run as r

            r()
        elif cmd == "stress":
            from busopt.experiments.stress import run as r

            r()
        elif cmd == "scalability":
            from busopt.experiments.scalability import run as r

            r()
        elif cmd == "figures":
            from busopt.viz.plots import main as m

            m()

    if args.cmd == "all":
        for c in (
            "download",
            "build",
            "forecast",
            "policies",
            "sensitivity",
            "stress",
            "scalability",
            "figures",
        ):
            print(f"== {c} ==", flush=True)
            run(c)
    elif args.cmd == "agent-eval":
        from busopt.agent import eval as ev

        sys.argv = ["busopt-agent-eval", "--agent", args.agent]
        ev.main()
    elif args.cmd == "ask":
        from busopt.agent import router
        from busopt.agent.tools import demo_session

        s = demo_session()
        if args.agent == "llm":
            from busopt.agent.llm import LLMAgent

            out = LLMAgent(s).answer(args.question)
        else:
            out = router.answer(args.question, s)
        print(out["answer"])
        for c in out["calls"]:
            print(f"  [tool] {c['tool']} {c['args']} ok={c['ok']}")
    else:
        run(args.cmd)


if __name__ == "__main__":
    main()
