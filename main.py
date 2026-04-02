"""
Entry point: run the multi-document research agent on a sample comparative query.
"""

from __future__ import annotations

from agent.workflow import run_agent


def main() -> None:
    question = (
        "Compare the Q3 financial results of Company A vs Company B, "
        "including revenue trends and profitability."
    )
    result = run_agent(question)
    print(result.get("report", "(no report in result)"))


if __name__ == "__main__":
    main()
