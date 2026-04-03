"""
CLI: run the research agent on a sample question (indexes come from uploads / ``rebuild_summary_catalog``).
"""

from __future__ import annotations

from agent.workflow import run_agent


def main() -> None:
    question = (
        "Summarize the main themes in the documents you have access to."
    )
    result = run_agent(question)
    print(result.get("report", "(no report in result)"))


if __name__ == "__main__":
    main()
