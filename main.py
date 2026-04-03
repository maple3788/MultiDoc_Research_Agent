"""
Run the full pipeline: ensure FAISS indexes exist, route (summary FAISS) -> retrieve -> synthesize, print report.
"""

from __future__ import annotations

from pathlib import Path

from agent.workflow import run_agent
from ingest import run_ingest
from tools.retriever_tool import clear_vector_cache

PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_ROOT = PROJECT_ROOT / "vector_stores"
REQUIRED_DOCS = ("company_a_q3", "company_b_q3")


def ensure_vector_indexes() -> None:
    """Build data/ + vector_stores/ if indexes are missing."""
    missing = any(not (VECTOR_ROOT / doc / "index.faiss").is_file() for doc in REQUIRED_DOCS)
    if missing:
        run_ingest()
        clear_vector_cache()


def main() -> None:
    ensure_vector_indexes()
    question = (
        "Compare the Q3 financial results of Company A vs Company B, "
        "including revenue trends and profitability."
    )
    result = run_agent(question)
    print(result.get("report", "(no report in result)"))


if __name__ == "__main__":
    main()
