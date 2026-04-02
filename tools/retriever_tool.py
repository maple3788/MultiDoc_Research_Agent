"""
Retriever tool interface (mock backend).

Replace `MOCK_DOCUMENTS` and `_mock_search` with a real FAISS / vector pipeline later.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import tool

# Placeholder corpus: document_id -> list of chunk strings (simulated pages/sections)
MOCK_DOCUMENTS: dict[str, list[str]] = {
    "company_a_q3": [
        "Company A Q3 revenue was $120M, up 8% YoY. Net margin 14%.",
        "Company A highlighted strong enterprise SaaS growth in Q3.",
    ],
    "company_b_q3": [
        "Company B Q3 revenue was $95M, up 12% YoY. Net margin 11%.",
        "Company B noted expansion in APAC and improved unit economics in Q3.",
    ],
}


def _simple_keyword_score(query: str, chunk: str) -> float:
    q = set(re.findall(r"\w+", query.lower()))
    c = set(re.findall(r"\w+", chunk.lower()))
    if not q:
        return 0.0
    return len(q & c) / len(q)


def _mock_search(document_id: str, query: str, top_k: int = 2) -> list[str]:
    """Return top-k chunks from the mock store for this document."""
    chunks = MOCK_DOCUMENTS.get(document_id, [])
    if not chunks:
        return [
            f"[mock] No indexed content for document_id={document_id!r}. "
            "In production, index this id in your vector store."
        ]
    ranked = sorted(chunks, key=lambda ch: _simple_keyword_score(query, ch), reverse=True)
    return ranked[:top_k]


@tool
def retrieve_document_chunks(query: str, document_id: str) -> str:
    """
    Retrieve text chunks relevant to the query from a single document.

    Args:
        query: Natural-language search query (what to look for).
        document_id: Logical id or name of the document corpus (e.g. company ticker report id).

    Returns:
        Concatenated chunk text for the agent to reason over.
    """
    chunks = _mock_search(document_id.strip(), query.strip())
    header = f"[document_id={document_id}]\n"
    return header + "\n---\n".join(chunks)


def get_tools() -> list[Any]:
    """Expose tools for binding to an LLM or graph nodes."""
    return [retrieve_document_chunks]
