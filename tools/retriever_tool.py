"""
FAISS-backed retriever: one local index per document_id under vector_stores/<document_id>/.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

from agent.llm import get_embeddings
from config import CHUNK_TOP_K

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_ROOT = PROJECT_ROOT / "vector_stores"


@lru_cache(maxsize=16)
def _load_faiss(document_id: str) -> FAISS:
    folder = VECTOR_ROOT / document_id.strip()
    index_faiss = folder / "index.faiss"
    if not index_faiss.is_file():
        raise FileNotFoundError(
            f"No FAISS index at {folder}. Upload a document or rebuild chunk indexes."
        )
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(folder),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_from_faiss(document_id: str, query: str, k: int | None = None) -> str:
    """Load the store for ``document_id`` and return the top-k chunk texts."""
    if k is None:
        k = CHUNK_TOP_K
    store = _load_faiss(document_id)
    pairs = store.similarity_search_with_score(query.strip(), k=k)
    lines: list[str] = [f"[document_id={document_id}]"]
    for i, (doc, score) in enumerate(pairs, start=1):
        lines.append(f"--- Chunk {i} (score={score:.4f}) ---")
        lines.append(doc.page_content.strip())
    return "\n".join(lines)


@tool
def retrieve_document_chunks(query: str, document_id: str) -> str:
    """
    Retrieve the top similar text chunks from a single indexed document.

    Args:
        query: What to search for in this document (focused retrieval query).
        document_id: UUID string folder name under vector_stores/.

    Returns:
        Top-K chunks (``CHUNK_TOP_K`` in config) for downstream synthesis.
    """
    return retrieve_from_faiss(document_id.strip(), query.strip(), k=CHUNK_TOP_K)


def get_tools() -> list[Any]:
    return [retrieve_document_chunks]


def clear_vector_cache() -> None:
    """Drop cached FAISS handles (e.g. after re-ingest in the same process)."""
    _load_faiss.cache_clear()
