"""
Milvus-backed retriever: chunk vectors scoped by ``document_id`` in ``MILVUS_COLLECTION_CHUNKS``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from config import CHUNK_TOP_K


def retrieve_from_store(document_id: str, query: str, k: int | None = None) -> str:
    """Load chunk vectors for ``document_id`` and return the top-k chunk texts."""
    from catalog.milvus_catalog import search_chunks

    if k is None:
        k = CHUNK_TOP_K
    pairs = search_chunks(document_id.strip(), query.strip(), k=k)
    lines: list[str] = [f"[document_id={document_id}]"]
    for i, (text, score) in enumerate(pairs, start=1):
        lines.append(f"--- Chunk {i} (score={score:.4f}) ---")
        lines.append(text.strip())
    return "\n".join(lines)


@tool
def retrieve_document_chunks(query: str, document_id: str) -> str:
    """
    Retrieve the top similar text chunks from a single indexed document.

    Args:
        query: What to search for in this document (focused retrieval query).
        document_id: UUID string for the document (Milvus ``document_id``).

    Returns:
        Top-K chunks (``CHUNK_TOP_K`` in config) for downstream synthesis.
    """
    return retrieve_from_store(document_id.strip(), query.strip(), k=CHUNK_TOP_K)


def get_tools() -> list[Any]:
    return [retrieve_document_chunks]


def clear_vector_cache() -> None:
    """Legacy hook after re-ingest (FAISS cache); Milvus needs no process-local cache."""

