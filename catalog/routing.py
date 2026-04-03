"""
Route user queries to document ids using the **summary-level FAISS** catalog (``catalog_store/``).

Flow: embed query → nearest neighbors on document summaries → keep ids that also have chunk
indexes under ``vector_stores/<id>/``. This replaces LLM-based "which document" planning.

**Note:** Summaries come from PostgreSQL (rebuilt into ``catalog_store/`` on ingest). Chunk-only
demo ids (e.g. ``company_a_q3`` from ``python ingest.py`` without a DB row) are **not** in the
summary index until those documents exist in the database with summaries.
"""

from __future__ import annotations

from agent.doc_index import list_chunk_indexed_ids
from catalog.ivf_pq_faiss import search_catalog
from config import CATALOG_ROUTE_MAX_L2, CATALOG_ROUTE_OVERFETCH, CATALOG_ROUTE_TOP_K


def route_query_to_documents(
    query: str,
    top_k: int | None = None,
    max_l2: float | None = None,
) -> list[tuple[str, float]]:
    """
    Return ``(document_id, l2_distance)`` for documents whose **summaries** best match ``query``,
    intersected with ids that have per-document **chunk** FAISS under ``vector_stores/``.

    ``max_l2`` overrides :envvar:`CATALOG_ROUTE_MAX_L2` when set; ``None`` uses the env default
    (typically no cutoff).
    """
    top_k = top_k if top_k is not None else CATALOG_ROUTE_TOP_K
    if max_l2 is None:
        max_l2 = CATALOG_ROUTE_MAX_L2

    chunk_ids = set(list_chunk_indexed_ids())
    if not chunk_ids:
        return []

    # Ask FAISS for more neighbors than we need, then filter to chunk-indexed ids.
    k_search = min(CATALOG_ROUTE_OVERFETCH, max(top_k * 8, top_k))
    hits = search_catalog(query.strip(), k=k_search)
    out: list[tuple[str, float]] = []
    for uuid_str, dist in hits:
        if uuid_str not in chunk_ids:
            continue
        if max_l2 is not None and dist > max_l2:
            continue
        out.append((uuid_str, dist))
        if len(out) >= top_k:
            break
    return out
