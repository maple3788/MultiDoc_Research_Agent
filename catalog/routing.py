"""
Route user queries to document ids using the **summary-level FAISS** catalog (``catalog_store/``).

Flow: embed query → nearest neighbors on document summaries → keep ids that also have chunk
indexes under ``vector_stores/<id>/``.

**Relevance:** weak matches are dropped: (1) if the best hit is still too far (optional absolute
ceiling on the best L2), the route returns nothing; (2) only documents within an **L2 margin** of the
best hit are kept, so irrelevant runner-ups are excluded even when ``TOP_K`` is large.

**Note:** Summaries come from PostgreSQL and are rebuilt into ``catalog_store/`` on each upload.
"""

from __future__ import annotations

from agent.doc_index import list_chunk_indexed_ids
from catalog.ivf_pq_faiss import search_catalog
from config import (
    CATALOG_ROUTE_L2_MARGIN,
    CATALOG_ROUTE_MAX_BEST_L2,
    CATALOG_ROUTE_MAX_L2,
    CATALOG_ROUTE_OVERFETCH,
    CATALOG_ROUTE_TOP_K,
)


def route_query_to_documents(
    query: str,
    top_k: int | None = None,
    max_l2: float | None = None,
    max_best_l2: float | None = None,
    l2_margin: float | None = None,
) -> list[tuple[str, float]]:
    """
    Return ``(document_id, l2_distance)`` for **relevant** documents: chunk-indexed ids whose
    summary vectors are close to the query in L2 space.

    Parameters override :mod:`config` when not ``None`` (tests / callers).
    """
    top_k = top_k if top_k is not None else CATALOG_ROUTE_TOP_K
    if max_l2 is None:
        max_l2 = CATALOG_ROUTE_MAX_L2
    if max_best_l2 is None:
        max_best_l2 = CATALOG_ROUTE_MAX_BEST_L2
    if l2_margin is None:
        l2_margin = CATALOG_ROUTE_L2_MARGIN

    chunk_ids = set(list_chunk_indexed_ids())
    if not chunk_ids:
        return []

    k_search = min(CATALOG_ROUTE_OVERFETCH, max(top_k * 8, top_k, 16))
    hits = search_catalog(query.strip(), k=k_search)

    candidates: list[tuple[str, float]] = []
    for uuid_str, dist in hits:
        if uuid_str not in chunk_ids:
            continue
        candidates.append((uuid_str, dist))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[1])
    best_dist = candidates[0][1]

    # Entire corpus is a poor match for this query → do not retrieve any document.
    if max_best_l2 is not None and best_dist > max_best_l2:
        return []

    # Keep only hits within additive L2 slack of the best (drops semantically weak runner-ups).
    if l2_margin is not None:
        ceiling = best_dist + l2_margin
        candidates = [(u, d) for u, d in candidates if d <= ceiling]

    if max_l2 is not None:
        candidates = [(u, d) for u, d in candidates if d <= max_l2]

    return candidates[:top_k]
