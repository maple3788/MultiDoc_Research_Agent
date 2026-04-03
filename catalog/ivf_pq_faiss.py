"""
Document-level FAISS catalog: one embedding per document summary.

Uses **IndexIVFPQ** (IVF + Product Quantization) when there are enough vectors;
otherwise **IndexFlatL2** inside **IndexIDMap** for small corpora (IVF training is unreliable).

Vectors are stored with sequential int64 ids; ``uuids.json`` aligns id → PostgreSQL document UUID.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from agent.llm import get_embeddings
from config import CATALOG_MIN_VECTORS_IVFPQ, CATALOG_STORE_DIR


def _pick_pq_m(dim: int) -> int:
    """Choose ``m`` so ``dim % m == 0`` (required by PQ). Prefer ~64."""
    for m in (96, 64, 48, 32, 24, 16):
        if dim % m == 0:
            return m
    return 1


def build_index(vectors: np.ndarray) -> faiss.Index:
    """
    ``vectors``: float32 array, shape (n, d), L2 normalized or raw (consistent with query).
    """
    n, d = vectors.shape
    assert vectors.dtype == np.float32

    if n == 0:
        raise ValueError("No vectors to index")

    ids = np.arange(n, dtype=np.int64)

    # Below this size, IVFPQ.train often fails or warns heavily (PQ/IVF needs enough points).
    if n < CATALOG_MIN_VECTORS_IVFPQ:
        flat = faiss.IndexFlatL2(d)
        index = faiss.IndexIDMap(flat)
        index.add_with_ids(vectors, ids)
        return index

    # IVF + PQ: pick nlist <= n and keep k-means well-conditioned (rule-of-thumb: n >= 39 * nlist).
    nlist = min(512, max(4, int(np.sqrt(n) * 4)))
    nlist = min(nlist, max(4, n // 40))

    m = _pick_pq_m(d)
    quantizer = faiss.IndexFlatL2(d)
    ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    ivfpq.train(vectors)
    index = faiss.IndexIDMap(ivfpq)
    index.add_with_ids(vectors, ids)
    return index


def save_index(index: faiss.Index, uuids: list[str], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(directory / "catalog.faiss"))
    (directory / "uuids.json").write_text(json.dumps(uuids), encoding="utf-8")


def load_index(directory: Path) -> tuple[faiss.Index, list[str]]:
    idx_path = directory / "catalog.faiss"
    meta_path = directory / "uuids.json"
    if not idx_path.is_file():
        raise FileNotFoundError(f"Missing {idx_path}")
    index = faiss.read_index(str(idx_path))
    uuids = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else []
    return index, uuids


def embed_summaries(texts: list[str]) -> np.ndarray:
    emb = get_embeddings()
    rows = [emb.embed_query(t) for t in texts]
    return np.array(rows, dtype=np.float32)


def rebuild_catalog_from_rows(
    rows: list[tuple[str, str]],
    store_dir: Path | None = None,
) -> None:
    """
    ``rows``: list of (document_uuid_str, summary_text).
    Writes FAISS index + uuids.json under ``store_dir``.
    """
    store_dir = store_dir or CATALOG_STORE_DIR
    if not rows:
        if (store_dir / "catalog.faiss").is_file():
            (store_dir / "catalog.faiss").unlink(missing_ok=True)
            (store_dir / "uuids.json").unlink(missing_ok=True)
        return

    uuids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    X = embed_summaries(texts)
    index = build_index(X)
    save_index(index, uuids, store_dir)


def search_catalog(
    query: str,
    k: int = 5,
    store_dir: Path | None = None,
) -> list[tuple[str, float]]:
    """
    Returns list of (document_uuid, distance_or_score).

    Lower distance is better for L2 / IVFPQ inner workflow (FAISS returns distances).
    """
    store_dir = store_dir or CATALOG_STORE_DIR
    try:
        index, uuids = load_index(store_dir)
    except FileNotFoundError:
        return []
    if not uuids:
        return []

    emb = get_embeddings()
    q = np.array([emb.embed_query(query)], dtype=np.float32)
    scores, indices = index.search(q, min(k, len(uuids)))
    out: list[tuple[str, float]] = []
    for dist, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        if idx >= len(uuids):
            continue
        out.append((uuids[int(idx)], float(dist)))
    return out
