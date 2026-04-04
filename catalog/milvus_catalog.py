"""
Milvus-backed vector catalog: **document-level summary vectors** + **per-document chunks**.

Replaces the previous FAISS + on-disk ``catalog_store/`` and ``vector_stores/`` layout.
Requires a running Milvus instance (see ``docker-compose.yml``) and consistent
embeddings from :func:`agent.llm.get_embeddings`.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, TypeVar

import numpy as np
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from agent.llm import get_embeddings
from config import (
    MILVUS_COLLECTION_CATALOG,
    MILVUS_COLLECTION_CHUNKS,
    MILVUS_DB_NAME,
    MILVUS_TIMEOUT_SEC,
    MILVUS_TOKEN,
    MILVUS_URI,
    milvus_uri_candidates,
)

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_client: MilvusClient | None = None
_dim: int | None = None

_T = TypeVar("_T")

_MILVUS_CONNECT_RETRIES = 4


def _is_milvus_connect_failure(exc: BaseException) -> bool:
    """True for gRPC channel-ready / connect failures (code=2 or matching message)."""
    try:
        from pymilvus.exceptions import MilvusException

        if isinstance(exc, MilvusException) and getattr(exc, "code", None) == 2:
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    return "fail connecting to server" in msg or "illegal connection params" in msg


def _milvus_rpc_retry(fn: Callable[[], _T]) -> _T:
    """
    Retry after ``reset_milvus_client()`` on connect/channel failures.

    Long-lived apps (e.g. Streamlit) can keep a dead gRPC handler; one retry is often not enough
    if Milvus was still starting or the channel was half-open.
    """
    last: BaseException | None = None
    for attempt in range(_MILVUS_CONNECT_RETRIES):
        try:
            return fn()
        except Exception as e:
            last = e
            if not _is_milvus_connect_failure(e) or attempt == _MILVUS_CONNECT_RETRIES - 1:
                raise
            logger.warning(
                "Milvus connect/RPC failed (attempt %s/%s), resetting client: %s",
                attempt + 1,
                _MILVUS_CONNECT_RETRIES,
                e,
            )
            reset_milvus_client()
            import time

            time.sleep(min(2.0 * (2**attempt), 8.0))
    assert last is not None
    raise last


def reset_milvus_client() -> None:
    """Drop cached client (e.g. after Milvus container restart). Thread-safe."""
    global _client
    with _lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:
                pass
            _client = None


def _milvus_client_kwargs_for_uri(uri: str) -> dict[str, Any]:
    # dedicated=True: avoid reusing a stale pooled handler after Docker restarts (code=2).
    kwargs: dict[str, Any] = {"uri": uri, "dedicated": True}
    if MILVUS_TIMEOUT_SEC is not None:
        kwargs["timeout"] = MILVUS_TIMEOUT_SEC
    if MILVUS_TOKEN:
        kwargs["token"] = MILVUS_TOKEN
    if MILVUS_DB_NAME:
        kwargs["db_name"] = MILVUS_DB_NAME
    if uri.startswith("http://"):
        kwargs["secure"] = False
    # Forwarded to GrpcHandler; helps some macOS / proxy setups where channel-ready never completes.
    kwargs["grpc_options"] = {
        "grpc.keepalive_time_ms": 10_000,
        "grpc.keepalive_timeout_ms": 20_000,
        "grpc.keepalive_permit_without_calls": True,
    }
    return kwargs


def _build_milvus_client() -> MilvusClient:
    from pymilvus.exceptions import MilvusException

    last_err: Exception | None = None
    uris = milvus_uri_candidates()
    for uri in uris:
        kwargs = _milvus_client_kwargs_for_uri(uri)
        for attempt in range(2):
            try:
                client = MilvusClient(**kwargs)
                if uri != MILVUS_URI:
                    logger.info("Milvus connected via fallback URI %s (MILVUS_URI=%s)", uri, MILVUS_URI)
                return client
            except MilvusException as e:
                last_err = e
                if getattr(e, "code", None) == 2 and attempt == 0:
                    logger.warning(
                        "Milvus connect code=2 for %s (attempt %s), retrying once: %s",
                        uri,
                        attempt + 1,
                        e,
                    )
                    import time

                    time.sleep(2.0)
                    continue
                logger.warning("Milvus connect failed for %s: %s", uri, e)
                break
            except Exception as e:
                last_err = e
                logger.warning("Milvus connect failed for %s: %s", uri, e)
                break
    assert last_err is not None
    raise last_err


def _get_embedding_dim() -> int:
    global _dim
    if _dim is not None:
        return _dim
    emb = get_embeddings()
    v = emb.embed_query("dimension probe")
    _dim = len(v)
    return _dim


def get_milvus_client() -> MilvusClient:
    """Singleton Milvus client; URI fallback + retries inside ``_build_milvus_client``."""
    global _client
    with _lock:
        if _client is None:
            _client = _build_milvus_client()
            logger.info("Milvus client ready MILVUS_URI=%s timeout=%s", MILVUS_URI, MILVUS_TIMEOUT_SEC)
        return _client


def _ensure_catalog_collection(client: MilvusClient, dim: int) -> None:
    name = MILVUS_COLLECTION_CATALOG
    if client.has_collection(name):
        return
    client.create_collection(
        collection_name=name,
        dimension=dim,
        primary_field_name="document_id",
        id_type="string",
        vector_field_name="vector",
        metric_type="L2",
        auto_id=False,
        max_length=64,
    )


def _chunks_schema(dim: int) -> CollectionSchema:
    # Chunk texts are small (see ingest splitter); keep VARCHAR bounded for Milvus.
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=16384),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    return CollectionSchema(fields=fields, enable_dynamic_field=False)


def _ensure_chunks_collection(client: MilvusClient, dim: int) -> None:
    name = MILVUS_COLLECTION_CHUNKS
    if client.has_collection(name):
        return
    schema = _chunks_schema(dim)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="L2")
    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )


def ensure_milvus_collections() -> None:
    """Create collections if missing (uses current embedding dimension)."""
    dim = _get_embedding_dim()
    client = get_milvus_client()
    _ensure_catalog_collection(client, dim)
    _ensure_chunks_collection(client, dim)


def embed_summaries(texts: list[str]) -> np.ndarray:
    emb = get_embeddings()
    rows = [emb.embed_query(t) for t in texts]
    return np.array(rows, dtype=np.float32)


def rebuild_catalog_from_rows(
    rows: list[tuple[str, str]],
    store_dir: Any = None,
) -> None:
    """
    ``rows``: list of (document_uuid_str, summary_text).
    Replaces all vectors in the Milvus summary collection.
    """
    del store_dir  # legacy FAISS path argument; ignored

    def _run() -> None:
        dim = _get_embedding_dim()
        client = get_milvus_client()
        _ensure_catalog_collection(client, dim)

        cat = MILVUS_COLLECTION_CATALOG
        if client.has_collection(cat):
            client.drop_collection(cat)
        _ensure_catalog_collection(client, dim)

        if not rows:
            return

        uuids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        X = embed_summaries(texts)
        data: list[dict[str, Any]] = []
        for i, u in enumerate(uuids):
            data.append({"document_id": u, "vector": X[i].tolist()})
        client.insert(collection_name=cat, data=data)
        client.flush(cat)

    _milvus_rpc_retry(_run)


def search_catalog(
    query: str,
    k: int = 5,
    store_dir: Any = None,
) -> list[tuple[str, float]]:
    """
    Returns list of (document_uuid, L2 distance). Lower is better.
    """
    del store_dir
    ensure_milvus_collections()
    client = get_milvus_client()
    cat = MILVUS_COLLECTION_CATALOG
    if not client.has_collection(cat):
        return []

    emb = get_embeddings()
    q = emb.embed_query(query.strip())
    res = client.search(
        collection_name=cat,
        data=[q],
        limit=k,
        output_fields=["document_id"],
        search_params={"metric_type": "L2", "params": {}},
        consistency_level="Strong",
    )
    out: list[tuple[str, float]] = []
    for hit in res[0]:
        ent = hit.get("entity") or {}
        if not isinstance(ent, dict):
            ent = {}
        did = ent.get("document_id") or hit.get("id")
        dist = float(hit.get("distance", 0.0))
        if did:
            out.append((str(did), dist))
    return out


def replace_document_chunks(document_id: str, texts: list[str], vectors: list[list[float]]) -> None:
    """Delete existing chunks for ``document_id``, insert new rows."""

    def _run() -> None:
        ensure_milvus_collections()
        client = get_milvus_client()
        chunks = MILVUS_COLLECTION_CHUNKS
        esc = document_id.replace("\\", "\\\\").replace('"', '\\"')
        client.delete(collection_name=chunks, filter=f'document_id == "{esc}"')
        if not texts:
            client.flush(chunks)
            return
        rows = [{"document_id": document_id, "text": t, "vector": v} for t, v in zip(texts, vectors)]
        client.insert(collection_name=chunks, data=rows)
        client.flush(chunks)

    _milvus_rpc_retry(_run)


def delete_document_vectors(document_id: str) -> None:
    """Remove catalog row and all chunks for this document."""
    client = get_milvus_client()
    esc = document_id.replace("\\", "\\\\").replace('"', '\\"')
    cat, chunks = MILVUS_COLLECTION_CATALOG, MILVUS_COLLECTION_CHUNKS
    if client.has_collection(cat):
        try:
            client.delete(collection_name=cat, filter=f'document_id == "{esc}"')
            client.flush(cat)
        except Exception as e:
            logger.warning("Milvus catalog delete %s: %s", document_id, e)
    if client.has_collection(chunks):
        try:
            client.delete(collection_name=chunks, filter=f'document_id == "{esc}"')
            client.flush(chunks)
        except Exception as e:
            logger.warning("Milvus chunks delete %s: %s", document_id, e)


def search_chunks(document_id: str, query: str, k: int) -> list[tuple[str, float]]:
    """Return (chunk_text, distance) for top-k chunks within one document."""
    ensure_milvus_collections()
    client = get_milvus_client()
    chunks = MILVUS_COLLECTION_CHUNKS
    if not client.has_collection(chunks):
        return []
    emb = get_embeddings()
    q = emb.embed_query(query.strip())
    esc = document_id.replace("\\", "\\\\").replace('"', '\\"')
    res = client.search(
        collection_name=chunks,
        data=[q],
        limit=k,
        filter=f'document_id == "{esc}"',
        output_fields=["text", "document_id"],
        search_params={"metric_type": "L2", "params": {}},
        consistency_level="Strong",
    )
    out: list[tuple[str, float]] = []
    for hit in res[0]:
        ent = hit.get("entity") or {}
        if not isinstance(ent, dict):
            ent = {}
        text = ent.get("text") or ""
        dist = float(hit.get("distance", 0.0))
        out.append((str(text), dist))
    return out


def list_chunk_indexed_ids() -> list[str]:
    """Distinct document_ids that have at least one chunk row in Milvus."""
    ensure_milvus_collections()
    client = get_milvus_client()
    chunks = MILVUS_COLLECTION_CHUNKS
    if not client.has_collection(chunks):
        return []
    try:
        rows = client.query(
            collection_name=chunks,
            filter='document_id != ""',
            output_fields=["document_id"],
            limit=100_000,
        )
    except Exception:
        rows = client.query(
            collection_name=chunks,
            filter="pk > 0",
            output_fields=["document_id"],
            limit=100_000,
        )
    seen: set[str] = set()
    for r in rows:
        did = r.get("document_id")
        if did:
            seen.add(str(did))
    return sorted(seen)


def invalidate_chunk_id_cache() -> None:
    """Legacy hook (FAISS cache); no-op for Milvus."""


def milvus_healthcheck() -> str | None:
    """Return error message if Milvus is unreachable, else None."""
    try:
        c = get_milvus_client()
        c.list_collections()
        return None
    except Exception as e:
        return str(e)
