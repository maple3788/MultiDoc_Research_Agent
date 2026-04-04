"""Project paths and environment-backed settings."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

# Resolve first so `.env` loads from the repo root even when cwd differs (e.g. `streamlit run /path/to/app.py`).
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _merge_no_proxy_for_local_grpc() -> None:
    """gRPC to local Milvus must not go through HTTP(S)_PROXY — that often yields pymilvus code=2."""
    must = ("127.0.0.1", "localhost", "::1")
    for key in ("NO_PROXY", "no_proxy"):
        existing = (os.environ.get(key) or "").strip()
        parts = [p.strip() for p in existing.split(",") if p.strip()]
        for m in must:
            if m not in parts:
                parts.append(m)
        os.environ[key] = ",".join(parts)


_merge_no_proxy_for_local_grpc()

# PostgreSQL (SQLAlchemy URL). Example: postgresql+psycopg://user:pass@localhost:5432/multidoc
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/multidoc",
)

UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", str(PROJECT_ROOT / "uploads")))
# Legacy: FAISS catalog path (unused with Milvus). Kept for older `.env` files.
CATALOG_STORE_DIR = Path(os.environ.get("CATALOG_STORE_DIR", str(PROJECT_ROOT / "catalog_store")))

# --- Optional MinIO / S3 for raw uploads (same MinIO container as in docker-compose.yml when you use it) ---
# When MINIO_BUCKET + MINIO_ENDPOINT + keys are set, uploads go to the bucket instead of local ``uploads/``.
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "")
MINIO_PREFIX = os.environ.get("MINIO_UPLOAD_PREFIX", "multidoc-research")
_MINIO_SSL = os.environ.get("MINIO_USE_SSL", "false").strip().lower()
MINIO_USE_SSL = _MINIO_SSL in ("1", "true", "yes")
# If false, boto3 skips TLS cert verification (common with local self-signed MinIO).
_MINIO_CERT = os.environ.get("MINIO_CERT_CHECK", "true").strip().lower()
MINIO_CERT_CHECK_ENABLED = _MINIO_CERT not in ("0", "false", "no")
# PEM path: CA bundle or MinIO ``public.crt`` — passed to boto3 ``verify=`` (trusted TLS without disabling verify).
_MINIO_CA_RAW = (os.environ.get("MINIO_CA_BUNDLE") or os.environ.get("MINIO_TLS_CA") or "").strip()
if _MINIO_CA_RAW:
    _minio_ca_path = Path(_MINIO_CA_RAW).expanduser()
    if _minio_ca_path.is_file():
        MINIO_TLS_VERIFY: bool | str = str(_minio_ca_path.resolve())
    else:
        warnings.warn(
            f"MINIO_CA_BUNDLE path not found ({_MINIO_CA_RAW!r}); falling back to MINIO_CERT_CHECK.",
            stacklevel=1,
        )
        MINIO_TLS_VERIFY = MINIO_CERT_CHECK_ENABLED
else:
    MINIO_TLS_VERIFY = MINIO_CERT_CHECK_ENABLED
# Boto3/MinIO: omit or wrong region can yield 400 on HeadBucket; path-style avoids virtual-host issues.
MINIO_REGION = (os.environ.get("MINIO_REGION") or "us-east-1").strip()
# If true, do not call create_bucket (create the bucket in MinIO console / ``mc mb`` first).
_MINIO_SKIP = os.environ.get("MINIO_SKIP_BUCKET_CREATE", "").strip().lower()
MINIO_SKIP_BUCKET_CREATE = _MINIO_SKIP in ("1", "true", "yes")

# --- Milvus (vector DB) ---
# Local standalone uses plain HTTP, e.g. http://127.0.0.1:19530 (not https).
MILVUS_URI = (os.environ.get("MILVUS_URI") or "http://127.0.0.1:19530").strip()
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")
MILVUS_DB_NAME = os.environ.get("MILVUS_DB_NAME", "")
# pymilvus: timeout for channel-ready + RPC. Use 0 or "none" for unlimited channel wait (slow cold start).
_MILVUS_TO_RAW = (os.environ.get("MILVUS_TIMEOUT_SEC") or "120").strip().lower()
if _MILVUS_TO_RAW in ("", "none", "unlimited"):
    MILVUS_TIMEOUT_SEC: float | None = None
else:
    _mv = float(_MILVUS_TO_RAW)
    MILVUS_TIMEOUT_SEC = None if _mv <= 0 else _mv
# If True, retry connection with http://localhost:PORT when 127.0.0.1 fails (some macOS / resolver setups).
_MILVUS_FB = os.environ.get("MILVUS_TRY_LOCALHOST_FALLBACK", "true").strip().lower()
MILVUS_TRY_LOCALHOST_FALLBACK = _MILVUS_FB in ("1", "true", "yes")


def milvus_uri_candidates() -> list[str]:
    """URIs to try in order (primary, then localhost variant if enabled)."""
    primary = MILVUS_URI
    out: list[str] = [primary]
    if MILVUS_TRY_LOCALHOST_FALLBACK and "127.0.0.1" in primary:
        alt = primary.replace("127.0.0.1", "localhost", 1)
        if alt not in out:
            out.append(alt)
    return out
MILVUS_COLLECTION_CATALOG = os.environ.get("MILVUS_COLLECTION_CATALOG", "multidoc_catalog")
MILVUS_COLLECTION_CHUNKS = os.environ.get("MILVUS_COLLECTION_CHUNKS", "multidoc_chunks")

# --- Embeddings (must match across ingest and query) ---
# ollama (default) | nvidia (NeMo Retriever–compatible NVIDIA API embeddings; requires NVIDIA_API_KEY)
EMBEDDING_PROVIDER = (os.environ.get("EMBEDDING_PROVIDER") or "ollama").strip().lower()
NVIDIA_EMBEDDING_MODEL = os.environ.get("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")

# Agent routing: query embedding → summary vectors in Milvus → document ids (see ``catalog/routing.py``).
CATALOG_ROUTE_TOP_K = int(os.environ.get("CATALOG_ROUTE_TOP_K", "5"))
CATALOG_ROUTE_OVERFETCH = int(os.environ.get("CATALOG_ROUTE_OVERFETCH", "48"))
# Hard ceiling per document: drop hits with L2 above this (empty = no per-doc ceiling).
_CATALOG_ROUTE_MAX_L2_RAW = os.environ.get("CATALOG_ROUTE_MAX_L2", "").strip()
CATALOG_ROUTE_MAX_L2: float | None = float(_CATALOG_ROUTE_MAX_L2_RAW) if _CATALOG_ROUTE_MAX_L2_RAW else None
# If the *best* summary match is weaker than this L2, treat the query as unrelated → no documents (empty route).
_CATALOG_ROUTE_MAX_BEST_L2_RAW = os.environ.get("CATALOG_ROUTE_MAX_BEST_L2", "").strip()
CATALOG_ROUTE_MAX_BEST_L2: float | None = (
    float(_CATALOG_ROUTE_MAX_BEST_L2_RAW) if _CATALOG_ROUTE_MAX_BEST_L2_RAW else None
)
# Keep only documents with L2 <= best_L2 + margin (drops weak runner-ups). Set to 999 or similar to disable effectively.
_CATALOG_ROUTE_L2_MARGIN_RAW = os.environ.get("CATALOG_ROUTE_L2_MARGIN", "0.22").strip()
CATALOG_ROUTE_L2_MARGIN: float | None = (
    float(_CATALOG_ROUTE_L2_MARGIN_RAW) if _CATALOG_ROUTE_L2_MARGIN_RAW else None
)

# Per-document chunk retrieval for synthesis (``tools/retriever_tool``). Higher = more context for books/long docs.
CHUNK_TOP_K = max(1, int(os.environ.get("CHUNK_TOP_K", "8")))
