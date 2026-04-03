"""Project paths and environment-backed settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

# PostgreSQL (SQLAlchemy URL). Example: postgresql+psycopg://user:pass@localhost:5432/multidoc
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/multidoc",
)

UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", str(PROJECT_ROOT / "uploads")))
CATALOG_STORE_DIR = Path(os.environ.get("CATALOG_STORE_DIR", str(PROJECT_ROOT / "catalog_store")))

# Doc-level catalog: use IndexIVFPQ only when there are enough vectors for stable
# FAISS training (PQ/IVF); otherwise IndexFlatL2. Default 256 matches common faiss
# training constraints for IndexIVFPQ with 8-bit PQ.
CATALOG_MIN_VECTORS_IVFPQ = int(os.environ.get("CATALOG_MIN_VECTORS_IVFPQ", "256"))

# Agent routing: query embedding → summary FAISS → document ids (see ``catalog/routing.py``).
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
