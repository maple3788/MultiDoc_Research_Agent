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
