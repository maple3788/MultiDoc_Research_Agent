"""Discover chunk-level FAISS indexes under ``vector_stores/`` (for debugging / UI helpers)."""

from __future__ import annotations

import uuid

from config import PROJECT_ROOT
from db.models import Document
from db.session import SessionLocal

VECTOR_ROOT = PROJECT_ROOT / "vector_stores"


def list_chunk_indexed_ids() -> list[str]:
    """Folder names under ``vector_stores/`` that contain ``index.faiss``."""
    if not VECTOR_ROOT.is_dir():
        return []
    return sorted(
        p.name
        for p in VECTOR_ROOT.iterdir()
        if p.is_dir() and (p / "index.faiss").is_file()
    )


def planner_doc_catalog_text() -> str:
    """
    Human-readable bullet list: id — filename when known (PostgreSQL).
    Non-UUID folder names (legacy local ids) show the raw id.
    """
    ids = list_chunk_indexed_ids()
    if not ids:
        return "(No chunk indexes yet. Upload documents in the **Upload** page.)"

    lines: list[str] = []
    session = SessionLocal()
    try:
        for did in ids:
            label = did
            try:
                u = uuid.UUID(did)
                row = session.get(Document, u)
                if row is not None:
                    label = f"{did} — {row.original_filename}"
            except ValueError:
                label = f"{did} — (non-UUID index)"
            lines.append(f"- {label}")
    finally:
        session.close()

    return "\n".join(lines)
