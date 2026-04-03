"""Discover chunk-level FAISS indexes under ``vector_stores/`` for planner prompts."""

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
    Human-readable bullet list for the planner: id — filename when known (PostgreSQL).
    Includes non-UUID demo ids (e.g. company_a_q3) without a DB row.
    """
    ids = list_chunk_indexed_ids()
    if not ids:
        return "(No chunk indexes yet. Upload text files or run `python ingest.py` for demo data.)"

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
                # e.g. company_a_q3
                label = f"{did} — (demo corpus)"
            lines.append(f"- {label}")
    finally:
        session.close()

    return "\n".join(lines)
