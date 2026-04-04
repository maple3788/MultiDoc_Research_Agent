"""Discover document ids that have chunk rows in Milvus (for routing / UI helpers)."""

from __future__ import annotations

import uuid

from catalog.milvus_catalog import list_chunk_indexed_ids
from db.models import Document
from db.session import SessionLocal


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
