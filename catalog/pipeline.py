"""Ingest uploaded files: PostgreSQL metadata + summary + FAISS catalog rebuild."""

from __future__ import annotations

import logging
import shutil
import uuid
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from sqlalchemy import select

from catalog.ivf_pq_faiss import rebuild_catalog_from_rows
from catalog.summarize import summarize_for_catalog
from config import PROJECT_ROOT, UPLOADS_DIR
from db.models import DocStatus, Document
from db.session import SessionLocal
from ingest import build_chunk_index_for_text
from tools.retriever_tool import clear_vector_cache


def _read_utf8_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t.strip())
    return "\n\n".join(parts).strip()


def _read_document_text(path: Path) -> str:
    """Load plain text from disk; PDFs are extracted with pypdf."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw = _extract_pdf_text(path.read_bytes())
        if not raw:
            raise ValueError(
                "No extractable text in this PDF (scanned pages or empty). "
                "Try OCR or a text-based PDF."
            )
        return raw
    return _read_utf8_file(path)


def _persist_and_summarize(dest: Path, doc_id: uuid.UUID, original_filename: str) -> None:
    session = SessionLocal()
    try:
        row = Document(
            id=doc_id,
            original_filename=original_filename,
            storage_path=str(dest.relative_to(PROJECT_ROOT)),
            status=DocStatus.processing.value,
        )
        session.add(row)
        session.commit()

        try:
            raw = _read_document_text(dest)
            summary = summarize_for_catalog(raw)
            row.summary = summary
            row.status = DocStatus.ready.value
            row.error_message = None
            session.commit()
            # Chunk-level FAISS for Research agent: same text, id = document UUID folder.
            try:
                build_chunk_index_for_text(str(doc_id), raw)
                clear_vector_cache()
            except Exception as ce:
                logging.exception("Chunk index failed for %s", doc_id)
                row = session.get(Document, doc_id)
                if row is not None:
                    row.error_message = (row.error_message or "") + f"\n[chunk index] {str(ce)[:800]}"
                    session.commit()
        except Exception as e:
            row.status = DocStatus.failed.value
            row.error_message = str(e)[:2000]
            session.commit()
    finally:
        session.close()

    _rebuild_catalog_from_db()


def ingest_file(local_path: Path) -> uuid.UUID:
    """
    Copy file into ``uploads/``, summarize with LLM, persist metadata, rebuild catalog index.

    Supports UTF-8 text (``.txt``, ``.md``, ``.csv``, ``.json``) and text extracted from ``.pdf``.
    Other text extensions are read with ``errors=replace`` (may be noisy for binary).
    """
    local_path = local_path.resolve()
    if not local_path.is_file():
        raise FileNotFoundError(local_path)

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    doc_id = uuid.uuid4()
    safe_name = local_path.name.replace("..", "_")
    dest = UPLOADS_DIR / f"{doc_id}_{safe_name}"
    shutil.copy2(local_path, dest)
    _persist_and_summarize(dest, doc_id, local_path.name)
    return doc_id


def ingest_bytes(original_filename: str, content: bytes) -> uuid.UUID:
    """Write bytes to ``uploads/``, then same summarize + DB + catalog flow as ``ingest_file``."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    doc_id = uuid.uuid4()
    safe_name = (original_filename or "upload.txt").replace("..", "_").replace("/", "_")
    dest = UPLOADS_DIR / f"{doc_id}_{safe_name}"
    dest.write_bytes(content)
    _persist_and_summarize(dest, doc_id, original_filename or safe_name)
    return doc_id


def delete_document(doc_id: uuid.UUID) -> bool:
    """Remove row from PostgreSQL, delete file from disk, rebuild catalog."""
    path: Path | None = None
    session = SessionLocal()
    try:
        row = session.get(Document, doc_id)
        if row is None:
            return False
        path = PROJECT_ROOT / row.storage_path
        session.delete(row)
        session.commit()
    finally:
        session.close()

    if path is not None and path.is_file():
        try:
            path.unlink()
        except OSError:
            pass

    vs_dir = PROJECT_ROOT / "vector_stores" / str(doc_id)
    if vs_dir.is_dir():
        shutil.rmtree(vs_dir, ignore_errors=True)
    clear_vector_cache()

    _rebuild_catalog_from_db()
    return True


def _demo_catalog_pairs() -> list[tuple[str, str]]:
    """
    Non-UUID demo ids under ``vector_stores/`` (from ``python ingest.py``): embed source text
    as pseudo-summaries so summary-FAISS routing can find them alongside PostgreSQL uploads.
    """
    from ingest import COMPANY_A_TEXT, COMPANY_B_TEXT

    pairs: list[tuple[str, str]] = []
    for doc_id, text in (("company_a_q3", COMPANY_A_TEXT), ("company_b_q3", COMPANY_B_TEXT)):
        vs = PROJECT_ROOT / "vector_stores" / doc_id / "index.faiss"
        if vs.is_file():
            snippet = text.strip()[:4000]
            pairs.append((doc_id, snippet))
    return pairs


def _rebuild_catalog_from_db() -> None:
    session = SessionLocal()
    try:
        q = (
            select(Document)
            .where(Document.status == DocStatus.ready.value)
            .where(Document.summary.is_not(None))
            .order_by(Document.created_at.asc())
        )
        rows = list(session.scalars(q).all())
        pairs: list[tuple[str, str]] = [(str(r.id), r.summary or "") for r in rows if r.summary]
    finally:
        session.close()

    seen = {p[0] for p in pairs}
    for did, summ in _demo_catalog_pairs():
        if did not in seen:
            pairs.append((did, summ))
    rebuild_catalog_from_rows(pairs)


def rebuild_summary_catalog() -> None:
    """Rebuild ``catalog_store/`` from DB summaries plus demo ``vector_stores`` ids (if present)."""
    _rebuild_catalog_from_db()


def list_all_documents() -> list[Document]:
    """All documents, newest first."""
    session = SessionLocal()
    try:
        q = select(Document).order_by(Document.created_at.desc())
        return list(session.scalars(q).all())
    finally:
        session.close()


def list_ready_summaries() -> list[tuple[str, str, str]]:
    """Return (id, filename, summary) for ready documents."""
    session = SessionLocal()
    try:
        q = select(Document).where(Document.status == DocStatus.ready.value).order_by(Document.created_at.desc())
        rows = list(session.scalars(q).all())
        return [(str(r.id), r.original_filename, r.summary or "") for r in rows]
    finally:
        session.close()
