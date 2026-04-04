"""Ingest uploaded files: PostgreSQL metadata + summary + Milvus vector rebuild."""

from __future__ import annotations

import logging
import uuid
from io import BytesIO
from pathlib import Path

_pipeline_log = logging.getLogger("multidoc.pipeline")

from pypdf import PdfReader

from sqlalchemy import select

from catalog.milvus_catalog import delete_document_vectors, rebuild_catalog_from_rows
from catalog.summarize import summarize_for_catalog
from catalog.upload_storage import delete_upload, read_upload_bytes, store_upload
from db.models import DocStatus, Document
from db.session import SessionLocal
from ingest import build_chunk_index_for_text
from tools.retriever_tool import clear_vector_cache


def _milvus_unreachable_hint() -> str:
    return (
        "Cannot connect to Milvus (MILVUS_URI, default `http://127.0.0.1:19530`). "
        "This error is usually gRPC failing to become ready (not plain TCP). "
        "Fixes: (1) Ensure `NO_PROXY`/`no_proxy` includes `127.0.0.1,localhost` if you use HTTP_PROXY. "
        "(2) Set `MILVUS_TIMEOUT_SEC=none` in `.env` if the server is slow to accept gRPC. "
        "(3) `docker compose ps` shows `standalone` healthy; image should match pymilvus 2.6 (see README). "
        "(4) Restart Streamlit after changing `.env`."
    )


def _is_milvus_connection_error(exc: BaseException) -> bool:
    try:
        from pymilvus.exceptions import MilvusException

        return isinstance(exc, MilvusException)
    except ImportError:
        return "MilvusException" in type(exc).__name__ or "Fail connecting to server" in str(exc)


def _read_utf8_bytes(data: bytes) -> str:
    return data.decode(encoding="utf-8", errors="replace")


def _extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t.strip())
    return "\n\n".join(parts).strip()


def _read_document_text(storage_path: str, original_filename: str) -> str:
    """Load plain text from stored bytes; PDFs are extracted with pypdf."""
    data = read_upload_bytes(storage_path)
    suffix = Path(original_filename).suffix.lower()
    if suffix == ".pdf":
        raw = _extract_pdf_text(data)
        if not raw:
            raise ValueError(
                "No extractable text in this PDF (scanned pages or empty). "
                "Try OCR or a text-based PDF."
            )
        return raw
    return _read_utf8_bytes(data)


def _persist_and_summarize(storage_path: str, doc_id: uuid.UUID, original_filename: str) -> None:
    session = SessionLocal()
    try:
        row = Document(
            id=doc_id,
            original_filename=original_filename,
            storage_path=storage_path,
            status=DocStatus.processing.value,
        )
        session.add(row)
        session.commit()

        try:
            raw = _read_document_text(storage_path, original_filename)
            summary = summarize_for_catalog(raw)
            row.summary = summary
            row.status = DocStatus.ready.value
            row.error_message = None
            session.commit()
            # Chunk vectors in Milvus for Research agent (same extracted text).
            try:
                build_chunk_index_for_text(str(doc_id), raw)
                clear_vector_cache()
            except Exception as ce:
                logging.exception("Chunk index failed for %s", doc_id)
                row = session.get(Document, doc_id)
                if row is not None:
                    detail = _milvus_unreachable_hint() if _is_milvus_connection_error(ce) else str(ce)[:800]
                    row.error_message = (row.error_message or "") + f"\n[chunk index] {detail}"
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
    Store file (local disk or MinIO), summarize with LLM, persist metadata, rebuild catalog index.

    Supports UTF-8 text (``.txt``, ``.md``, ``.csv``, ``.json``) and text extracted from ``.pdf``.
    Other text extensions are read with ``errors=replace`` (may be noisy for binary).
    """
    local_path = local_path.resolve()
    if not local_path.is_file():
        raise FileNotFoundError(local_path)

    doc_id = uuid.uuid4()
    content = local_path.read_bytes()
    storage_path = store_upload(doc_id, local_path.name, content)
    _persist_and_summarize(storage_path, doc_id, local_path.name)
    return doc_id


def ingest_bytes(original_filename: str, content: bytes) -> uuid.UUID:
    """Store bytes (local or MinIO), then same summarize + DB + catalog flow as ``ingest_file``."""
    _pipeline_log.info(
        "ingest_bytes start name=%r bytes=%s",
        original_filename,
        len(content),
    )
    doc_id = uuid.uuid4()
    safe_name = (original_filename or "upload.txt").replace("..", "_").replace("/", "_")
    storage_path = store_upload(doc_id, original_filename or safe_name, content)
    _pipeline_log.info("ingest_bytes stored path=%r doc_id=%s", storage_path, doc_id)
    _persist_and_summarize(storage_path, doc_id, original_filename or safe_name)
    _pipeline_log.info("ingest_bytes finished doc_id=%s", doc_id)
    return doc_id


def delete_document(doc_id: uuid.UUID) -> bool:
    """Remove row from PostgreSQL, delete stored file (disk or MinIO), rebuild catalog."""
    storage_path: str | None = None
    session = SessionLocal()
    try:
        row = session.get(Document, doc_id)
        if row is None:
            return False
        storage_path = row.storage_path
        session.delete(row)
        session.commit()
    finally:
        session.close()

    if storage_path:
        delete_upload(storage_path)

    delete_document_vectors(str(doc_id))
    clear_vector_cache()

    _rebuild_catalog_from_db()
    return True


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

    try:
        rebuild_catalog_from_rows(pairs)
    except Exception as e:
        _reraise_if_milvus_unreachable(e)
        raise


def _reraise_if_milvus_unreachable(exc: Exception) -> None:
    """Replace opaque pymilvus timeouts with a short operational hint."""
    if _is_milvus_connection_error(exc):
        raise RuntimeError(_milvus_unreachable_hint()) from exc


def rebuild_summary_catalog() -> None:
    """Rebuild Milvus summary collection from PostgreSQL document summaries."""
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
