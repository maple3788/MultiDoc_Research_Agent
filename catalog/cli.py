"""CLI: init database, ingest files, search the summary catalog."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from sqlalchemy import select

from catalog.ivf_pq_faiss import search_catalog
from catalog.pipeline import ingest_file, list_ready_summaries
from db.models import Document
from db.session import SessionLocal, init_db


def cmd_init_db() -> None:
    init_db()
    print("Database tables created (SQLAlchemy).")


def cmd_upload(path: str) -> None:
    doc_id = ingest_file(Path(path))
    print(f"Ingested document id: {doc_id}")


def cmd_search(query: str, k: int) -> None:
    hits = search_catalog(query, k=k)
    if not hits:
        print("No catalog index or empty. Upload documents first.")
        return
    session = SessionLocal()
    try:
        for uuid_str, dist in hits:
            uid = uuid.UUID(uuid_str)
            row = session.scalar(select(Document).where(Document.id == uid))
            title = row.original_filename if row else uuid_str
            print(f"\n--- {title} (id={uuid_str}, L2={dist:.4f}) ---")
            if row and row.summary:
                print(row.summary[:1200])
    finally:
        session.close()


def cmd_list() -> None:
    rows = list_ready_summaries()
    if not rows:
        print("No ready documents.")
        return
    for uid, name, summary in rows:
        preview = (summary[:300] + "...") if len(summary) > 300 else summary
        print(f"\n* {name} [{uid}]\n  {preview}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PostgreSQL + FAISS catalog (IVF+PQ) tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db", help="Create database tables")

    u = sub.add_parser("upload", help="Ingest a UTF-8 text file")
    u.add_argument("path", type=str, help="Path to file")

    s = sub.add_parser("search", help="Search catalog by natural language (summary embeddings)")
    s.add_argument("query", type=str)
    s.add_argument("-k", type=int, default=5, help="Top-k documents")

    sub.add_parser("list", help="List ready documents and summary previews")

    args = p.parse_args(argv)

    if args.cmd == "init-db":
        cmd_init_db()
    elif args.cmd == "upload":
        cmd_upload(args.path)
    elif args.cmd == "search":
        cmd_search(args.query, args.k)
    elif args.cmd == "list":
        cmd_list()
    else:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
