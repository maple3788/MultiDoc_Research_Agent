"""
Build per-document chunk FAISS indexes under ``vector_stores/<doc_id>/``.

Used by ``catalog.pipeline`` after each upload. Embeddings must stay consistent with
``agent.llm.get_embeddings()`` (same Ollama model as at query time).
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.llm import get_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_ROOT = PROJECT_ROOT / "vector_stores"


def _chunk_text(text: str, source_id: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=c, metadata={"source": source_id}) for c in chunks]


def build_faiss_for_doc(doc_id: str, text: str, embeddings) -> None:
    docs = _chunk_text(text, doc_id)
    store = FAISS.from_documents(docs, embeddings)
    out = VECTOR_ROOT / doc_id
    out.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out))


def build_chunk_index_for_text(doc_id: str, text: str) -> None:
    """Chunk ``text``, embed, and save ``vector_stores/<doc_id>/``."""
    embeddings = get_embeddings()
    build_faiss_for_doc(doc_id, text, embeddings)
