"""
Build per-document **chunk vectors** in Milvus (``MILVUS_COLLECTION_CHUNKS``).

Embeddings must match ``agent.llm.get_embeddings()`` (same provider at query time).
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.llm import get_embeddings
from catalog.milvus_catalog import replace_document_chunks, reset_milvus_client


def _chunk_text(text: str, source_id: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=c, metadata={"source": source_id}) for c in chunks]


def build_chunk_index_for_text(doc_id: str, text: str) -> None:
    """Chunk ``text``, embed, and upsert vectors for ``doc_id`` in Milvus."""
    docs = _chunk_text(text, doc_id)
    if not docs:
        reset_milvus_client()
        replace_document_chunks(doc_id, [], [])
        return
    texts = [d.page_content for d in docs]
    emb = get_embeddings()
    vectors = emb.embed_documents(texts)
    # Reconnect after (possibly long) embedding so we do not reuse a stale Milvus client.
    reset_milvus_client()
    replace_document_chunks(doc_id, texts, vectors)
