"""
LLM and embedding model factories (Ollama by default).

Uses ``langchain-ollama`` so ``ChatOllama`` supports structured output and matches current APIs.

Requires a running Ollama server (`ollama serve`) and pulled models, e.g.:
  ollama pull llama3.2
  ollama pull nomic-embed-text
"""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama, OllamaEmbeddings


def _ollama_base_url(explicit: str | None) -> str | None:
    return explicit or os.environ.get("OLLAMA_BASE_URL") or None


def get_chat_llm(
    model: str | None = None,
    *,
    temperature: float = 0.2,
    base_url: str | None = None,
) -> ChatOllama:
    """Default chat model: Llama 3.2 via Ollama (override with OLLAMA_CHAT_MODEL)."""
    kwargs: dict = {
        "model": model or os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2"),
        "temperature": temperature,
    }
    bu = _ollama_base_url(base_url)
    if bu:
        kwargs["base_url"] = bu
    return ChatOllama(**kwargs)


def get_embeddings(
    model: str | None = None,
    *,
    base_url: str | None = None,
) -> OllamaEmbeddings:
    """Embeddings for FAISS ingest and retrieval (must match ingest-time model)."""
    kwargs: dict = {"model": model or os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")}
    bu = _ollama_base_url(base_url)
    if bu:
        kwargs["base_url"] = bu
    return OllamaEmbeddings(**kwargs)
