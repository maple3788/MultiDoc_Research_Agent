"""
LLM and embedding model factories.

- **Chat:** ``LLM_PROVIDER=ollama`` (default), ``gemini``, or ``zai`` (GLM via ``zai-sdk``).
  Gemini: ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``. Z.ai: ``ZAI_API_KEY`` (see ``.env.example``).
- **Embeddings:** Ollama only (must match vectors in ``catalog_store/`` and ``vector_stores/``).
"""

from __future__ import annotations

import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings


def _ollama_base_url(explicit: str | None) -> str | None:
    return explicit or os.environ.get("OLLAMA_BASE_URL") or None


def _google_api_key() -> str | None:
    return (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
    )


def _zai_api_key() -> str | None:
    return os.environ.get("ZAI_API_KEY") or os.environ.get("BIGMODEL_API_KEY")


def get_chat_llm(
    model: str | None = None,
    *,
    temperature: float = 0.2,
    base_url: str | None = None,
    provider: str | None = None,
) -> BaseChatModel:
    """
    Chat model for summarization and synthesis.

    ``provider`` overrides ``LLM_PROVIDER`` when set (e.g. ``ollama``, ``gemini``, or ``zai``).
    Gemini needs a Google AI key; Z.ai needs ``ZAI_API_KEY`` and ``pip install zai-sdk``.
    Default chat is Ollama (``OLLAMA_CHAT_MODEL``, default ``llama3.2``).
    """
    if provider is not None and str(provider).strip():
        p = str(provider).strip().lower()
    else:
        p = (os.environ.get("LLM_PROVIDER") or "ollama").strip().lower()

    if p in ("gemini", "google", "google_genai"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "Gemini requires: pip install langchain-google-genai"
            ) from e
        key = _google_api_key()
        if not key:
            raise ValueError(
                "Set GOOGLE_API_KEY or GEMINI_API_KEY in the environment for LLM_PROVIDER=gemini."
            )
        gemini_model = (
            model
            or os.environ.get("GEMINI_MODEL")
            or os.environ.get("GOOGLE_GENAI_MODEL")
            or "gemini-2.0-flash"
        )
        return ChatGoogleGenerativeAI(
            model=gemini_model,
            google_api_key=key,
            temperature=temperature,
        )

    if p in ("zai", "glm", "zhipu", "bigmodel"):
        from agent.zai_chat import ChatZai

        key = _zai_api_key()
        if not key:
            raise ValueError(
                "Set ZAI_API_KEY (or BIGMODEL_API_KEY) in the environment for LLM_PROVIDER=zai."
            )
        zai_model = model or os.environ.get("ZAI_CHAT_MODEL") or "glm-4.7-flash"
        zai_base = (os.environ.get("ZAI_BASE_URL") or "").strip() or None
        return ChatZai(
            model=zai_model,
            api_key=key,
            temperature=temperature,
            base_url=zai_base,
        )

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
    """Embeddings for FAISS ingest and retrieval (must stay consistent across rebuilds)."""
    kwargs: dict = {"model": model or os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")}
    bu = _ollama_base_url(base_url)
    if bu:
        kwargs["base_url"] = bu
    return OllamaEmbeddings(**kwargs)
