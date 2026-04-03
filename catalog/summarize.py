"""Summarize raw text with Ollama (LCEL) for catalog routing."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.llm import get_chat_llm

MAX_CHARS = 48_000

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You summarize documents for a retrieval catalog.
Write a concise English summary (roughly 120–220 words unless the source is tiny).
Include: main topic, key entities (companies, products), time period, and any numbers or metrics mentioned.
Do not invent facts. If the input is not readable text, say so briefly.""",
        ),
        ("human", "{text}"),
    ]
)


def summarize_for_catalog(text: str, *, max_chars: int = MAX_CHARS) -> str:
    """Produce a routing summary suitable for embedding."""
    clipped = text.strip()
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars] + "\n\n[...truncated...]"
    llm = get_chat_llm(temperature=0.15)
    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    return chain.invoke({"text": clipped}).strip()
