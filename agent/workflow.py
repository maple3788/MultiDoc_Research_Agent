"""
LangGraph workflow: **summary Milvus routing** → chunk retrieval per document → LLM synthesis.

Document selection: embed the user query, search the document-level summary index in **Milvus**,
then intersect with ids that have chunk vectors in Milvus. No LLM "planner" for routing.
"""

from __future__ import annotations

import json
import operator
import os
import pprint
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from agent.llm import get_chat_llm
from tools.retriever_tool import retrieve_document_chunks

SYNTH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a research assistant answering from the **retrieved excerpts** below (your only source for factual claims about the documents).

**Grounding:** Paraphrase and summarize what the excerpts actually say. When the text supports a point, state it clearly (e.g. "The material explains that…", "According to the excerpt…"). Do not invent quotes, page numbers, or facts not supported by the excerpts.

**Tone:** Match the task—financial comparisons can be executive-style; books, memos, or policies should read like clear notes or study takeaways, not a financial filing.

**Avoid repetitive disclaimers:** Do **not** repeat "no specific examples in the retrieved context" under every bullet. If the excerpts are enough to answer, answer directly. If something is missing (e.g. the user asked for "chapter 1" but the excerpts do not clearly contain that section), say **once** in a short note at the end—not after every point.

**Figures:** Do not invent numbers; only use those present in the excerpts.""",
        ),
        (
            "human",
            """User question:
{question}

Retrieved context (document excerpts):
{context}
""",
        ),
    ]
)

OUT_OF_CATALOG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You assist users of a document-grounded research app.
The indexed document library does NOT apply to this question (no relevant documents were selected by summary search).

Rules:
- Do NOT invent company names, financial metrics, comparison tables, or pretend you read indexed reports.
- Do NOT write an executive-style comparative report unless the user's question clearly asks for that and it fits general knowledge alone.
- Briefly explain that no matching documents were found in the corpus for this question.
- Then either give a short, honest general-knowledge answer if the question allows it, or politely decline if an answer would need sources you do not have.
- Use Markdown. Keep the reply concise unless the user clearly asks for depth.""",
        ),
        ("human", "{question}"),
    ]
)


def _default_search_query(question: str) -> str:
    q = question.strip()
    return q[:800] if len(q) > 800 else q


def _synth_chain(llm_provider: str | None = None):
    llm = get_chat_llm(temperature=0.3, provider=llm_provider)
    return SYNTH_PROMPT | llm | StrOutputParser()


def _synth_chain_out_of_catalog(llm_provider: str | None = None):
    llm = get_chat_llm(temperature=0.25, provider=llm_provider)
    return OUT_OF_CATALOG_PROMPT | llm | StrOutputParser()


def _format_context(retrieved: list[tuple[str, str]]) -> str:
    blocks = []
    for doc_id, text in retrieved:
        blocks.append(f"#### {doc_id}\n{text}")
    return "\n\n".join(blocks) if blocks else "(No chunks retrieved.)"


TRACE_MAX_CHARS = 24_000


def _trunc(text: str, max_chars: int = TRACE_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n… [truncated, {len(text)} chars total]"


def _messages_to_trace(msgs: list[BaseMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in msgs:
        content = m.content
        if not isinstance(content, str):
            content = json.dumps(content, default=str)
        out.append({"role": getattr(m, "type", m.__class__.__name__), "content": _trunc(content, TRACE_MAX_CHARS)})
    return out


def _env_trace_print() -> bool:
    return os.environ.get("AGENT_TRACE_PRINT", "").strip().lower() in ("1", "true", "yes")


def _emit_trace(entry: dict[str, Any]) -> None:
    if _env_trace_print():
        print(f"\n=== LangGraph node: {entry.get('node')} ===", flush=True)
        pprint.pprint(entry, width=120)


class GraphState(TypedDict):
    query: str
    llm_provider: NotRequired[str]
    relevant_to_catalog: NotRequired[bool]
    skip_retrieval: NotRequired[bool]
    plan_steps: list[dict[str, str]]
    retrieved: Annotated[list[tuple[str, str]], operator.add]
    report: NotRequired[str]
    debug_trace: Annotated[list[dict[str, Any]], operator.add]


def _route_node(state: GraphState) -> dict:
    """Pick documents by embedding the query against summary Milvus, then chunk-index intersection."""
    from catalog.routing import route_query_to_documents

    q = state["query"]
    routed = route_query_to_documents(q)
    sq = _default_search_query(q)
    steps = [{"target_doc": doc_id, "search_query": sq} for doc_id, _dist in routed]
    relevant = len(steps) > 0
    skip_retrieval = not relevant

    entry = {
        "node": "route",
        "state_in": {"query": q},
        "summary_catalog_hits": [{"document_id": d, "l2": dist} for d, dist in routed],
        "state_update": {
            "plan_steps": steps,
            "relevant_to_catalog": relevant,
            "skip_retrieval": skip_retrieval,
        },
    }
    _emit_trace(entry)
    return {
        "plan_steps": steps,
        "relevant_to_catalog": relevant,
        "skip_retrieval": skip_retrieval,
        "debug_trace": [entry],
    }


def _route_after_plan(state: GraphState) -> Literal["retrieve", "synthesize"]:
    if state.get("skip_retrieval"):
        return "synthesize"
    return "retrieve"


def _action_node(state: GraphState) -> dict:
    retrieved: list[tuple[str, str]] = []
    calls: list[dict[str, Any]] = []
    for step in state["plan_steps"]:
        doc_id = step["target_doc"]
        sq = step["search_query"]
        try:
            payload = retrieve_document_chunks.invoke({"query": sq, "document_id": doc_id})
        except FileNotFoundError as e:
            payload = (
                f"[document_id={doc_id}]\n--- Retrieval error ---\n"
                f"No chunk vectors in Milvus for this document. Re-upload the file or run ingest.\n({e})"
            )
        retrieved.append((doc_id, payload))
        calls.append(
            {
                "target_doc": doc_id,
                "search_query": sq,
                "tool": "retrieve_document_chunks",
                "retrieved_text": _trunc(payload),
            }
        )

    entry = {
        "node": "retrieve",
        "state_in": {"plan_steps": state["plan_steps"]},
        "retrieval_calls": calls,
        "state_update": {"retrieved": [(d, _trunc(t)) for d, t in retrieved]},
    }
    _emit_trace(entry)
    return {"retrieved": retrieved, "debug_trace": [entry]}


def _synthesis_node(state: GraphState) -> dict:
    q = state["query"]
    relevant = state.get("relevant_to_catalog", True)
    llm_provider = state.get("llm_provider")

    if not relevant:
        chain = _synth_chain_out_of_catalog(llm_provider)
        messages = OUT_OF_CATALOG_PROMPT.format_messages(question=q)
        report = chain.invoke({"question": q})
        entry = {
            "node": "synthesize",
            "mode": "out_of_catalog",
            "llm_provider": llm_provider,
            "state_in": {
                "query": q,
                "relevant_to_catalog": False,
                "retrieved_pairs_count": 0,
            },
            "prompt_messages": _messages_to_trace(messages),
            "llm_output_text": _trunc(report, TRACE_MAX_CHARS),
            "state_update": {"report": _trunc(report, TRACE_MAX_CHARS)},
        }
    else:
        chain = _synth_chain(llm_provider)
        context = _format_context(state["retrieved"])
        messages = SYNTH_PROMPT.format_messages(question=q, context=context)
        report = chain.invoke({"question": q, "context": context})
        entry = {
            "node": "synthesize",
            "mode": "grounded",
            "llm_provider": llm_provider,
            "state_in": {
                "query": q,
                "relevant_to_catalog": True,
                "retrieved_pairs_count": len(state["retrieved"]),
            },
            "prompt_messages": _messages_to_trace(messages),
            "llm_output_text": _trunc(report, TRACE_MAX_CHARS),
            "state_update": {"report": _trunc(report, TRACE_MAX_CHARS)},
        }
    _emit_trace(entry)
    return {"report": report, "debug_trace": [entry]}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("route", _route_node)
    g.add_node("retrieve", _action_node)
    g.add_node("synthesize", _synthesis_node)

    g.set_entry_point("route")
    g.add_conditional_edges(
        "route",
        _route_after_plan,
        {"retrieve": "retrieve", "synthesize": "synthesize"},
    )
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", END)

    return g.compile()


def _initial_state(query: str, llm_provider: str | None = None) -> GraphState:
    state: GraphState = {
        "query": query,
        "relevant_to_catalog": True,
        "skip_retrieval": False,
        "plan_steps": [],
        "retrieved": [],
        "debug_trace": [],
    }
    if llm_provider is not None:
        state["llm_provider"] = llm_provider
    return state


def run_agent(query: str, *, llm_provider: str | None = None) -> GraphState:
    graph = build_graph()
    return graph.invoke(_initial_state(query, llm_provider))


def stream_agent_updates(query: str, *, llm_provider: str | None = None):
    """
    Stream LangGraph ``stream_mode='updates'`` events (one dict per completed node).

    Each yield is like ``{'route': {'plan_steps': ..., 'debug_trace': [...]}}``.
    Optional ``llm_provider`` (``ollama`` / ``gemini`` / ``zai``) overrides ``LLM_PROVIDER`` for synthesis.
    """
    graph = build_graph()
    yield from graph.stream(_initial_state(query, llm_provider), stream_mode="updates")


def get_agent_flow_assets() -> tuple[bytes, str]:
    """PNG (rendered diagram) and Mermaid source from a single compiled graph build."""
    g = build_graph().get_graph()
    return g.draw_mermaid_png(), g.draw_mermaid()


def get_agent_flow_mermaid() -> str:
    """Mermaid source for the compiled graph (for UIs and docs)."""
    return get_agent_flow_assets()[1]


def get_agent_flow_png() -> bytes:
    """PNG image of the graph (Mermaid rendered by LangGraph)."""
    return get_agent_flow_assets()[0]
