"""
LangGraph workflow: structured LLM plan -> FAISS retrieval per step -> LLM synthesis (LCEL).

Planner sees **dynamic** document ids under ``vector_stores/`` (upload UUIDs + demo ids like ``company_a_q3``).
"""

from __future__ import annotations

import json
import operator
import os
import pprint
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agent.doc_index import planner_doc_catalog_text
from agent.llm import get_chat_llm
from tools.retriever_tool import retrieve_document_chunks

# --- Structured planning ---


class RetrievalStep(BaseModel):
    """One retrieval call against a single indexed document."""

    target_doc: str = Field(
        description="Exact document id: folder name under vector_stores/ (UUID string or e.g. company_a_q3).",
    )
    search_query: str = Field(
        description="Short, focused query for semantic search in that document only.",
    )


class ResearchPlan(BaseModel):
    """Planner output: ordered retrieval steps for the user's question."""

    steps: list[RetrievalStep] = Field(
        description="Steps to gather evidence; typically one or more documents when comparing.",
    )


SYNTH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a financial analyst writing for an executive audience.
Using ONLY the retrieved context below, write a detailed, professional comparative report in Markdown.
Include: executive summary, side-by-side metrics where possible, and brief commentary on trends.
If some information is missing from the context, say so explicitly. Do not invent figures.""",
        ),
        (
            "human",
            """User question:
{question}

Retrieved context (from indexed reports):
{context}
""",
        ),
    ]
)


def _plan_prompt() -> ChatPromptTemplate:
    catalog = planner_doc_catalog_text()
    system = f"""You are a research planner for multi-document Q&A.

Available chunk-indexed document ids (use exact strings as target_doc):
{catalog}

Decompose the user's question into retrieval steps. Each step targets ONE document with a
specific search_query tailored for vector retrieval (metrics, regions, margins, growth, etc.).
When comparing multiple sources, use separate steps per document. Output valid structured data only."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )


def _plan_chain():
    llm = get_chat_llm(temperature=0)
    structured = llm.with_structured_output(ResearchPlan)
    return _plan_prompt() | structured


def _synth_chain():
    llm = get_chat_llm(temperature=0.25)
    return SYNTH_PROMPT | llm | StrOutputParser()


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
    plan_steps: list[dict[str, str]]
    retrieved: Annotated[list[tuple[str, str]], operator.add]
    report: NotRequired[str]
    debug_trace: Annotated[list[dict[str, Any]], operator.add]


def _plan_node(state: GraphState) -> dict:
    q = state["query"]
    prompt = _plan_prompt()
    messages = prompt.format_messages(question=q)
    chain = _plan_chain()
    plan: ResearchPlan = chain.invoke({"question": q})
    steps = [s.model_dump() for s in plan.steps]

    entry = {
        "node": "plan",
        "state_in": {"query": q},
        "prompt_messages": _messages_to_trace(messages),
        "llm_output_structured": plan.model_dump(),
        "state_update": {"plan_steps": steps},
    }
    _emit_trace(entry)
    return {"plan_steps": steps, "debug_trace": [entry]}


def _action_node(state: GraphState) -> dict:
    retrieved: list[tuple[str, str]] = []
    calls: list[dict[str, Any]] = []
    for step in state["plan_steps"]:
        doc_id = step["target_doc"]
        sq = step["search_query"]
        payload = retrieve_document_chunks.invoke({"query": sq, "document_id": doc_id})
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
    chain = _synth_chain()
    context = _format_context(state["retrieved"])
    q = state["query"]
    messages = SYNTH_PROMPT.format_messages(question=q, context=context)
    report = chain.invoke({"question": q, "context": context})

    entry = {
        "node": "synthesize",
        "state_in": {
            "query": q,
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
    g.add_node("plan", _plan_node)
    g.add_node("retrieve", _action_node)
    g.add_node("synthesize", _synthesis_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", END)

    return g.compile()


def run_agent(query: str) -> GraphState:
    graph = build_graph()
    initial: GraphState = {
        "query": query,
        "plan_steps": [],
        "retrieved": [],
        "debug_trace": [],
    }
    return graph.invoke(initial)


def stream_agent_updates(query: str):
    """
    Stream LangGraph ``stream_mode='updates'`` events (one dict per completed node).

    Each yield is like ``{'plan': {'plan_steps': ..., 'debug_trace': [...]}}``.
    Use this for UIs that show trace rows as each step finishes.
    """
    graph = build_graph()
    initial: GraphState = {
        "query": query,
        "plan_steps": [],
        "retrieved": [],
        "debug_trace": [],
    }
    yield from graph.stream(initial, stream_mode="updates")


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
