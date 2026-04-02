"""
LangGraph workflow: Plan -> Retrieve (per document) -> Synthesize.

Retrieval uses the mock `retrieve_document_chunks` tool. Swap synthesis for an LLM call when ready.
"""

from __future__ import annotations

import operator
from typing import Annotated, NotRequired, TypedDict

from langgraph.graph import END, StateGraph

from tools.retriever_tool import retrieve_document_chunks


class GraphState(TypedDict):
    """Shared state for the multi-document research graph."""

    query: str
    plan: str
    targets: list[str]
    retrieved: Annotated[list[tuple[str, str]], operator.add]
    report: NotRequired[str]


def _plan_node(state: GraphState) -> dict:
    """
    Break down the comparative question into retrieval targets.

    Placeholder logic: map known comparative phrasing to mock document ids.
    Replace with an LLM planner for open-ended questions.
    """
    q = state["query"].lower()
    if "company a" in q and "company b" in q:
        targets = ["company_a_q3", "company_b_q3"]
        plan = (
            "Steps: (1) Fetch Q3 financial context for Company A and Company B. "
            "(2) Align metrics (revenue, margin, narrative). (3) Produce a concise comparison."
        )
    else:
        targets = ["company_a_q3", "company_b_q3"]
        plan = "Default plan: retrieve both default corpora and synthesize a comparison."

    return {"plan": plan, "targets": targets}


def _action_node(state: GraphState) -> dict:
    """Call the retriever tool once per target document."""
    retrieved: list[tuple[str, str]] = []
    q = state["query"]
    for doc_id in state["targets"]:
        payload = retrieve_document_chunks.invoke({"query": q, "document_id": doc_id})
        retrieved.append((doc_id, payload))
    return {"retrieved": retrieved}


def _synthesis_node(state: GraphState) -> dict:
    """Combine retrieved chunks into a structured report (placeholder, no LLM)."""
    lines = [
        "# Comparative research report",
        "",
        f"**Original question:** {state['query']}",
        "",
        "## Plan",
        state["plan"],
        "",
        "## Retrieved context",
    ]
    for doc_id, text in state["retrieved"]:
        lines.append(f"### {doc_id}")
        lines.append(text)
        lines.append("")

    lines.extend(
        [
            "## Synthesis (placeholder)",
            "Side-by-side: Company A reported higher absolute revenue in Q3; Company B showed stronger YoY revenue growth. "
            "Margins were higher for Company A; Company B emphasized regional expansion. "
            "Replace this section with an LLM that cites the chunks above.",
        ]
    )
    report = "\n".join(lines)
    return {"report": report}


def build_graph():
    """Compile the LangGraph state machine."""
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
    """Run the graph end-to-end."""
    graph = build_graph()
    initial: GraphState = {
        "query": query,
        "plan": "",
        "targets": [],
        "retrieved": [],
    }
    return graph.invoke(initial)
