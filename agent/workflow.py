"""
LangGraph workflow: structured LLM plan -> FAISS retrieval per step -> LLM synthesis (LCEL).
"""

from __future__ import annotations

import operator
from typing import Annotated, NotRequired, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agent.llm import get_chat_llm
from tools.retriever_tool import retrieve_document_chunks

# --- Structured planning ---


class RetrievalStep(BaseModel):
    """One retrieval call against a single indexed document."""

    target_doc: str = Field(
        description="Document id: company_a_q3 or company_b_q3 (must match vector_stores/)."
    )
    search_query: str = Field(
        description="Short, focused query for semantic search in that document only."
    )


class ResearchPlan(BaseModel):
    """Planner output: ordered retrieval steps for the user's question."""

    steps: list[RetrievalStep] = Field(
        description="Steps to gather evidence; typically one or more per company/report."
    )


PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a research planner for multi-document Q&A.
Available indexed document ids (use these exact strings):
- company_a_q3 — Company A Q3 FY2026 report
- company_b_q3 — Company B Q3 FY2026 report

Decompose the user's question into retrieval steps. Each step targets ONE document with a
specific search_query tailored for vector retrieval (metrics, regions, margins, growth, etc.).
Prefer separate steps per company when comparing. Output valid structured data only.""",
        ),
        ("human", "{question}"),
    ]
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


def _plan_chain():
    llm = get_chat_llm(temperature=0)
    structured = llm.with_structured_output(ResearchPlan)
    return PLAN_PROMPT | structured


def _synth_chain():
    llm = get_chat_llm(temperature=0.25)
    return SYNTH_PROMPT | llm | StrOutputParser()


def _format_context(retrieved: list[tuple[str, str]]) -> str:
    blocks = []
    for doc_id, text in retrieved:
        blocks.append(f"#### {doc_id}\n{text}")
    return "\n\n".join(blocks) if blocks else "(No chunks retrieved.)"


class GraphState(TypedDict):
    query: str
    plan_steps: list[dict[str, str]]
    retrieved: Annotated[list[tuple[str, str]], operator.add]
    report: NotRequired[str]


def _plan_node(state: GraphState) -> dict:
    chain = _plan_chain()
    plan: ResearchPlan = chain.invoke({"question": state["query"]})
    steps = [s.model_dump() for s in plan.steps]
    return {"plan_steps": steps}


def _action_node(state: GraphState) -> dict:
    retrieved: list[tuple[str, str]] = []
    for step in state["plan_steps"]:
        doc_id = step["target_doc"]
        sq = step["search_query"]
        payload = retrieve_document_chunks.invoke({"query": sq, "document_id": doc_id})
        retrieved.append((doc_id, payload))
    return {"retrieved": retrieved}


def _synthesis_node(state: GraphState) -> dict:
    chain = _synth_chain()
    context = _format_context(state["retrieved"])
    report = chain.invoke({"question": state["query"], "context": context})
    return {"report": report}


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
    }
    return graph.invoke(initial)
