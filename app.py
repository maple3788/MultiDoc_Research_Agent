"""
Streamlit UI: sidebar navigation; main area shows the selected tool.

Research agent uses LangGraph ``stream_mode='updates'`` so each node's trace is rendered
as that step completes (same script pass; see sidebar note on Streamlit rendering).

Run: ``streamlit run app.py`` (use project ``.venv``).
"""

from __future__ import annotations

import os
import uuid
from io import BytesIO

import streamlit as st
from sqlalchemy import select

from agent.workflow import get_agent_flow_assets, get_agent_flow_mermaid, stream_agent_updates
from catalog.ivf_pq_faiss import search_catalog
from catalog.pipeline import delete_document, ingest_bytes, list_all_documents
from db.models import Document
from db.session import SessionLocal, init_db
PAGES = ["Library", "Upload", "Search catalog", "Agent flow", "Research agent"]

# Max prior messages (user+assistant pairs) to inject into the agent prompt for follow-ups.
_RESEARCH_CONTEXT_MAX_MESSAGES = 8


def _init_state() -> None:
    if "db_ready" not in st.session_state:
        st.session_state.db_ready = False
    if "research_chat_messages" not in st.session_state:
        st.session_state.research_chat_messages = []
    if "research_last_trace" not in st.session_state:
        st.session_state.research_last_trace = []
    if "research_run_pending" not in st.session_state:
        st.session_state.research_run_pending = False
    if "research_llm_provider" not in st.session_state:
        env_p = (os.environ.get("LLM_PROVIDER") or "ollama").strip().lower()
        if env_p in ("gemini", "google", "google_genai"):
            st.session_state.research_llm_provider = "gemini"
        elif env_p in ("zai", "glm", "zhipu", "bigmodel"):
            st.session_state.research_llm_provider = "zai"
        else:
            st.session_state.research_llm_provider = "ollama"


def build_research_agent_query(messages: list[dict[str, str]]) -> str:
    """Turn chat history + latest user turn into one prompt for the graph (session memory)."""
    if not messages:
        return ""
    current = messages[-1]["content"].strip()
    if messages[-1]["role"] != "user":
        return current
    prior = messages[:-1]
    if not prior:
        return current
    tail = prior[-_RESEARCH_CONTEXT_MAX_MESSAGES:]
    lines: list[str] = []
    for m in tail:
        label = "User" if m["role"] == "user" else "Assistant"
        text = (m.get("content") or "").strip()
        if len(text) > 4000:
            text = text[:4000] + "…"
        lines.append(f"{label}: {text}")
    return "Conversation so far:\n" + "\n".join(lines) + f"\n\nCurrent question:\n{current}"


@st.cache_data(show_spinner=False)
def _cached_agent_flow() -> tuple[bytes, str]:
    return get_agent_flow_assets()


def _ensure_database() -> bool:
    if st.session_state.db_ready:
        return True
    try:
        init_db()
        st.session_state.db_ready = True
        return True
    except Exception as e:
        st.error(
            "Could not connect to PostgreSQL or create tables. "
            "Start Postgres (e.g. `docker compose up -d`) and set `DATABASE_URL` in `.env`.\n\n"
            f"Details: `{e}`"
        )
        return False


st.set_page_config(
    page_title="Multi-Doc Research",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

_init_state()

st.markdown(
    """
<style>
    .main-header { font-size: 1.75rem; font-weight: 650; margin-bottom: 0.25rem; }
    .subtle { color: #666; font-size: 0.95rem; }
    /* Research agent: chat history scrolls above the full-width chat input */
    div[data-testid="stChatInput"] { position: sticky; bottom: 0; z-index: 2; }
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.title("Navigate")
page = st.sidebar.radio(
    "Page",
    PAGES,
    label_visibility="collapsed",
    key="nav_page",
)

st.sidebar.caption(
    "Uploads build **summary** (`catalog_store/`) and **chunk** FAISS (`vector_stores/<id>/`). "
    "On **Research agent**, **Chat LLM** can use Ollama, Gemini, or Z.ai GLM (API keys in `.env`)."
)

st.sidebar.caption(
    "Research agent: **chat** on the left, **trace** on the right (refreshes each run). "
    "History is kept for this browser session until you clear it or reload."
)

if page == "Research agent":
    st.sidebar.selectbox(
        "Chat LLM",
        options=["ollama", "gemini", "zai"],
        format_func=lambda x: {
            "ollama": "Ollama (local)",
            "gemini": "Gemini (Google AI)",
            "zai": "GLM-4.7-Flash (Z.ai)",
        }[x],
        key="research_llm_provider",
        help="Overrides `LLM_PROVIDER` for synthesis. Gemini: `GOOGLE_API_KEY` / `GEMINI_API_KEY`. Z.ai: `ZAI_API_KEY` and `pip install zai-sdk`.",
    )
    if st.sidebar.button("Clear chat & trace", use_container_width=True):
        st.session_state.research_chat_messages = []
        st.session_state.research_last_trace = []
        st.session_state.research_run_pending = False
        st.rerun()

if not _ensure_database():
    st.stop()

# ----- Main area (right) -----
if page == "Library":
    st.markdown('<p class="main-header">Document library</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtle">Review uploads, summaries, and chunk-index notes. Remove deletes files, chunk folders, and rebuilds the summary catalog.</p>',
        unsafe_allow_html=True,
    )

    docs = list_all_documents()
    if not docs:
        st.info('No documents yet. Choose **Upload** in the sidebar.')
    else:
        for doc in docs:
            status_emoji = {"ready": "✅", "failed": "⚠️", "processing": "⏳"}.get(doc.status, "•")
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"**{doc.original_filename}**  {status_emoji} `{doc.status}`")
                    st.caption(f"id: `{doc.id}` · uploaded {doc.created_at}")
                    if doc.status == "failed" and doc.error_message:
                        st.error(doc.error_message[:500])
                    elif doc.status == "ready" and doc.error_message:
                        st.warning(doc.error_message[:800])
                    if doc.summary and doc.status != "failed":
                        with st.expander("Summary (catalog embedding)"):
                            st.write(doc.summary)
                with c2:
                    if st.button("Remove", key=f"delete_{doc.id}", type="secondary"):
                        if delete_document(doc.id):
                            st.rerun()
                        else:
                            st.warning("Document not found.")

elif page == "Upload":
    st.markdown('<p class="main-header">Upload files</p>', unsafe_allow_html=True)
    st.markdown(
        "<p class='subtle'>Files go to <code>uploads/</code>, are summarized (<strong>llama3.2</strong>), "
        "stored in PostgreSQL, and both the <strong>summary catalog</strong> and <strong>chunk</strong> indexes "
        "under <code>vector_stores/&lt;document-id&gt;/</code> are updated. "
        "<strong>PDF</strong> text is extracted automatically (image-only PDFs need OCR elsewhere).</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Choose one or more files",
        type=["txt", "md", "csv", "json", "pdf"],
        accept_multiple_files=True,
        help="Text: UTF-8. PDF: text is extracted with pypdf (scanned-only PDFs may fail).",
    )
    if uploaded:
        if st.button("Process uploads", type="primary"):
            for f in uploaded:
                try:
                    data = f.getvalue()
                    with st.spinner(f"Processing **{f.name}**…"):
                        ingest_bytes(f.name, data)
                    st.success(f"Indexed: **{f.name}**")
                except Exception as e:
                    st.error(f"{f.name}: {e}")
            st.rerun()

elif page == "Search catalog":
    st.markdown('<p class="main-header">Search catalog</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtle">Semantic search over **summaries** (doc-level vectors).</p>',
        unsafe_allow_html=True,
    )

    q = st.text_input("Query", placeholder="e.g. Q3 revenue margins APAC")
    k = st.slider("Top‑K documents", 1, 20, 5)
    if st.button("Search", type="primary") and q.strip():
        with st.spinner("Searching…"):
            hits = search_catalog(q.strip(), k=k)
        if not hits:
            st.warning("No results. Upload documents first or check `catalog_store/`.")
        else:
            session = SessionLocal()
            try:
                for uuid_str, dist in hits:
                    uid = uuid.UUID(uuid_str)
                    row = session.scalar(select(Document).where(Document.id == uid))
                    title = row.original_filename if row else uuid_str
                    with st.expander(f"**{title}** — L2 `{dist:.4f}`"):
                        if row and row.summary:
                            st.write(row.summary)
                        else:
                            st.caption("No summary in database.")
            finally:
                session.close()

elif page == "Agent flow":
    st.markdown('<p class="main-header">Agent flow (static graph)</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtle">Compiled <strong>LangGraph</strong> topology. For live prompts and tool I/O, use <strong>Research agent</strong>.</p>',
        unsafe_allow_html=True,
    )
    try:
        png_bytes, mermaid_src = _cached_agent_flow()
        st.image(BytesIO(png_bytes), caption="LangGraph · route (summary FAISS) → retrieve → synthesize")
        with st.expander("Mermaid source"):
            st.code(mermaid_src, language="text")
        st.download_button(
            label="Download flowchart PNG",
            data=png_bytes,
            file_name="agent_graph.png",
            mime="image/png",
        )
    except Exception as e:
        st.warning(f"Could not render graph image ({e}).")
        try:
            st.code(get_agent_flow_mermaid(), language="text")
        except Exception as e2:
            st.error(str(e2))

elif page == "Research agent":
    st.markdown('<p class="main-header">Multi-document research agent</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtle">Scroll the <strong>chat</strong> panel to read history; the input stays at the bottom. '
        "<strong>Trace</strong> on the right updates each run. "
        "Plan → chunk retrieval per <code>document_id</code> → synthesis; prior turns are context for follow-ups.</p>",
        unsafe_allow_html=True,
    )

    chat_col, trace_col = st.columns([1.65, 1], gap="large")

    # Left: fixed-height scrollable history (input is below, full width).
    with chat_col:
        history = st.container(height=560, border=True, autoscroll=True)
        with history:
            if not st.session_state.research_chat_messages:
                with st.chat_message("assistant"):
                    st.markdown(
                        "Ask a question about your uploaded documents. "
                        "Example: *Compare Q3 results for Company A vs Company B.*"
                    )
            for msg in st.session_state.research_chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

    # Right: trace (render before submit handler so rerun refreshes it).
    with trace_col:
        st.markdown("##### Trace")
        st.caption("Latest run only · updates after each message")
        trace_box = st.container(height=560, border=True)
        with trace_box:
            steps = st.session_state.research_last_trace
            if not steps:
                st.info("Send a message to see LangGraph node traces here.", icon="➡️")
            else:
                for i, step in enumerate(steps, start=1):
                    node = step.get("node", "?")
                    entry = step.get("entry", step)
                    with st.expander(f"{i}. `{node}`", expanded=(i == len(steps))):
                        st.json(entry)

    # Next pass runs the agent so the user message appears in the scroll area first (no bubbles below the input).
    if st.session_state.research_run_pending:
        agent_query = build_research_agent_query(st.session_state.research_chat_messages)
        trace_steps: list[dict[str, object]] = []
        report_final: str | None = None
        err: str | None = None

        try:
            with st.status("Running agent…", expanded=True):
                for event in stream_agent_updates(
                    agent_query,
                    llm_provider=st.session_state.research_llm_provider,
                ):
                    for node_name, upd in event.items():
                        for entry in upd.get("debug_trace", []):
                            trace_steps.append({"node": node_name, "entry": entry})
                        if node_name == "synthesize" and upd.get("report"):
                            report_final = upd["report"]
        except Exception as e:
            err = str(e)

        st.session_state.research_last_trace = trace_steps

        if err:
            reply = f"**Error:** {err}"
        elif report_final:
            reply = report_final
        else:
            reply = "_No report was produced. Check indexes and the trace panel._"

        st.session_state.research_chat_messages.append({"role": "assistant", "content": reply})
        st.session_state.research_run_pending = False
        st.rerun()

    # Full-width input bar at the bottom; history scrolls in the column above.
    if prompt := st.chat_input("Ask about your documents…"):
        user_text = prompt.strip()
        if user_text:
            st.session_state.research_chat_messages.append({"role": "user", "content": user_text})
            st.session_state.research_last_trace = []
            st.session_state.research_run_pending = True
            st.rerun()

else:
    st.error("Unknown page.")
