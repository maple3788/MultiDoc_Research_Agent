"""
Create dummy Q3 reports, chunk them, and build per-document FAISS indexes under vector_stores/.

Run once (or after changing source text):

  python ingest.py
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.llm import get_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_ROOT = PROJECT_ROOT / "vector_stores"

# Rich enough for multi-chunk indexes; content is fictional.
COMPANY_A_TEXT = """\
Company A — Quarterly Financial Summary (Q3 FY2026)

Executive overview: Company A delivered solid Q3 results with revenue of $120.4 million,
representing 8% year-over-year growth. Management attributed growth to enterprise SaaS
adoption and expansion within existing accounts. Gross margin held steady at 62%.

Profitability: Operating income was $19.5 million; net margin for the quarter was 14%.
Adjusted EBITDA margin improved sequentially due to lower sales and marketing spend as a
percentage of revenue.

Segment notes: The Platform segment grew 11% YoY; Services grew 4%. The company highlighted
wins in regulated industries and emphasized multi-year commitments from large customers.

Risks and outlook: Leadership noted currency headwinds and competitive pricing pressure in
mid-market deals. Guidance for Q4 assumes mid-single-digit sequential revenue growth.
"""

COMPANY_B_TEXT = """\
Company B — Q3 FY2026 Results and Commentary

Revenue: Company B reported Q3 revenue of $95.2 million, up 12% compared to the prior-year
quarter. Growth was driven by APAC expansion, new logo acquisition, and improved net revenue
retention in the core product suite.

Margins and costs: Net margin for Q3 was 11%. Gross margin improved modestly as the company
scaled infrastructure and shifted mix toward higher-margin software lines. Operating expenses
rose in line with headcount additions in engineering and customer success.

Regional highlights: APAC revenue increased sharply; EMEA was flat; Americas remained the
largest region. Management called out improved unit economics and shorter sales cycles in
selected verticals.

Forward look: For Q4, the company expects continued YoY growth with investments in R&D
temporarily pressuring operating margin.
"""


def _write_sources() -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "company_a_q3": DATA_DIR / "company_a_q3.txt",
        "company_b_q3": DATA_DIR / "company_b_q3.txt",
    }
    paths["company_a_q3"].write_text(COMPANY_A_TEXT.strip() + "\n", encoding="utf-8")
    paths["company_b_q3"].write_text(COMPANY_B_TEXT.strip() + "\n", encoding="utf-8")
    return paths


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
    """Chunk ``text``, embed, and save ``vector_stores/<doc_id>/`` (used for uploads)."""
    embeddings = get_embeddings()
    build_faiss_for_doc(doc_id, text, embeddings)


def run_ingest() -> None:
    paths = _write_sources()
    embeddings = get_embeddings()
    VECTOR_ROOT.mkdir(parents=True, exist_ok=True)

    for doc_id, path in paths.items():
        text = path.read_text(encoding="utf-8")
        build_faiss_for_doc(doc_id, text, embeddings)
        print(f"Indexed {doc_id} -> {VECTOR_ROOT / doc_id}")

    from catalog.pipeline import rebuild_summary_catalog

    rebuild_summary_catalog()
    print(f"Updated summary catalog -> {PROJECT_ROOT / 'catalog_store'}")


def main() -> None:
    run_ingest()


if __name__ == "__main__":
    main()
