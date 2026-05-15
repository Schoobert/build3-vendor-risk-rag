import os
import sys
import tempfile
from pathlib import Path

import chromadb
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from embed import embed_chunks
from ingest import ingest_pdf
from risk_summary import analyze_vendor_risk
from vectorstore import CHROMA_DIR, store_chunks

COLLECTION_NAME = "vendor_risk"

SAMPLE_PDF = Path(__file__).parent.parent / "sample_data" / "NovaSoft_SOC2_TypeII_2024.pdf"
SAMPLE_LABEL = "NovaSoft_SOC2_TypeII_2024.pdf"
SAMPLE_DISPLAY = "NovaSoft Technologies SOC 2 Type II (2024)"

RATING_COLORS = {
    "low": "green",
    "medium": "orange",
    "high": "red",
    "critical": "red",
}


def _rating_color(rating: str) -> str:
    return RATING_COLORS.get(rating.lower(), "gray")


def _reset_collection() -> None:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass


def _ingest_file(uploaded_file) -> int:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        _reset_collection()
        chunks = ingest_pdf(tmp_path)
        embedded = embed_chunks(chunks)
        store_chunks(embedded, COLLECTION_NAME, source_filename=uploaded_file.name)
        return len(chunks)
    finally:
        os.unlink(tmp_path)


def _ingest_sample() -> int:
    _reset_collection()
    chunks = ingest_pdf(str(SAMPLE_PDF))
    embedded = embed_chunks(chunks)
    store_chunks(embedded, COLLECTION_NAME, source_filename=SAMPLE_LABEL)
    return len(chunks)


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Vendor Risk Assessment Tool", layout="wide")
st.title("Vendor Risk Assessment Tool")

# ── Session state ─────────────────────────────────────────────────────────────

for key, default in [
    ("ingested_file", None),
    ("chunk_count", 0),
    ("last_question", None),
    ("last_result", None),
    ("use_sample", False),
    ("pending_question", ""),
    ("auto_run", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    st.caption(
        "Upload a SOC 2 report, security questionnaire, or vendor privacy policy"
    )

# ── Main area ─────────────────────────────────────────────────────────────────

# Uploaded file always takes priority over sample mode
if uploaded_file is not None:
    st.session_state.use_sample = False

if uploaded_file is not None:
    # ── Uploaded file branch ──────────────────────────────────────────────────
    if st.session_state.ingested_file != uploaded_file.name:
        with st.spinner("Analyzing document..."):
            chunk_count = _ingest_file(uploaded_file)
        st.session_state.ingested_file = uploaded_file.name
        st.session_state.chunk_count = chunk_count
        st.session_state.last_question = None
        st.session_state.last_result = None

    st.success(
        f"Ready: **{uploaded_file.name}** — {st.session_state.chunk_count} chunks indexed"
    )

elif st.session_state.use_sample:
    # ── Sample document branch ────────────────────────────────────────────────
    if st.session_state.ingested_file != SAMPLE_LABEL:
        with st.spinner("Loading sample document..."):
            chunk_count = _ingest_sample()
        st.session_state.ingested_file = SAMPLE_LABEL
        st.session_state.chunk_count = chunk_count
        st.session_state.last_question = None
        st.session_state.last_result = None

    st.info(
        f"Using sample document: **{SAMPLE_DISPLAY}** — "
        "or upload your own PDF using the sidebar"
    )

else:
    # ── Landing page ──────────────────────────────────────────────────────────
    st.markdown(
        """
        ### What this tool does

        This tool helps you assess vendor risk by analyzing vendor-provided security
        documents using AI-powered retrieval and language models.

        **How it works:**
        1. Upload a vendor security document using the sidebar
        2. The document is parsed, chunked, and stored in a vector database
        3. Ask any risk-related question about the vendor
        4. Get a structured risk assessment — rating, summary, citations, and
           recommended action

        **Supported document types:**
        - SOC 2 Type I / Type II audit reports
        - Security questionnaires (VSA, SIG, CAIQ)
        - Vendor privacy policies and data processing agreements
        """
    )

    if st.button("Try with sample document", type="primary"):
        st.session_state.use_sample = True
        st.rerun()

SUGGESTED_QUESTIONS = [
    "What penetration test or vulnerability findings are open or overdue?",
    "What is the vendor's backup and recovery posture?",
    "How is customer data encrypted and protected?",
    "What access controls and authentication requirements are in place?",
    "Are there any exceptions or audit findings I should be aware of?",
]

# ── Q&A panel (shown whenever a document is ready) ────────────────────────────

if st.session_state.ingested_file:
    st.markdown("### Ask a Risk Question")

    # If a suggestion was clicked on the previous run, pre-fill the input and
    # arm auto_run before the widget renders so the value takes effect.
    if st.session_state.pending_question:
        st.session_state.question_input = st.session_state.pending_question
        st.session_state.pending_question = ""
        st.session_state.auto_run = True

    # Consume auto_run immediately so a subsequent rerun doesn't re-fire it.
    should_auto_run = st.session_state.auto_run
    st.session_state.auto_run = False

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        question = st.text_input(
            "Question",
            key="question_input",
            label_visibility="collapsed",
            placeholder="e.g. What security exceptions or audit findings were identified?",
        )
    with col_btn:
        submitted = st.button("Analyze", use_container_width=True)

    st.caption("Suggested questions:")
    sug_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for i, suggestion in enumerate(SUGGESTED_QUESTIONS):
        with sug_cols[i]:
            if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()

    if (submitted or should_auto_run) and question:
        if question != st.session_state.last_question:
            with st.spinner("Generating risk summary..."):
                st.session_state.last_result = analyze_vendor_risk(question)
                st.session_state.last_question = question

    if st.session_state.last_result:
        result = st.session_state.last_result
        rating = result.get("risk_rating", "Unknown")
        color = _rating_color(rating)

        st.markdown("---")

        st.markdown(f"#### Risk Rating: :{color}[{rating}]")

        st.markdown("**Summary**")
        st.write(result.get("summary", "N/A"))

        st.markdown("**Citations**")
        st.write(result.get("citations", "N/A"))

        st.markdown("**Recommended Action**")
        st.write(result.get("recommended_action", "N/A"))
