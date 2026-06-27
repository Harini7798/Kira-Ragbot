"""Measurable RAG — local web UI (Streamlit).

Three sources (sidebar):
  * 📄 My documents — upload PDFs/text; indexed in memory (the default).
  * 🌐 Web search — search the web live and answer from the top pages.
  * 🧪 SciFact benchmark — 5,183 pre-loaded labeled abstracts (the measured demo).

Whichever you pick, you get the full pipeline: retrieved sources, an answer with
citations, each citation VERIFIED against its source (supporting span
highlighted), the abstention decision, and live metrics.

Run:  streamlit run app.py
Needs a Groq API key (sidebar, or GROQ_API_KEY in the environment).
"""
from __future__ import annotations

import hashlib
import html
import os

# Use the already-downloaded models without a slow network check.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import streamlit as st

from measurable_rag import config
from measurable_rag.data.user_docs import load_documents
from measurable_rag.generation.abstain import grounding_score, should_abstain
from measurable_rag.generation.generator import Generator
from measurable_rag.generation.verify import Verifier, citation_precision, faithfulness
from measurable_rag.retrieval.build import build_retriever
from measurable_rag.retrieval.dense import DenseIndex
from measurable_rag.retrieval.embedder import Embedder
from measurable_rag.retrieval.pipeline import HybridRetriever
from measurable_rag.retrieval.rerank import Reranker
from measurable_rag.retrieval.sparse import SparseIndex
from measurable_rag.retrieval.web import search_web

st.set_page_config(page_title="Measurable RAG", layout="wide")

DOCS, WEB, SCIFACT = "📄 My documents", "🌐 Web search", "🧪 SciFact benchmark"


@st.cache_resource(show_spinner="Loading embedder (once)...")
def load_embedder():
    return Embedder()


@st.cache_resource(show_spinner="Loading SciFact index (once)...")
def load_scifact(_embedder):
    index = DenseIndex.load(config.INDEX_DIR)
    return index, SparseIndex(index.chunks)


@st.cache_resource(show_spinner="Loading reranker (once)...")
def load_reranker():
    return Reranker()


@st.cache_resource(show_spinner="Indexing your documents...")
def build_user_retriever(_embedder, _reranker, key, _payloads):
    docs = load_documents(_payloads)
    retriever, n_chunks = build_retriever(docs, _embedder, _reranker)
    return retriever, n_chunks, len(docs)


@st.cache_resource(show_spinner="Searching the web + indexing results...")
def build_web_retriever(_embedder, _reranker, query, max_results):
    docs = search_web(query, max_results=max_results)
    if not docs:
        return None, 0, 0
    retriever, n_chunks = build_retriever(docs, _embedder, _reranker)
    return retriever, n_chunks, len(docs)


def highlight(text: str, quotes: list[str]) -> str:
    escaped = html.escape(text)
    for q in quotes:
        if q:
            eq = html.escape(q)
            escaped = escaped.replace(eq, f"<mark>{eq}</mark>")
    return escaped.replace("\n", "<br>")


def render_result(query, chunks, api_key, threshold):
    os.environ["GROQ_API_KEY"] = api_key
    with st.spinner("Generating answer..."):
        answer = Generator().answer(query, chunks)
    refused = answer.strip().startswith("INSUFFICIENT")
    with st.spinner("Verifying citations..."):
        verdicts = [] if refused else Verifier().verify(answer, chunks)
    abstain = should_abstain(answer, verdicts, threshold)
    ground = grounding_score(verdicts)

    left, right = st.columns([3, 2])
    with left:
        if abstain:
            st.error("🚫 **Abstained** — not enough grounded evidence to answer reliably.")
            if not refused:
                st.caption(f"Grounding {ground:.0%} is below the {threshold:.0%} threshold.")
        else:
            st.success("✅ **Answered** (grounded in retrieved sources)")
        st.markdown("### Answer")
        st.write(answer)
        if verdicts:
            st.markdown("### Citation checks")
            for v in verdicts:
                icon = "✅" if v.supported else "❌"
                st.markdown(f"{icon} **[{v.source_num}]** (`{v.doc_id}`) — {v.claim}")
            cp = citation_precision(verdicts)
            c1, c2, c3 = st.columns(3)
            c1.metric("Citation precision", f"{cp:.0%}" if cp is not None else "n/a")
            c2.metric("Faithfulness", f"{faithfulness(verdicts):.0%}")
            c3.metric("Grounding", f"{ground:.0%}")
    with right:
        st.markdown("### Sources")
        quotes: dict[int, list[str]] = {}
        for v in verdicts:
            if v.supported and v.quote:
                quotes.setdefault(v.source_num, []).append(v.quote)
        for i, c in enumerate(chunks, start=1):
            with st.expander(f"[{i}] {c.doc_id}", expanded=i in quotes):
                st.markdown(highlight(c.text, quotes.get(i, [])), unsafe_allow_html=True)


# --- sidebar ----------------------------------------------------------------
st.sidebar.header("Settings")
corpus = st.sidebar.radio("Source", [DOCS, WEB, SCIFACT])  # docs-first by default
api_key = st.sidebar.text_input("Groq API key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
mode = st.sidebar.selectbox("Retrieval", ["hybrid (dense+BM25)", "dense only", "hybrid + rerank"])
top_k = st.sidebar.slider("Sources (top-k)", 3, 10, config.GENERATION_TOP_K)
max_web = st.sidebar.slider("Web pages to fetch", 3, 8, 5)
threshold = st.sidebar.slider("Abstention grounding threshold", 0.0, 1.0, float(config.ABSTAIN_THRESHOLD), 0.05)
st.sidebar.caption("Abstains if grounding is below this, or the model flags the sources as insufficient.")

st.title("Measurable RAG")
st.caption("Ask questions over your documents or the web — with every citation verified against its source, and honest refusal when the answer isn't there.")

embedder = load_embedder()
reranker = load_reranker() if mode == "hybrid + rerank" else None


def retrieve(retriever, query):
    if mode == "dense only":
        return retriever.dense(query, top_k)
    if mode == "hybrid + rerank":
        return retriever.hybrid_reranked(query, top_k)
    return retriever.hybrid(query, top_k)


# --- per-source flow --------------------------------------------------------
if corpus == DOCS:
    uploads = st.file_uploader("Upload documents (PDF, TXT, MD)", type=["pdf", "txt", "md"],
                               accept_multiple_files=True)
    retriever = None
    if uploads:
        payloads = [(u.name, u.getvalue()) for u in uploads]
        key = hashlib.sha1(b"".join(n.encode() + d for n, d in payloads)).hexdigest()
        try:
            retriever, n_chunks, n_docs = build_user_retriever(embedder, reranker, key, payloads)
            st.success(f"Indexed **{n_docs} document(s)** → {n_chunks} chunks. Ask away below.")
        except ValueError as e:
            st.error(str(e))
    else:
        st.info("⬆️ Upload one or more documents to get started.")

    query = st.text_input("Ask a question about your documents")
    if st.button("Ask", type="primary") and query.strip():
        if retriever is None:
            st.warning("Upload at least one document first.")
        elif not api_key:
            st.error("Enter your Groq API key in the sidebar.")
        else:
            with st.spinner("Retrieving..."):
                chunks = retrieve(retriever, query)
            render_result(query, chunks, api_key, threshold)

elif corpus == WEB:
    st.caption("Type a question; it searches the web live, reads the top pages, and answers with verified citations to the URLs.")
    query = st.text_input("Ask anything")
    if st.button("Search & answer", type="primary") and query.strip():
        if not api_key:
            st.error("Enter your Groq API key in the sidebar.")
        else:
            retriever, n_chunks, n_docs = build_web_retriever(embedder, reranker, query, max_web)
            if retriever is None:
                st.error("No web results (search/fetch failed — possibly rate-limited or offline). Try again.")
            else:
                st.success(f"Read **{n_docs} web page(s)** → {n_chunks} chunks.")
                with st.spinner("Retrieving..."):
                    chunks = retrieve(retriever, query)
                render_result(query, chunks, api_key, threshold)

else:  # SCIFACT
    index, sparse = load_scifact(embedder)
    retriever = HybridRetriever(index, sparse, embedder, reranker)
    st.caption("Corpus: **SciFact** — 5,183 pre-loaded biomedical abstracts (the labeled benchmark the measured results are computed on).")
    query = st.text_input("Ask a question about the SciFact corpus",
                          value="Does a high-fat diet increase the risk of cardiovascular disease?")
    if st.button("Ask", type="primary") and query.strip():
        if not api_key:
            st.error("Enter your Groq API key in the sidebar.")
        else:
            with st.spinner("Retrieving..."):
                chunks = retrieve(retriever, query)
            render_result(query, chunks, api_key, threshold)
