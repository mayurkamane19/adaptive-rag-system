"""Streamlit Frontend Application for Adaptive RAG System."""

import os
import time
from typing import Dict, Any, List, Optional
import streamlit as st
import requests

# Page Configuration
st.set_page_config(
    page_title="Adaptive RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS with Glassmorphism, Responsive Hover Effects & Smooth Animations
CUSTOM_CSS = """
<style>
/* ==========================================================================
   Adaptive RAG System — Premium UI Styles & Lightweight Animations
   ========================================================================== */

/* 1. Hardware-Accelerated Ambient Gradient Background & Drift */
.stApp {
    background: radial-gradient(ellipse at 15% 15%, rgba(30, 41, 59, 0.45) 0%, rgba(15, 23, 42, 0.75) 45%, #0b0f19 100%),
                linear-gradient(135deg, #090d16 0%, #0d121f 50%, #0b0f19 100%);
    background-attachment: fixed;
    color: #f1f5f9;
}

.stApp::before {
    content: "";
    position: fixed;
    top: -40%;
    left: -40%;
    width: 180%;
    height: 180%;
    background: radial-gradient(circle at 30% 25%, rgba(99, 102, 241, 0.05) 0%, transparent 35%),
                radial-gradient(circle at 75% 15%, rgba(56, 189, 248, 0.04) 0%, transparent 30%),
                radial-gradient(circle at 45% 85%, rgba(168, 85, 247, 0.04) 0%, transparent 35%);
    animation: ambientShift 30s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
    will-change: transform;
}

@keyframes ambientShift {
    0% { transform: translate(0, 0) rotate(0deg); }
    50% { transform: translate(-25px, 15px) rotate(1deg); }
    100% { transform: translate(15px, -20px) rotate(-1deg); }
}

/* 2. Glassmorphism Styling */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.78) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 12px 18px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.25s ease, box-shadow 0.25s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 6px 10px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    color: #94a3b8;
    border: 1px solid transparent;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #f8fafc;
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(255, 255, 255, 0.1);
    transform: translateY(-1px);
}

.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.18) !important;
    color: #a5b4fc !important;
    border-color: rgba(99, 102, 241, 0.35) !important;
    box-shadow: 0 2px 12px rgba(99, 102, 241, 0.25);
}

/* 3. Smooth Hover Effects for Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 500;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
    border-color: rgba(99, 102, 241, 0.5);
}

.stButton > button:active {
    transform: translateY(0);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    border: 1px solid rgba(165, 180, 252, 0.3);
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%);
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.55);
    border-color: rgba(165, 180, 252, 0.6);
}

/* 4. Polished Loading Spinner Indicator */
[data-testid="stSpinner"] {
    background: rgba(30, 41, 59, 0.75);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 12px;
    padding: 16px 22px;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
    animation: spinnerGlow 2.2s ease-in-out infinite;
}

@keyframes spinnerGlow {
    0%, 100% {
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
    }
    50% {
        border-color: rgba(99, 102, 241, 0.65);
        box-shadow: 0 8px 36px rgba(99, 102, 241, 0.38);
    }
}

/* 5. Subtle Answer Reveal Animation */
.answer-reveal, [data-testid="stAlert"] {
    animation: revealSlideUp 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 12px;
}

@keyframes revealSlideUp {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* 6. Expanders and Input Elements */
[data-testid="stExpander"] {
    background: rgba(30, 41, 59, 0.35) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stExpander"]:hover {
    border-color: rgba(99, 102, 241, 0.3) !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.2);
}

.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: rgba(15, 23, 42, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
    transition: all 0.2s ease !important;
}

.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
}

/* 7. Typography & Visual Hierarchy */
.hero-header {
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.hero-title {
    font-size: 2.15rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.hero-subtitle {
    font-size: 1.0rem;
    color: #94a3b8;
    line-height: 1.5;
    margin-bottom: 0.5rem;
}

.source-card {
    background: rgba(15, 23, 42, 0.55);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    animation: revealSlideUp 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.source-card:hover {
    background: rgba(30, 41, 59, 0.65);
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateX(4px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
}

.chunk-badge {
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.3);
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-family: monospace;
    margin-left: 6px;
}

.route-badge-simple {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    font-weight: 600;
    font-size: 0.85rem;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.15);
}

.route-badge-complex {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    background: rgba(244, 63, 94, 0.15);
    border: 1px solid rgba(244, 63, 94, 0.4);
    color: #fb7185;
    font-weight: 600;
    font-size: 0.85rem;
    box-shadow: 0 0 12px rgba(244, 63, 94, 0.15);
}

/* 8. Reduced Motion Accessibility */
@media (prefers-reduced-motion: reduce) {
    *, ::before, ::after {
        animation-duration: 0.001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.001ms !important;
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Application Header
st.markdown(
    """
    <div class="hero-header">
        <h1 class="hero-title">🤖 Adaptive Retrieval-Augmented Generation (RAG) System</h1>
        <p class="hero-subtitle">
            Enterprise multi-stage document intelligence with LangGraph adaptive routing, FAISS vector search, Cross-Encoder re-ranking, and grounded Ollama generation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info("ℹ️ **Production Ready**: Enterprise Adaptive RAG System with Cross-Encoder Re-Ranking, Conversational Memory, and Verified Grounding.")

# Sidebar Configuration
st.sidebar.header("⚙️ System Status & Configuration")
st.sidebar.markdown("**Backend Status**: 🟢 Online & Production Ready")
st.sidebar.markdown("**LLM Provider**: Ollama (Local)")
st.sidebar.markdown("**LLM Model**: `llama3`")
st.sidebar.markdown("**Workflow Engine**: LangGraph StateGraph")
st.sidebar.markdown("**Routing Logic**: Deterministic Heuristic Classifier")
st.sidebar.markdown("**Query Rewriter**: History-Aware LLM Reformulator")
st.sidebar.markdown("**First-Stage Vector Store**: FAISS")
st.sidebar.markdown("**Second-Stage Re-Ranker**: `BAAI/bge-reranker-base`")
st.sidebar.markdown("**Candidate Pool (Simple/Complex)**: 8 / 16")
st.sidebar.markdown("**Final Top-K Depth (Simple/Complex)**: 4 / 6")
st.sidebar.markdown("**Embedding Model**: `all-MiniLM-L6-v2`")
st.sidebar.markdown("**Evaluation Benchmark**: `data/evaluation/eval_dataset.json`")

default_api_url = os.environ.get("BACKEND_API_URL", "http://127.0.0.1:8000")
api_base_url = st.sidebar.text_input("Backend API Base URL", value=default_api_url)
api_key_env = os.environ.get("API_KEY", "")
api_key = st.sidebar.text_input("API Key (Optional)", value=api_key_env, type="password", help="Sent as X-API-Key header if configured")

# Helper to build headers
def get_auth_headers():
    return {"X-API-Key": api_key.strip()} if api_key.strip() else {}


def format_latency(ms: float) -> str:
    """Format millisecond latency into clean human-readable units (seconds or milliseconds)."""
    if ms is None:
        return "N/A"
    if ms >= 1000.0:
        return f"{ms / 1000.0:.2f} s"
    return f"{ms:.1f} ms"


def render_stage_latencies_grid(timings: Dict[str, Any], client_elapsed_ms: float):
    """Renders operational stage latencies in clear execution order with unified units."""
    ordered_stages = [
        ("load_conversation_ms", "Load History"),
        ("rewrite_query_ms", "Rewrite Query"),
        ("analyze_query_ms", "Analyze Query"),
        ("routing_ms", "Routing"),
        ("retrieval_ms", "Retrieval"),
        ("rerank_ms", "Re-ranking"),
        ("context_ms", "Context Build"),
        ("prompt_ms", "Prompt Build"),
        ("generation_ms", "Generation"),
        ("save_conversation_ms", "Save History"),
    ]

    active_stages = []
    for key, label in ordered_stages:
        val = timings.get(key)
        if isinstance(val, (int, float)) and val > 0:
            active_stages.append((label, val))

    server_total = timings.get("total_ms")
    if isinstance(server_total, (int, float)) and server_total > 0:
        active_stages.append(("Server Total", server_total))
    active_stages.append(("Client Round-Trip", client_elapsed_ms))

    st.markdown("**⏱️ Operational Stage Latencies:**")
    cols = st.columns(min(len(active_stages), 4))
    for idx, (label, val_ms) in enumerate(active_stages):
        with cols[idx % 4]:
            st.metric(label, format_latency(val_ms), help=f"{val_ms:.1f} ms ({val_ms / 1000.0:.3f} s)")


# Backend Health Check Button in Sidebar
if st.sidebar.button("Check Backend Health"):
    try:
        response = requests.get(f"{api_base_url}/health/ready", headers=get_auth_headers(), timeout=4)
        if response.status_code == 200:
            data = response.json()
            st.sidebar.success(f"Backend Ready: {data.get('status').upper()}")
            comps = data.get("components", {})
            st.sidebar.caption(f"Vector Store: {comps.get('vector_store', {}).get('status')}")
            st.sidebar.caption(f"LLM: {comps.get('llm_service', {}).get('status')}")
        else:
            st.sidebar.error(f"Backend Returned HTTP {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"Failed to connect to backend: {str(e)}")

st.divider()

# Layout Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📄 PDF Document Indexing",
        "🔍 Semantic Search",
        "📋 Basic RAG Context Construction",
        "🧠 Adaptive RAG (LangGraph)",
        "💬 RAG Answer Generation",
        "🗣️ Conversational RAG",
        "📊 RAG Evaluation & Benchmarks",
    ]
)

with tab1:
    st.subheader("📄 PDF Document Indexing")
    st.markdown("Upload and index PDF documents into the local FAISS vector store.")
    uploaded_file = st.file_uploader("Choose a PDF document", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Index Document into FAISS", type="primary"):
            with st.spinner("Indexing document (Parsing, chunking, generating embeddings, updating FAISS)..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{api_base_url}/api/v1/documents/index", files=files, timeout=90)

                    if response.status_code == 200:
                        data = response.json()
                        status_val = str(data.get("status", "indexed")).lower()
                        if status_val == "duplicate":
                            st.warning(f"⚠️ {data.get('message')}")
                        elif status_val == "warning":
                            st.warning(f"⚠️ {data.get('message')}")
                        else:
                            st.success("✅ Document Indexed Successfully!")

                        col_i1, col_i2, col_i3 = st.columns(3)
                        with col_i1:
                            st.metric("Pages Extracted", data.get("page_count", 0))
                        with col_i2:
                            st.metric("Chunks Indexed", data.get("chunk_count", 0))
                        with col_i3:
                            st.metric("Indexing Status", str(data.get("status", "indexed")).capitalize())

                        st.markdown(f"**Filename:** `{data.get('filename')}`")
                        st.info(f"**Message:** {data.get('message')}")
                    else:
                        error_detail = response.json().get("detail", "Unknown error occurred.")
                        st.error(f"❌ Indexing Failed (HTTP {response.status_code}): {error_detail}")

                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")

with tab2:
    st.subheader("🔍 Semantic Similarity Search")
    st.markdown("Search across indexed document chunks using vector embeddings.")

    search_query = st.text_input(
        "Enter search query:",
        placeholder="e.g., What are the main revenue projections for next year?",
        key="sem_search_query",
    )

    top_k_search = st.slider("Number of retrieved chunks (top_k)", min_value=1, max_value=10, value=4, key="top_k_search")

    if st.button("Search Vector Store", type="primary", key="btn_search"):
        if not search_query.strip():
            st.error("Please enter a search query before submitting.")
        else:
            with st.spinner("Searching FAISS vector database..."):
                try:
                    payload = {"query": search_query.strip(), "top_k": top_k_search}
                    response = requests.post(f"{api_base_url}/api/v1/search", json=payload, timeout=15)

                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])

                        st.markdown(f"### 🎯 Search Results for: *\"{search_query}\"*")
                        st.caption(f"Retrieved {len(results)} relevant chunk(s)")

                        if not results:
                            st.warning("No matching document chunks found.")
                        else:
                            for idx, item in enumerate(results, 1):
                                with st.expander(
                                    f"Chunk #{idx} | {item.get('original_filename')} (Page {item.get('page')})",
                                    expanded=(idx == 1),
                                ):
                                    st.markdown(f"**Chunk ID:** `{item.get('chunk_id')}`")
                                    if item.get("score") is not None:
                                        st.markdown(f"**Distance Score:** `{item.get('score')}`")
                                    st.markdown("**Content Excerpt:**")
                                    st.markdown(f"> {item.get('content')}")

                    elif response.status_code == 404:
                        st.warning("⚠️ Vector store is empty or does not exist. Please index a PDF document first.")
                    else:
                        error_detail = response.json().get("detail", "Unknown error occurred.")
                        st.error(f"❌ Search Failed (HTTP {response.status_code}): {error_detail}")

                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")

with tab3:
    st.subheader("📋 Basic RAG Context Construction")
    st.markdown("Retrieve matching document chunks and format a bounded RAG context payload for downstream LLM generation.")

    context_query = st.text_input(
        "Enter question to construct RAG context:",
        placeholder="e.g., How should remote employees connect to corporate networks?",
        key="rag_context_query",
    )

    top_k_context = st.slider("Max retrieved chunks (top_k)", min_value=1, max_value=10, value=4, key="top_k_context")

    if st.button("Construct RAG Context", type="primary", key="btn_context"):
        if not context_query.strip():
            st.error("Please enter a question before submitting.")
        else:
            with st.spinner("Retrieving chunks and constructing bounded RAG context..."):
                try:
                    payload = {"query": context_query.strip(), "top_k": top_k_context}
                    response = requests.post(f"{api_base_url}/api/v1/rag/context", json=payload, timeout=15)

                    if response.status_code == 200:
                        data = response.json()
                        status_str = data.get("status")
                        retrieved_cnt = data.get("retrieved_count", 0)
                        included_cnt = data.get("included_count", 0)
                        context_str = data.get("context", "")
                        sources_list = data.get("sources", [])

                        st.markdown(f"### 🎯 Context Results for: *\"{context_query}\"*")
                        col_c1, col_c2, col_c3 = st.columns(3)
                        with col_c1:
                            st.metric("Context Status", str(status_str).capitalize())
                        with col_c2:
                            st.metric("Chunks Retrieved", retrieved_cnt)
                        with col_c3:
                            st.metric("Included in Context", included_cnt)

                        if status_str == "no_results" or not context_str:
                            st.warning("⚠️ No relevant documents found in vector store to construct context.")
                        else:
                            st.markdown("#### 📚 Source Reference Metadata")
                            for s in sources_list:
                                st.markdown(
                                    f'<div class="source-card">📄 <strong>{s.get("original_filename")}</strong> (Page {s.get("page")}) <span class="chunk-badge">{s.get("chunk_id")}</span></div>',
                                    unsafe_allow_html=True,
                                )

                            st.markdown("#### 📝 Formatted RAG Context Payload")
                            st.text_area(
                                label="Constructed Context String:",
                                value=context_str,
                                height=280,
                            )
                            st.info("ℹ️ **Context only — LLM generation available in Tab 5.**")

                    else:
                        error_detail = response.json().get("detail", "Unknown error occurred.")
                        st.error(f"❌ Context Construction Failed (HTTP {response.status_code}): {error_detail}")

                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")

with tab4:
    st.subheader("🧠 Adaptive RAG (LangGraph Orchestration)")
    st.markdown("Dynamic stateful query routing via LangGraph: automatically classifies query complexity to select optimal retrieval depth.")

    adaptive_query = st.text_input(
        "Ask an adaptive question:",
        placeholder="e.g., Compare the company's remote work policy with its office attendance rules.",
        key="adaptive_rag_query",
    )

    if st.button("Execute Adaptive RAG Graph", type="primary", key="btn_adaptive"):
        if not adaptive_query.strip():
            st.error("Please enter a query before submitting.")
        else:
            with st.spinner("Executing LangGraph adaptive state graph (Classification -> Routing -> Retrieval -> Context)..."):
                try:
                    payload = {"query": adaptive_query.strip()}
                    t_client_start = time.perf_counter()
                    response = requests.post(f"{api_base_url}/api/v1/rag/adaptive-context", json=payload, timeout=60)
                    client_elapsed_ms = round((time.perf_counter() - t_client_start) * 1000, 1)

                    if response.status_code == 200:
                        data = response.json()
                        route_name = data.get("route", "simple").upper()
                        reason_str = data.get("route_reason", "")
                        confidence = data.get("route_confidence", 0.0)
                        top_k_allocated = data.get("top_k", 4)
                        status_str = data.get("status")
                        retrieved_cnt = data.get("retrieved_count", 0)
                        included_cnt = data.get("included_count", 0)
                        context_str = data.get("context", "")
                        sources_list = data.get("sources", [])

                        # Routing Badge & Details
                        if route_name == "COMPLEX":
                            badge_html = '<span class="route-badge-complex">🔴 COMPLEX ROUTE</span>'
                        else:
                            badge_html = '<span class="route-badge-simple">🟢 SIMPLE ROUTE</span>'
                        st.markdown(f"### Route Selected: {badge_html}", unsafe_allow_html=True)

                        col_r1, col_r2, col_r3 = st.columns(3)
                        with col_r1:
                            st.metric("Retrieval Depth (top_k)", f"{top_k_allocated}")
                        with col_r2:
                            st.metric("Heuristic Routing Confidence", f"{confidence:.2f}")
                        with col_r3:
                            st.metric("Chunks Included / Retrieved", f"{included_cnt} / {retrieved_cnt}")

                        st.markdown(f"**Routing Rationale:** {reason_str}")
                        st.caption("ℹ️ *Confidence is a deterministic heuristic score based on query signals.*")

                        st.divider()

                        if status_str == "no_results" or not context_str:
                            st.warning("⚠️ No relevant documents found in vector store for adaptive context.")
                        else:
                            st.markdown("#### 📚 Source Reference Metadata")
                            for s in sources_list:
                                st.markdown(
                                    f'<div class="source-card">📄 <strong>{s.get("original_filename")}</strong> (Page {s.get("page")}) <span class="chunk-badge">{s.get("chunk_id")}</span></div>',
                                    unsafe_allow_html=True,
                                )

                            st.markdown("#### 📝 Formatted Adaptive RAG Context Payload")
                            st.text_area(
                                label="Constructed Context String:",
                                value=context_str,
                                height=280,
                                key="adaptive_context_area",
                            )

                            with st.expander("🔬 Retrieval & Workflow Diagnostics", expanded=False):
                                cand_cnt = data.get("candidate_count", 0)
                                rerank_cnt = data.get("reranked_count", 0)
                                rr_enabled = data.get("reranker_enabled", True)
                                rr_model = data.get("reranker_model", "BAAI/bge-reranker-base")
                                rr_status = data.get("reranker_status", "success")

                                col_diag1, col_diag2, col_diag3, col_diag4 = st.columns(4)
                                with col_diag1:
                                    st.metric("Candidate Pool (FAISS)", cand_cnt)
                                with col_diag2:
                                    st.metric("Final Documents (Re-ranked)", rerank_cnt)
                                with col_diag3:
                                    st.metric("Re-ranker Active", "Yes" if rr_enabled else "No")
                                with col_diag4:
                                    st.metric("Re-ranker Status", rr_status)

                                st.markdown(f"**Model Used:** `{rr_model}`")

                                timings = data.get("timings")
                                if timings:
                                    render_stage_latencies_grid(timings, client_elapsed_ms)

                    else:
                        error_detail = response.json().get("detail", "Unknown error occurred.")
                        st.error(f"❌ Adaptive RAG Execution Failed (HTTP {response.status_code}): {error_detail}")

                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")

with tab5:
    st.subheader("💬 RAG Answer Generation")
    st.markdown("End-to-end Adaptive RAG pipeline: Query Routing → Vector Retrieval → Grounded Custom Prompt → Local LLM Answer Generation.")

    rag_answer_query = st.text_input(
        "Ask a question to generate a grounded AI answer:",
        placeholder="e.g., What is the annual leave allowance for employees?",
        key="rag_answer_query",
    )

    if st.button("Generate Grounded Answer", type="primary", key="btn_generate_answer"):
        if not rag_answer_query.strip():
            st.error("Please enter a question before submitting.")
        else:
            with st.spinner("Executing full Adaptive RAG pipeline & LLM generation..."):
                try:
                    payload = {"query": rag_answer_query.strip()}
                    t_client_start = time.perf_counter()
                    response = requests.post(f"{api_base_url}/api/v1/rag/answer", json=payload, timeout=180)
                    client_elapsed_ms = round((time.perf_counter() - t_client_start) * 1000, 1)

                    if response.status_code == 200:
                        data = response.json()
                        route_name = data.get("route", "simple").upper()
                        reason_str = data.get("route_reason", "")
                        confidence = data.get("route_confidence", 0.0)
                        top_k_allocated = data.get("top_k", 4)
                        answer_text = data.get("answer", "")
                        sources_list = data.get("sources", [])
                        provider = data.get("llm_provider", "ollama")
                        model_name = data.get("llm_model", "llama3")

                        # Routing Badge & Details
                        if route_name == "COMPLEX":
                            badge_html = '<span class="route-badge-complex">🔴 COMPLEX ROUTE</span>'
                        else:
                            badge_html = '<span class="route-badge-simple">🟢 SIMPLE ROUTE</span>'
                        st.markdown(f"### Route Selected: {badge_html}", unsafe_allow_html=True)

                        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                        with col_a1:
                            st.metric("Retrieval Depth (top_k)", f"{top_k_allocated}")
                        with col_a2:
                            st.metric("Heuristic Confidence", f"{confidence:.2f}")
                        with col_a3:
                            st.metric("LLM Provider", f"{provider}")
                        with col_a4:
                            st.metric("LLM Model", f"{model_name}")

                        st.markdown(f"**Routing Rationale:** {reason_str}")
                        st.divider()

                        st.markdown("### 🤖 Grounded AI Response")
                        gen_status = data.get("generation_status")
                        if gen_status == "failed" or answer_text.startswith("LLM generation service is currently unavailable"):
                            st.error(answer_text)
                            st.info(
                                "💡 **Ollama Server Offline**: The local LLM could not be reached.\n\n"
                                "**Windows Startup Steps:**\n"
                                "1. Verify Ollama is installed from [ollama.com](https://ollama.com).\n"
                                "2. In PowerShell or Command Prompt, run: `ollama serve` (or launch the Ollama desktop app).\n"
                                "3. Download the model: `ollama pull llama3`\n"
                                "4. If Ollama runs on a custom address, set `OLLAMA_BASE_URL` in `.env`.\n\n"
                                "*(Note: Vector retrieval & re-ranking succeeded! Retrieved supporting sources are preserved below.)*"
                            )
                        else:
                            st.success(answer_text)

                        st.markdown("#### 📄 Retrieved Supporting Sources")
                        if not sources_list:
                            st.info("No sources referenced.")
                        else:
                            for s in sources_list:
                                st.markdown(
                                    f'<div class="source-card">📄 <strong>{s.get("original_filename")}</strong> (Page {s.get("page")}) <span class="chunk-badge">{s.get("chunk_id")}</span></div>',
                                    unsafe_allow_html=True,
                                )

                        with st.expander("🔬 Retrieval & Re-Ranking Diagnostics", expanded=False):
                            cand_cnt = data.get("candidate_count", 0)
                            rerank_cnt = data.get("reranked_count", 0)
                            rr_enabled = data.get("reranker_enabled", True)
                            rr_model = data.get("reranker_model", "BAAI/bge-reranker-base")
                            rr_status = data.get("reranker_status", "success")

                            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                            with col_d1:
                                st.metric("Candidate Pool (FAISS)", cand_cnt)
                            with col_d2:
                                st.metric("Final Documents (Re-ranked)", rerank_cnt)
                            with col_d3:
                                st.metric("Re-ranker Active", "Yes" if rr_enabled else "No")
                            with col_d4:
                                st.metric("Re-ranker Status", rr_status)

                            st.markdown(f"**Model Used:** `{rr_model}`")

                            timings = data.get("timings")
                            if timings:
                                tps = timings.get("tokens_per_second")
                                if tps is not None:
                                    st.markdown("**⚡ Model Inference Performance (Ollama):**")
                                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                                    with col_m1:
                                        p_eval_ms = timings.get("prompt_eval_ms", 0.0)
                                        p_eval_cnt = timings.get("prompt_eval_count", 0)
                                        st.metric("Prompt Eval", format_latency(p_eval_ms), help=f"Evaluated {p_eval_cnt} prompt tokens ({p_eval_ms:.1f} ms)")
                                    with col_m2:
                                        eval_ms = timings.get("eval_ms", 0.0)
                                        eval_cnt = timings.get("eval_count", 0)
                                        st.metric("Token Generation", format_latency(eval_ms), help=f"Generated {eval_cnt} tokens ({eval_ms:.1f} ms)")
                                    with col_m3:
                                        st.metric("Inference Speed", f"{tps:.2f} tok/s")
                                    with col_m4:
                                        st.metric("Client Round-Trip", format_latency(client_elapsed_ms), help=f"Full client-observed elapsed time ({client_elapsed_ms:.1f} ms)")

                                render_stage_latencies_grid(timings, client_elapsed_ms)

                    else:
                        error_detail = response.json().get("detail", "Unknown error occurred.")
                        st.error(f"❌ Answer Generation Failed (HTTP {response.status_code}): {error_detail}")

                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")

with tab6:
    st.subheader("🗣️ Multi-Turn Conversational RAG & Memory")
    st.markdown("Interactive multi-turn conversation interface with history-aware query reformulation and session isolation.")

    if "chat_session_id" not in st.session_state:
        st.session_state["chat_session_id"] = "session-demo1"

    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        current_session = st.text_input("Active Session ID:", value=st.session_state["chat_session_id"], key="input_session_id")
        st.session_state["chat_session_id"] = current_session
    with col_s2:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear Session History", type="secondary"):
            try:
                res = requests.delete(
                    f"{api_base_url}/api/v1/rag/chat/{current_session}",
                    headers=get_auth_headers(),
                    timeout=5,
                )
                if res.status_code == 200:
                    st.success("Session history cleared!")
                    st.session_state["chat_history_messages"] = []
                else:
                    st.error("Failed to clear session.")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

    user_input = st.text_input("Ask a question or follow-up:", placeholder="e.g., Can I carry over my unused leave to next year?", key="conv_user_input")

    if st.button("Send Message", type="primary", key="btn_send_conv"):
        if not user_input.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Processing conversational RAG turn (Query Rewriting -> Adaptive Retrieval -> Grounded Generation)..."):
                try:
                    payload = {"query": user_input.strip(), "session_id": current_session}
                    t_client_start = time.perf_counter()
                    res = requests.post(f"{api_base_url}/api/v1/rag/chat", json=payload, timeout=180)
                    client_elapsed_ms = round((time.perf_counter() - t_client_start) * 1000, 1)

                    if res.status_code == 200:
                        data = res.json()
                        st.markdown("### 💬 Turn Results")

                        if data.get("was_rewritten"):
                            st.info(f"🔄 **Query Rewritten into Standalone Search Query:** `{data.get('standalone_query')}`")
                        else:
                            st.caption("ℹ️ *Query kept in original form (no history needed or query already standalone).*")

                        route_name = data.get("route", "simple").upper()
                        if route_name == "COMPLEX":
                            badge_html = '<span class="route-badge-complex">🔴 COMPLEX ROUTE</span>'
                        else:
                            badge_html = '<span class="route-badge-simple">🟢 SIMPLE ROUTE</span>'
                        st.markdown(f"**Route Selected:** {badge_html} &nbsp;*(retrieval top_k={data.get('top_k')})*", unsafe_allow_html=True)

                        st.markdown("### 🤖 Assistant Answer")
                        gen_status = data.get("generation_status")
                        ans_val = data.get("answer", "")
                        if gen_status == "failed" or ans_val.startswith("LLM generation service is currently unavailable"):
                            st.error(ans_val)
                            st.info(
                                "💡 **Ollama Server Offline**: Start Ollama with `ollama serve` and pull `llama3` (`ollama pull llama3`). "
                                "Retrieved document sources are preserved below."
                            )
                        else:
                            st.success(ans_val)

                        sources_list = data.get("sources", [])
                        if sources_list:
                            st.markdown("#### 📄 Retrieved Supporting Sources")
                            for s in sources_list:
                                st.markdown(
                                    f'<div class="source-card">📄 <strong>{s.get("original_filename")}</strong> (Page {s.get("page")}) <span class="chunk-badge">{s.get("chunk_id")}</span></div>',
                                    unsafe_allow_html=True,
                                )

                        with st.expander("🔬 Retrieval & Re-Ranking Diagnostics", expanded=False):
                            cand_cnt = data.get("candidate_count", 0)
                            rerank_cnt = data.get("reranked_count", 0)
                            rr_enabled = data.get("reranker_enabled", True)
                            rr_model = data.get("reranker_model", "BAAI/bge-reranker-base")
                            rr_status = data.get("reranker_status", "success")

                            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                            with col_d1:
                                st.metric("Candidate Pool (FAISS)", cand_cnt)
                            with col_d2:
                                st.metric("Final Documents (Re-ranked)", rerank_cnt)
                            with col_d3:
                                st.metric("Re-ranker Active", "Yes" if rr_enabled else "No")
                            with col_d4:
                                st.metric("Re-ranker Status", rr_status)

                            st.markdown(f"**Model Used:** `{rr_model}`")

                            timings = data.get("timings")
                            if timings:
                                tps = timings.get("tokens_per_second")
                                if tps is not None:
                                    st.markdown("**⚡ Model Inference Performance (Ollama):**")
                                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                                    with col_m1:
                                        p_eval_ms = timings.get("prompt_eval_ms", 0.0)
                                        p_eval_cnt = timings.get("prompt_eval_count", 0)
                                        st.metric("Prompt Eval", format_latency(p_eval_ms), help=f"Evaluated {p_eval_cnt} prompt tokens ({p_eval_ms:.1f} ms)")
                                    with col_m2:
                                        eval_ms = timings.get("eval_ms", 0.0)
                                        eval_cnt = timings.get("eval_count", 0)
                                        st.metric("Token Generation", format_latency(eval_ms), help=f"Generated {eval_cnt} tokens ({eval_ms:.1f} ms)")
                                    with col_m3:
                                        st.metric("Inference Speed", f"{tps:.2f} tok/s")
                                    with col_m4:
                                        st.metric("Client Round-Trip", format_latency(client_elapsed_ms), help=f"Full client-observed elapsed time ({client_elapsed_ms:.1f} ms)")

                                render_stage_latencies_grid(timings, client_elapsed_ms)

                    else:
                        st.error(f"Chat request failed with HTTP {res.status_code}: {res.json().get('detail')}")

                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

    st.divider()
    st.markdown("### 📜 Session Conversation History")
    if st.button("Refresh History View"):
        try:
            res = requests.get(f"{api_base_url}/api/v1/rag/chat/{current_session}/history", timeout=5)
            if res.status_code == 200:
                hist_data = res.json()
                messages = hist_data.get("messages", [])
                if not messages:
                    st.info("No conversation history found for this session ID.")
                else:
                    for msg in messages:
                        is_user = msg["role"] == "user"
                        role_icon = "👤" if is_user else "🤖"
                        role_label = msg["role"].upper()
                        border_color = "#3b82f6" if is_user else "#10b981"
                        st.markdown(
                            f'<div class="source-card" style="border-left-color: {border_color};">'
                            f'<strong>{role_icon} {role_label}</strong> '
                            f'<span style="font-size: 0.8rem; color: #94a3b8; margin-left: 8px;">{msg.get("timestamp", "")}</span>'
                            f'<div style="margin-top: 6px; color: #e2e8f0; line-height: 1.5;">{msg.get("content", "")}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.error("Failed to fetch history.")
        except Exception as e:
            st.error(f"Connection error: {str(e)}")

with tab7:
    st.subheader("📊 RAG Evaluation, Answer Quality & Benchmarks")
    st.markdown(
        "Evaluate RAG retrieval accuracy and answer quality across benchmark datasets. "
        "Deterministic metrics include **Hit Rate / Recall@K**, **MRR**, **Token F1**, "
        "**Lexical Grounding Ratio**, **Refusal Accuracy**, and **Operational Latency Profiling**."
    )

    col_e1, col_e2 = st.columns([3, 1])
    with col_e1:
        eval_dataset_input = st.text_input(
            "Evaluation Dataset Path:",
            value="data/evaluation/eval_dataset.json",
            key="eval_dataset_input",
            help="Relative or absolute path within project to benchmark JSON dataset.",
        )
    with col_e2:
        st.write("")
        st.write("")
        eval_use_reranker = st.checkbox("Enable Re-Ranking", value=True, key="eval_use_reranker")

    col_eb1, col_eb2 = st.columns(2)
    with col_eb1:
        btn_run_eval = st.button("🚀 Run Evaluation Benchmark", type="primary", key="btn_run_eval")
    with col_eb2:
        btn_view_dataset = st.button("📄 Inspect Dataset Samples", type="secondary", key="btn_view_dataset")

    if btn_run_eval:
        with st.spinner("Executing RAG evaluation benchmark across dataset samples..."):
            try:
                eval_payload = {
                    "dataset_path": eval_dataset_input.strip(),
                    "use_reranker": eval_use_reranker,
                }
                res = requests.post(
                    f"{api_base_url}/api/v1/evaluation/run",
                    json=eval_payload,
                    headers=get_auth_headers(),
                    timeout=180,
                )

                if res.status_code == 200:
                    eval_data = res.json()
                    summary = eval_data.get("summary", {})
                    sample_results = eval_data.get("sample_results", [])
                    errors_list = eval_data.get("errors", [])

                    st.markdown("### 📈 Benchmark Aggregate Metrics")

                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        total_s = summary.get("total_samples", 0)
                        succ_s = summary.get("successful_runs", 0)
                        st.metric("Total Samples / Success", f"{succ_s} / {total_s}")
                    with col_m2:
                        hit_rate = summary.get("mean_hit_rate")
                        st.metric("Mean Recall@K / Hit Rate", f"{hit_rate:.2%}" if hit_rate is not None else "N/A")
                    with col_m3:
                        mrr = summary.get("mean_mrr")
                        st.metric("Mean MRR", f"{mrr:.4f}" if mrr is not None else "N/A")
                    with col_m4:
                        mean_f1 = summary.get("mean_answer_f1", 0.0)
                        st.metric("Mean Token F1", f"{mean_f1:.2%}")

                    col_m5, col_m6, col_m7, col_m8 = st.columns(4)
                    with col_m5:
                        grounding = summary.get("mean_grounding_score", 0.0)
                        st.metric("Mean Grounding Ratio", f"{grounding:.2%}")
                    with col_m6:
                        refusal_acc = summary.get("refusal_accuracy")
                        st.metric("Refusal Accuracy", f"{refusal_acc:.2%}" if refusal_acc is not None else "N/A")
                    with col_m7:
                        empty_ctx = summary.get("empty_context_count", 0)
                        st.metric("Empty Context Count", f"{empty_ctx}")
                    with col_m8:
                        missing_src = summary.get("missing_sources_count", 0)
                        st.metric("Missing Sources Count", f"{missing_src}")

                    # Operational Latencies
                    mean_latencies = summary.get("mean_latencies_ms", {})
                    if mean_latencies:
                        st.markdown("#### ⏱️ Mean Operational Stage Latencies")
                        lat_cols = st.columns(min(len(mean_latencies), 6))
                        for idx, (stg, val) in enumerate(mean_latencies.items()):
                            with lat_cols[idx % min(len(mean_latencies), 6)]:
                                st.metric(stg.replace("_ms", "").replace("_", " ").title(), f"{val} ms")

                    st.info(
                        f"ℹ️ **Deterministic Metrics Note:** {summary.get('deterministic_metrics_note', '')}"
                    )

                    if errors_list:
                        with st.expander("⚠️ Benchmark Warnings & Non-Fatal Errors", expanded=False):
                            for err in errors_list:
                                st.warning(err)

                    # Detailed itemized samples
                    st.markdown("### 📋 Itemized Sample Evaluation Results")
                    for s_res in sample_results:
                        s_id = s_res.get("id")
                        s_cat = s_res.get("category", "generic")
                        s_q = s_res.get("question", "")
                        s_status = s_res.get("status")
                        s_f1 = s_res.get("answer_f1", 0.0)
                        s_grounding = s_res.get("grounding_score", 0.0)
                        s_hit = s_res.get("hit_rate")
                        s_mrr = s_res.get("reciprocal_rank")

                        icon = "✅" if s_status == "success" else "❌"
                        header_title = f"{icon} [{s_id}] ({s_cat.upper()}) {s_q[:60]}... — F1: {s_f1:.2%}, Grounding: {s_grounding:.2%}"

                        with st.expander(header_title, expanded=False):
                            col_sub1, col_sub2 = st.columns(2)
                            with col_sub1:
                                st.markdown(f"**Question:** {s_q}")
                                st.markdown(f"**Expected Answer:** {s_res.get('expected_answer')}")
                                st.markdown(f"**Expected Sources:** `{s_res.get('expected_sources')}`")
                                st.markdown(f"**Is Answerable:** `{s_res.get('is_answerable')}`")
                            with col_sub2:
                                st.markdown(f"**Generated Answer:** {s_res.get('generated_answer')}")
                                st.markdown(f"**Retrieved Sources:** `{s_res.get('retrieved_sources')}`")
                                st.markdown(f"**Assigned Route:** `{s_res.get('route')}`")
                                st.markdown(f"**Refusal Correct:** `{s_res.get('refusal_correct')}`")

                            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                            with col_met1:
                                st.metric("Hit Rate (Recall@K)", f"{s_hit:.2f}" if s_hit is not None else "N/A")
                            with col_met2:
                                st.metric("MRR", f"{s_mrr:.4f}" if s_mrr is not None else "N/A")
                            with col_met3:
                                st.metric("Token Precision", f"{s_res.get('answer_precision', 0.0):.2%}")
                            with col_met4:
                                st.metric("Token Recall", f"{s_res.get('answer_recall', 0.0):.2%}")

                            s_timings = s_res.get("timings", {})
                            if s_timings:
                                st.caption(f"Latency breakdown: {s_timings}")

                else:
                    st.error(f"❌ Evaluation Run Failed (HTTP {res.status_code}): {res.json().get('detail')}")

            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")

    if btn_view_dataset:
        with st.spinner("Fetching dataset samples..."):
            try:
                res = requests.get(
                    f"{api_base_url}/api/v1/evaluation/dataset",
                    params={"dataset_path": eval_dataset_input.strip()},
                    timeout=10,
                )
                if res.status_code == 200:
                    ds_info = res.json()
                    st.markdown(f"### 📄 Dataset: `{ds_info.get('dataset_path')}` ({ds_info.get('total_samples')} Samples)")
                    for smp in ds_info.get("samples", []):
                        with st.expander(f"[{smp.get('id')}] ({smp.get('category').upper()}) {smp.get('question')}", expanded=False):
                            st.markdown(f"**Expected Answer:** {smp.get('expected_answer')}")
                            st.markdown(f"**Expected Sources:** `{smp.get('expected_sources')}`")
                            st.markdown(f"**Is Answerable:** `{smp.get('is_answerable')}`")
                            if smp.get("conversation_history"):
                                st.markdown(f"**Conversation History Turns:** {len(smp.get('conversation_history'))}")
                else:
                    st.error(f"❌ Failed to load dataset (HTTP {res.status_code}): {res.json().get('detail')}")
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")


