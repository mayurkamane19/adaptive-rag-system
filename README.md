# Enterprise Adaptive Retrieval-Augmented Generation (RAG) System

[![CI Pipeline](https://github.com/mayurkamane19/adaptive-rag-system/actions/workflows/ci.yml/badge.svg)](https://github.com/mayurkamane19/adaptive-rag-system)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful%20Workflow-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Tests Passing](https://img.shields.io/badge/Tests-173%2F173%20Passed-brightgreen.svg)](https://pytest.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

An enterprise-grade, stateful **Adaptive Retrieval-Augmented Generation (RAG)** system featuring **deterministic query routing**, two-stage semantic retrieval with **FAISS** and **Cross-Encoder re-ranking**, multi-turn **conversational memory with query rewriting**, and **Ollama local LLM inference**.

Designed for high accuracy, grounded document question-answering, strict anti-hallucination guardrails, and enterprise production readiness.

---

## 🏛️ System Architecture

```text
                                 [USER REQUEST]
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              [Single-Turn RAG Query]      [Multi-Turn Conversation]
                         │                             │
                         │                   [Load Session History]
                         │                   (In-Memory + Disk JSON)
                         │                             │
                         │                 [Query Rewriter (LLM)]
                         │               (Standalone Search Query)
                         │                             │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         [Deterministic Query Router]
                         (Heuristic Query Classifier)
                                        │
                       ┌────────────────┴────────────────┐
                       ▼                                 ▼
             [SIMPLE ROUTE (Low Cost)]      [COMPLEX ROUTE (Deep Synthesis)]
             - Candidate Pool: K = 8        - Candidate Pool: K = 16
             - Final Context: Top-K = 4     - Final Context: Top-K = 4
                       │                                 │
                       └────────────────┬────────────────┘
                                        │
                                        ▼
                         [First-Stage Retrieval: FAISS]
                         (IndexFlatL2, all-MiniLM-L6-v2)
                                        │
                                        ▼
                      [Second-Stage: Cross-Encoder Reranker]
                             (BAAI/bge-reranker-base)
                                        │
                                        ▼
                            [Context Builder & Dedup]
                       - Deduplicates overlapping chunks
                       - Character bounded (12,000 chars)
                                        │
                                        ▼
                            [Grounded Prompt Engine]
                       - Strict grounding rules 1–7
                       - Preamble suppression
                       - Source citation mapping
                                        │
                                        ▼
                          [Local LLM Inference Engine]
                       (Ollama / llama3:latest via HTTP)
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
               [Grounded Answer Output]      [Save Session History]
               - Accurate facts              - Atomic disk persistence
               - Supporting citations        - Session isolation
               - Stage timing metrics
```

---

## ✨ Core Capabilities

- **Adaptive Routing Engine**: Deterministic classifier dynamically allocates retrieval budgets (`SIMPLE` with 8 candidate chunks vs. `COMPLEX` with 16 candidate chunks) based on query semantics, analytical keywords, and syntactic structure.
- **Two-Stage Precision Retrieval**:
  - **Stage 1 (Bi-Encoder / FAISS)**: High-speed candidate vector retrieval over document embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
  - **Stage 2 (Cross-Encoder Re-Ranking)**: Joint-attention relevance scoring via `BAAI/bge-reranker-base`, ensuring the most relevant chunks are placed in top context positions.
- **Multi-Turn Conversational Memory**:
  - In-memory cache with atomic JSON disk persistence in `data/conversations/`.
  - History-aware query reformulator resolves pronouns and references before retrieval.
  - Dedicated session management (`DELETE /api/v1/rag/chat/{session_id}`).
- **Strict Grounding & Anti-Hallucination**:
  - Enforces strict factual adherence: if retrieved context does not contain sufficient facts, the model returns:
    > *"I couldn't find enough information in the provided documents to answer this question."*
  - Tested against unanswerable out-of-domain questions with **zero hallucination**.
  - Preamble suppression eliminates conversational filler (`"Based on the provided context..."`), saving CPU compute time and avoiding truncation.
- **Enterprise Production Hardening**:
  - Kubernetes/Docker probes: `/health` (system), `/health/live` (liveness), `/health/ready` (subsystem readiness).
  - Rate limiting (sliding window per IP) and optional `X-API-Key` authentication.
  - File upload guardrails: 25MB max size limit, strict `.pdf` validation, duplicate document detection.
  - Sanitized error responses: internal exceptions never leak stack traces or private document contents.

---

## 🛠️ Technology Stack

| Layer | Component | Description |
|---|---|---|
| **Backend Framework** | [FastAPI](https://fastapi.tiangolo.com/) | Asynchronous, OpenAPI-compliant REST API engine |
| **Server Engine** | [Uvicorn](https://www.uvicorn.org/) | High-performance ASGI server with threadpool isolation |
| **Workflow Graph** | [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful, inspectable DAG workflow orchestration |
| **Vector Database** | [FAISS CPU](https://github.com/facebookresearch/faiss) | Dense vector index (`IndexFlatL2`) with local disk persistence |
| **Embedding Model** | `all-MiniLM-L6-v2` | 384-dimensional dense semantic embeddings |
| **Re-Ranking Model** | `BAAI/bge-reranker-base` | Sentence Transformers Cross-Encoder for joint-attention scoring |
| **Local LLM Engine** | [Ollama](https://ollama.com/) | Local CPU/GPU inference serving `llama3:latest` (8.0B parameters) |
| **Frontend UI** | [Streamlit](https://streamlit.io/) | Interactive multi-tab web dashboard & evaluation workbench |
| **Testing & QA** | [pytest](https://pytest.org/) | 173 automated unit, integration, and regression tests |
| **Containerization** | [Docker](https://www.docker.com/) | Multi-stage Dockerfiles with unprivileged `appuser` and health checks |

---

## 📁 Repository Structure

```text
adaptive-rag-system/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── health.py             # Health probes (/health, /health/live, /health/ready)
│   │       ├── documents.py          # PDF upload and vector indexing endpoints
│   │       ├── chat.py               # RAG search, context, answer, and conversational endpoints
│   │       └── evaluation.py         # RAG evaluation benchmark execution endpoints
│   ├── core/
│   │   ├── config.py                 # Pydantic Settings configuration from .env
│   │   ├── logging_config.py         # Structured application logger
│   │   └── security.py               # API key validation, rate limiting, and request sanitization
│   ├── graph/
│   │   ├── state.py                  # LangGraph RAGGraphState definition
│   │   └── workflow.py               # Compiled LangGraph DAG state machine
│   ├── models/
│   │   └── message.py                # Conversation message domain models
│   ├── prompts/
│   │   ├── rag_prompts.py            # Grounded RAG system instructions and templates
│   │   └── conversation_prompts.py   # Query rewriting and multi-turn prompt templates
│   ├── schemas/
│   │   ├── common.py                 # Base API response schemas
│   │   ├── document.py               # Document upload and indexing schemas
│   │   ├── chat.py                   # RAG search, answer, and chat schemas
│   │   └── evaluation.py             # RAG evaluation metrics schemas
│   ├── services/
│   │   ├── ingestion_service.py      # PDF parsing, cleaning, and character-overlap chunking
│   │   ├── embedding_service.py      # Cached HuggingFace embeddings
│   │   ├── retrieval_service.py      # FAISS vector store management and duplicate checking
│   │   ├── reranker_service.py       # Cross-Encoder re-ranking service
│   │   ├── context_service.py        # Context builder with chunk deduplication
│   │   ├── query_router.py           # Deterministic heuristic query classifier
│   │   ├── query_rewriter.py         # History-aware query reformulator
│   │   ├── llm_service.py            # Ollama client with concurrency locking & metrics
│   │   ├── conversation_service.py   # Multi-turn session persistence engine
│   │   ├── rag_service.py            # High-level RAG facade
│   │   └── evaluation_service.py     # Evaluation pipeline (Answer F1, Grounding, Precision)
│   ├── utils/
│   │   └── file_utils.py             # File extension validation and path traversal defenses
│   └── main.py                       # FastAPI application entry point and lifespan manager
├── frontend/
│   └── streamlit_app.py              # 7-tab Streamlit dashboard and diagnostics UI
├── data/
│   ├── documents/                    # Raw uploaded PDF documents
│   ├── vectorstore/                  # Persistent FAISS index (index.faiss, index.pkl)
│   ├── conversations/                # Saved JSON chat sessions
│   └── evaluation/                   # Default benchmark dataset (eval_dataset.json)
├── scripts/
│   ├── start_system.ps1              # Windows PowerShell startup script
│   ├── stop_system.ps1               # Windows PowerShell graceful stop script
│   └── check_health.ps1              # Subsystem health & connectivity check
├── tests/                            # 173 automated pytest test cases
├── .github/workflows/ci.yml          # GitHub Actions CI workflow
├── Dockerfile                        # Backend production container
├── Dockerfile.streamlit              # Frontend Streamlit container
├── docker-compose.yml                # Multi-service container orchestration
├── .env.example                      # Environment configuration template
├── .gitignore                        # Git ignore rules protecting data and secrets
├── requirements.txt                  # Python dependencies
└── run.py                            # CLI entrypoint for Uvicorn
```

---

## 🚀 Quickstart Guide (Windows)

### 1. Prerequisites
- **Python**: 3.10 or 3.11
- **Ollama**: Installed from [ollama.com](https://ollama.com)
- **Model**: `llama3:latest` pulled via Ollama:
  ```powershell
  ollama pull llama3
  ```

### 2. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/mayurkamane19/adaptive-rag-system.git
cd "adaptive-rag-system"

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Copy configuration
Copy-Item .env.example .env
```

### 3. One-Click PowerShell Startup
Use the included automation script to verify dependencies and start all services:
```powershell
.\scripts\start_system.ps1
```

Or start the services individually in separate PowerShell terminals:

```powershell
# Terminal 1: Ensure Ollama is running
ollama serve

# Terminal 2: Start FastAPI Backend (Port 8000)
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal 3: Start Streamlit Frontend (Port 8501)
.\.venv\Scripts\Activate.ps1
streamlit run frontend/streamlit_app.py
```

### 4. Health Verification
Verify all services are active and healthy:
```powershell
.\scripts\check_health.ps1
```

### 5. Access Endpoints
- **Streamlit Web UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Interactive Docs (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Readiness Health Probe**: [http://127.0.0.1:8000/health/ready](http://127.0.0.1:8000/health/ready)

### 6. Graceful Shutdown
To stop the FastAPI and Streamlit processes without disturbing other services:
```powershell
.\scripts\stop_system.ps1
```

---

## 📡 REST API Reference

### 1. Document Indexing
`POST /api/v1/documents/index` (Multipart Form)
- Indexes a PDF document into FAISS with automatic duplicate detection.
- *(Note: `sample_document.pdf` is an illustrative placeholder; provide the path to any local PDF file you wish to index).*
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/documents/index" \
     -F "file=@sample_document.pdf"
```

### 2. Grounded RAG Answer Generation
`POST /api/v1/rag/answer`
```json
{
  "query": "What degree did Mayur complete, and what was his CGPA?"
}
```
**Response:**
```json
{
  "query": "What degree did Mayur complete, and what was his CGPA?",
  "route": "simple",
  "route_reason": "Query matches simple single-fact retrieval pattern.",
  "route_confidence": 0.85,
  "top_k": 4,
  "candidate_count": 8,
  "reranked_count": 4,
  "reranker_enabled": true,
  "reranker_model": "BAAI/bge-reranker-base",
  "reranker_status": "success",
  "answer": "Bachelor of Technology (Information Technology) with a CGPA of 8.39.",
  "sources": [
    {
      "source": "data/documents/sample_document.pdf",
      "page": 1,
      "chunk_id": "sample_document_p1_005",
      "original_filename": "sample_document.pdf"
    }
  ],
  "status": "success",
  "generation_status": "success",
  "timings": {
    "routing_ms": 0.12,
    "retrieval_ms": 0.22,
    "rerank_ms": 3730.0,
    "prompt_ms": 0.15,
    "prompt_eval_ms": 200.0,
    "eval_ms": 3800.0,
    "tokens_per_second": 4.74
  }
}
```

### 3. Multi-Turn Conversational RAG
`POST /api/v1/rag/chat`
```json
{
  "query": "What projects has he worked on?",
  "session_id": "session-interview-1"
}
```

### 4. Re-Ranked Semantic Search
`POST /api/v1/search/reranked`
```json
{
  "query": "Generative AI and LangGraph experience",
  "candidate_k": 8,
  "top_k": 4
}
```

---

## 🧪 Automated Testing & QA

The repository includes **173 automated tests** covering unit, integration, error-handling, concurrency, and security scenarios with 100% pass rate:

```powershell
# Run the complete test suite
pytest
```

**Test Execution Summary:**
```text
collected 173 items

tests/test_adaptive_rag_api.py ....                                      [  2%]
tests/test_context_service.py .....                                      [  5%]
tests/test_conversation_service.py .....                                 [  8%]
tests/test_conversational_chat_api.py ....                               [ 10%]
tests/test_documents_api.py ......                                       [ 13%]
tests/test_embeddings.py ....                                            [ 16%]
tests/test_evaluation_api.py ......                                      [ 19%]
tests/test_evaluation_metrics.py .....................                   [ 31%]
tests/test_evaluation_service.py .........                               [ 36%]
tests/test_generation_workflow.py ..                                     [ 38%]
tests/test_health.py .                                                   [ 38%]
tests/test_ingestion.py ...........                                      [ 45%]
tests/test_llm_service.py ..........                                     [ 50%]
tests/test_multi_doc_retrieval.py ...                                    [ 52%]
tests/test_performance_and_reliability.py .................              [ 62%]
tests/test_phase11_release_readiness.py .............                    [ 69%]
tests/test_production_hardening.py ..........                            [ 75%]
tests/test_prompt_builder.py ..                                          [ 76%]
tests/test_query_rewriter.py ...                                         [ 78%]
tests/test_query_router.py ...                                           [ 80%]
tests/test_rag_answer_api.py ........                                    [ 84%]
tests/test_rag_context_api.py ....                                       [ 87%]
tests/test_rag_service.py ...                                            [ 89%]
tests/test_reranked_search_api.py ..                                     [ 90%]
tests/test_reranker_service.py ...                                       [ 91%]
tests/test_reranking_workflow.py ..                                      [ 93%]
tests/test_retrieval.py ....                                             [ 95%]
tests/test_search_api.py ....                                            [ 97%]
tests/test_workflow.py ....                                              [100%]

======================= 173 passed in 80.87s (0:01:20) ========================
```

---

## ⚡ Performance Profiling & Hardware Realities

During Phase 10 and 11, extensive empirical profiling was conducted on the host machine (**12th Gen Intel Core i5-1235U, 10 cores, 15W TDP, 16GB RAM, CPU-only**):

### Key Empirical Findings:
1. **Ollama Thread Scheduling**: Leaving `num_thread=None` (Ollama auto-schedule) delivered the best throughput (**21.74 tok/s prompt eval, 5.19 tok/s generation**). Manual thread pinning degraded throughput by up to 55% due to Intel Alder Lake hybrid P-core / E-core synchronization overhead.
2. **KV Cache Prefix Reuse**:
   - **Cold Requests**: Evaluating ~1,200–1,330 prompt tokens on a 15W CPU requires **46s to 91s** (~14–22 tokens/second).
   - **Warm Repeated Requests**: When prompt prefixes match in Ollama's KV cache, prompt evaluation completes in **0.18s to 0.20s** (**99.8% latency reduction**).
3. **Measured Live Timings on Sample Resume (`sample_document.pdf`)**:
   - **Degree & CGPA**: Cold: 102.37s | **Warm: 10.16s** (18 tokens generated @ 4.74 tok/s)
   - **Projects**: Cold: 63.02s | **Warm: 8.12s** (14 tokens generated @ 5.11 tok/s)
   - **Technical Skills**: Cold: 98.36s | **Warm: 38.44s** (126 tokens generated @ 3.80 tok/s)
   - **AI/ML vs GenAI**: Cold: 101.20s | **Warm: 16.52s** (47 tokens generated @ 4.37 tok/s)
   - **Detailed Summary**: Cold: 122.04s | **Warm: 61.39s** (206 tokens generated @ 3.70 tok/s)
4. **Anti-Hallucination Verification**: Tested on unanswerable query (*"What is Mayur's passport expiry date?"*) -> The system returned:
   > *"I couldn't find enough information in the provided documents to answer this question."*

*Full technical audit: see [PERFORMANCE_OPTIMIZATION_AUDIT.md](PERFORMANCE_OPTIMIZATION_AUDIT.md).*

---

## 🔧 Troubleshooting Guide

| Issue | Root Cause | Solution |
|---|---|---|
| **Ollama Offline** (`Could not connect to Ollama server...`) | Ollama daemon is not running on port 11434 | Run `ollama serve` in a terminal or start the Ollama desktop app. Verify `http://localhost:11434`. |
| **Model Missing** (`model 'llama3' not found`) | The model has not been downloaded | Run `ollama pull llama3` in PowerShell. |
| **Port 8000 or 8501 Busy** | Another process is holding the port | Run `.\scripts\stop_system.ps1` to stop prior instances, or check `Get-NetTCPConnection -LocalPort 8000`. |
| **Timeout on Complex Query** | CPU inference on long answers took >60s | Timeouts have been increased to 180s in `.env` (`OLLAMA_TIMEOUT_SECONDS=180.0`) and Streamlit. |
| **Duplicate Document Warning** | Document with same filename was already indexed | The system preserves existing vectors to prevent index bloat. |

---

## 🔒 Security & Data Privacy

- **Safe Local Defaults**: By default, bindings listen on `127.0.0.1` and `API_KEY` is optional for development. In production, configure `ALLOWED_ORIGINS` and set `API_KEY`.
- **Zero Data Leakage**: `.env`, `data/vectorstore/*`, `data/documents/*`, and `data/conversations/*` are protected in `.gitignore`.
- **Sanitized Logging**: Document contents, session messages, and user queries are never logged at WARN/ERROR levels, and exceptions are sanitized to prevent stack trace disclosure.

---

## 📜 License & Credits

Developed by **Mayur Mohan Kamane** as an end-to-end showcase of modern enterprise AI engineering, RAG architecture, and production readiness.
