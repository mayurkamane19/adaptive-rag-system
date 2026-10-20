"""Test suite for performance optimizations, model caching, non-blocking execution, and concurrency control."""

import io
import time
import uuid
import threading
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.main import app
from app.services.embedding_service import EmbeddingService
from app.services.reranker_service import RerankerService
from app.services.llm_service import LLMService, LLMServiceError
from app.api.routes import documents, chat
from app.graph import workflow

client = TestClient(app)


def generate_perf_test_pdf(title: str, text: str) -> bytes:
    """Creates a temporary in-memory PDF for performance testing."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 750, title)
    c.drawString(100, 700, text)
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture(autouse=True)
def setup_isolated_env(tmp_path, monkeypatch):
    """Isolates vectorstore and LLM service for API performance tests."""
    temp_db_path = tmp_path / "perf_test_vectorstore"

    monkeypatch.setattr(documents.rag_service.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(documents.rag_service.retrieval_service, "vector_store", None)

    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_store", None)

    monkeypatch.setattr(workflow.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(workflow.retrieval_service, "vector_store", None)

    mock_llm = MagicMock()
    mock_llm.generate_answer.return_value = "Optimized grounded response with low latency [Source 1]."
    monkeypatch.setattr(workflow, "llm_service", mock_llm)
    monkeypatch.setattr(chat.rag_service, "llm_service", mock_llm)


def test_embedding_service_caching():
    """Verify that multiple EmbeddingService instantiations reuse the cached model."""
    svc1 = EmbeddingService()
    svc2 = EmbeddingService()
    assert svc1.get_embeddings() is svc2.get_embeddings()


def test_embedding_service_prewarm():
    """Verify prewarm class method returns an initialized service and populates cache."""
    svc = EmbeddingService.prewarm()
    assert svc is not None
    assert svc.model_name in EmbeddingService._models_cache


def test_reranker_service_caching_and_prewarm():
    """Verify RerankerService caches models and prewarm runs gracefully."""
    mock_ce = MagicMock()
    mock_ce.predict.return_value = [0.9]

    # Pre-populate cache
    RerankerService._models_cache["custom-test-model"] = mock_ce
    svc1 = RerankerService(model_name="custom-test-model")
    svc2 = RerankerService(model_name="custom-test-model")

    assert svc1._get_model() is mock_ce
    assert svc2._get_model() is mock_ce

    # Prewarm custom model
    svc1.prewarm()
    assert svc1._model is mock_ce


def test_llm_service_concurrency_lock():
    """Verify LLMService concurrency lock blocks simultaneous inference calls gracefully."""
    llm = LLMService(timeout=0.2)

    # Acquire lock in advance
    llm._generation_lock.acquire()
    try:
        with pytest.raises(LLMServiceError, match="busy"):
            llm.generate_answer("Test question")
    finally:
        llm._generation_lock.release()


def test_rag_answer_api_returns_timings():
    """Verify POST /api/v1/rag/answer returns diagnostic timing metrics."""
    pdf_bytes = generate_perf_test_pdf(
        "Candidate Profile",
        "Mayur Mohan Kamane holds a Bachelor of Engineering with CGPA 8.44 in Computer Engineering.",
    )
    client.post("/api/v1/documents/index", files={"file": ("profile.pdf", pdf_bytes, "application/pdf")})

    resp = client.post("/api/v1/rag/answer", json={"query": "What is Mayur's degree and CGPA?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "timings" in data
    assert data["timings"] is not None
    assert "retrieval_ms" in data["timings"] or "routing_ms" in data["timings"]


def test_conversational_chat_api_returns_timings():
    """Verify POST /api/v1/rag/chat returns diagnostic timing metrics."""
    pdf_bytes = generate_perf_test_pdf(
        "Technical Skills Doc",
        "Primary skills include Python, FastAPI, PyTorch, FAISS, Docker, LangGraph, and Ollama.",
    )
    client.post("/api/v1/documents/index", files={"file": ("skills.pdf", pdf_bytes, "application/pdf")})

    resp = client.post(
        "/api/v1/rag/chat",
        json={"query": "What technical skills are listed?", "session_id": "test-perf-sess"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "timings" in data
    assert data["timings"] is not None
    assert "retrieval_ms" in data["timings"] or "routing_ms" in data["timings"]


def test_adaptive_rag_context_api_returns_timings():
    """Verify POST /api/v1/rag/adaptive-context returns diagnostic timing metrics."""
    pdf_bytes = generate_perf_test_pdf(
        "Projects Doc",
        "Project 1: Adaptive RAG System with dynamic routing and cross-encoder re-ranking.",
    )
    client.post("/api/v1/documents/index", files={"file": ("projects.pdf", pdf_bytes, "application/pdf")})

    resp = client.post(
        "/api/v1/rag/adaptive-context",
        json={"query": "Compare projects and explain their architecture."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "timings" in data
    assert data["timings"] is not None
    assert "routing_ms" in data["timings"]


def test_adaptive_retrieve_context_does_not_execute_llm(monkeypatch):
    """Regression test: adaptive_retrieve_context must short-circuit after build_context and never invoke LLM generation."""
    mock_llm = MagicMock()
    monkeypatch.setattr(workflow, "llm_service", mock_llm)

    res = chat.rag_service.adaptive_retrieve_context("What is the degree qualification?")
    assert res is not None
    assert "context" in res
    mock_llm.generate_answer.assert_not_called()


def test_ollama_generation_payload_options(monkeypatch):
    """Regression test: LLMService passes num_predict <= 250, keep_alive, and temperature to Ollama."""
    captured_payload = {}

    class MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {
                "response": "Test answer",
                "prompt_eval_count": 25,
                "prompt_eval_duration": 500000000,
                "eval_count": 10,
                "eval_duration": 1000000000,
                "total_duration": 1500000000,
            }

    class MockClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, json=None, **kwargs):
            captured_payload.update(json)
            return MockResponse()

    monkeypatch.setattr("httpx.Client", MockClient)

    llm = LLMService(max_tokens=1000)
    ans = llm.generate_answer("Prompt text", max_tokens=150)
    assert ans == "Test answer"
    assert captured_payload["options"]["num_predict"] == 150
    assert captured_payload["keep_alive"] == "60m"
    assert llm.last_generation_metrics["prompt_eval_count"] == 25
    assert llm.last_generation_metrics["prompt_eval_ms"] == 500.0
    assert llm.last_generation_metrics["eval_count"] == 10
    assert llm.last_generation_metrics["eval_ms"] == 1000.0
    assert llm.last_generation_metrics["tokens_per_second"] == 10.0


def test_rag_answer_records_llm_metrics_in_timings(monkeypatch):
    """Regression test: generate_rag_answer records detailed Ollama evaluation and generation metrics in timings."""
    pdf_bytes = generate_perf_test_pdf(
        "Candidate Profile",
        "Mayur Mohan Kamane holds a Bachelor of Engineering with CGPA 8.44 in Computer Engineering.",
    )
    client.post("/api/v1/documents/index", files={"file": ("profile_timings.pdf", pdf_bytes, "application/pdf")})

    mock_llm = MagicMock()
    mock_llm.generate_answer.return_value = "Detailed answer [Source 1]."
    mock_llm.last_generation_metrics = {
        "prompt_eval_count": 42,
        "prompt_eval_ms": 350.5,
        "eval_count": 18,
        "eval_ms": 1200.0,
        "tokens_per_second": 15.0,
        "num_predict": 250,
    }
    monkeypatch.setattr(workflow, "llm_service", mock_llm)

    res = chat.rag_service.generate_rag_answer("What is Mayur's degree qualification?")
    assert res is not None
    assert "timings" in res
    assert res["timings"].get("prompt_eval_ms") == 350.5
    assert res["timings"].get("eval_ms") == 1200.0
    assert res["timings"].get("eval_count") == 18
    assert res["timings"].get("prompt_eval_count") == 42
    assert res["timings"].get("tokens_per_second") == 15.0


def test_failed_generation_preserves_sources_and_sets_failed_status(monkeypatch):
    """Regression test: LLM generation failure sets generation_status='failed' but preserves retrieved sources."""
    pdf_bytes = generate_perf_test_pdf(
        "Incident Report",
        "Security incident 404 was mitigated by isolating the affected host.",
    )
    client.post("/api/v1/documents/index", files={"file": ("incident.pdf", pdf_bytes, "application/pdf")})

    mock_llm = MagicMock()
    mock_llm.generate_answer.side_effect = LLMServiceError("Ollama connection timed out after 180s")
    monkeypatch.setattr(workflow, "llm_service", mock_llm)

    res = chat.rag_service.generate_rag_answer("How was incident 404 mitigated?")
    assert res is not None
    assert res["generation_status"] == "failed"
    assert "timed out" in res["generation_error"]
    assert len(res["sources"]) > 0
    assert "LLM generation service is currently unavailable" in res["answer"]


def test_failed_generation_does_not_save_conversation_history(monkeypatch):
    """Regression test: Failed LLM turn is never saved to conversation session history."""
    pdf_bytes = generate_perf_test_pdf(
        "Policy Doc",
        "Company policy requires all employees to follow security procedures.",
    )
    client.post("/api/v1/documents/index", files={"file": ("policy_fail.pdf", pdf_bytes, "application/pdf")})

    mock_llm = MagicMock()
    mock_llm.generate_answer.side_effect = LLMServiceError("Inference engine busy")
    monkeypatch.setattr(workflow, "llm_service", mock_llm)

    session_id = f"test-fail-{uuid.uuid4().hex[:8]}"
    res = chat.rag_service.conversational_rag_chat(
        query="What is the company policy?",
        session_id=session_id,
    )
    assert res["generation_status"] == "failed"
    history = chat.rag_service.get_conversation_history(session_id)
    assert len(history) == 0


def test_context_builder_deduplicates_identical_chunks():
    """Regression test: ContextBuilder eliminates identical or duplicate chunks."""
    from langchain_core.documents import Document
    from app.services.context_service import ContextBuilder

    builder = ContextBuilder(max_context_chars=5000)
    doc1 = Document(
        page_content="Common section repeated twice.",
        metadata={"source": "doc.pdf", "page": 1, "chunk_id": "chunk_dup_1", "original_filename": "doc.pdf"},
    )
    doc2 = Document(
        page_content="Common section repeated twice.",
        metadata={"source": "doc.pdf", "page": 1, "chunk_id": "chunk_dup_1", "original_filename": "doc.pdf"},
    )
    doc3 = Document(
        page_content="Unique section content.",
        metadata={"source": "doc.pdf", "page": 2, "chunk_id": "chunk_unique_2", "original_filename": "doc.pdf"},
    )

    result = builder.build_context([doc1, doc2, doc3])
    assert result["included_count"] == 2
    assert result["context"].count("Common section repeated twice.") == 1
    assert result["context"].count("Unique section content.") == 1


def test_health_endpoints_remain_healthy():
    """Regression test: /health and /health/live probes respond HTTP 200."""
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    res_live = client.get("/health/live")
    assert res_live.status_code == 200
    assert res_live.json()["status"] == "alive"


def test_timings_total_ms_excludes_sub_metrics(monkeypatch):
    """Regression test: total_ms sums only discrete pipeline stage latencies and does not add token counts or rates."""
    mock_llm = MagicMock()
    mock_llm.generate_answer.return_value = "Verified test answer [Source 1]."
    mock_llm.last_generation_metrics = {
        "prompt_eval_count": 500,
        "prompt_eval_ms": 250.0,
        "eval_count": 50,
        "eval_ms": 1000.0,
        "tokens_per_second": 10.0,
        "num_predict": 250,
    }
    monkeypatch.setattr(workflow, "llm_service", mock_llm)

    res = chat.rag_service.generate_rag_answer("What is Mayur's degree qualification?")
    timings = res.get("timings", {})
    assert "total_ms" in timings
    # Total ms should not include prompt_eval_count (500) or eval_count (50) or tokens_per_second (10.0)
    # Nor should it add prompt_eval_ms + eval_ms on top of generation_ms
    stage_keys = (
        "load_conversation_ms",
        "rewrite_query_ms",
        "analyze_query_ms",
        "routing_ms",
        "retrieval_ms",
        "rerank_ms",
        "context_ms",
        "prompt_ms",
        "generation_ms",
        "save_conversation_ms",
    )
    expected_sum = round(sum(timings.get(k, 0.0) for k in stage_keys), 2)
    assert timings["total_ms"] == expected_sum


def test_adaptive_rag_context_includes_total_ms():
    """Regression test: adaptive_retrieve_context computes total_ms even when generate_answer is False."""
    res = chat.rag_service.adaptive_retrieve_context("Compare remote and office work policies.")
    timings = res.get("timings", {})
    assert "total_ms" in timings
    assert timings["total_ms"] >= 0.0


def test_streamlit_app_imports_and_ast_integrity():
    """Regression test: frontend/streamlit_app.py imports required standard library modules including 'time'."""
    import ast
    from pathlib import Path

    app_path = Path("frontend/streamlit_app.py")
    assert app_path.is_file(), "frontend/streamlit_app.py must exist"

    source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(app_path))

    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)

    assert "time" in imported_names, "frontend/streamlit_app.py must import 'time' module"
    assert "os" in imported_names, "frontend/streamlit_app.py must import 'os' module"
    assert "requests" in imported_names, "frontend/streamlit_app.py must import 'requests' module"
    assert "st" in imported_names, "frontend/streamlit_app.py must import 'streamlit as st'"




