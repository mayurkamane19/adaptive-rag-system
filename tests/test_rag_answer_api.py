"""Integration test suite for RAG answer generation REST API endpoint."""

import io
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.main import app
from app.api.routes import documents, chat
from app.graph import workflow

client = TestClient(app)


def generate_answer_test_pdf_bytes(title: str, text_content: str) -> bytes:
    """Generates an in-memory PDF file for RAG answer API tests."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(100, 750, title)
    c.drawString(100, 700, text_content)
    c.showPage()
    c.save()
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def override_vectorstore_dir_and_llm(tmp_path, monkeypatch):
    """Fixture ensuring tests run against an isolated vectorstore and a mocked LLM service."""
    temp_db_path = tmp_path / "answer_api_test_vectorstore"

    monkeypatch.setattr(documents.rag_service.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(documents.rag_service.retrieval_service, "vector_store", None)

    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_store", None)

    monkeypatch.setattr(workflow.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(workflow.retrieval_service, "vector_store", None)

    mock_llm = MagicMock()
    mock_llm.generate_answer.return_value = "Employees receive 20 days of paid vacation leave annually [Source 1]."
    monkeypatch.setattr(workflow, "llm_service", mock_llm)
    monkeypatch.setattr(chat.rag_service, "llm_service", mock_llm)


def test_rag_answer_endpoint_success():
    """Test POST /api/v1/rag/answer returns grounded LLM answer and source metadata."""
    pdf_bytes = generate_answer_test_pdf_bytes(
        title="Employee Handbook 2026",
        text_content="Employees receive 20 days of paid vacation leave annually.",
    )
    # Index document
    client.post("/api/v1/documents/index", files={"file": ("handbook.pdf", pdf_bytes, "application/pdf")})

    # Request RAG answer
    payload = {"query": "What is the annual paid vacation leave allowance?"}
    response = client.post("/api/v1/rag/answer", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["query"] == payload["query"]
    assert "20 days of paid vacation leave" in data["answer"]
    assert data["route"] in ("simple", "complex")
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["original_filename"] == "handbook.pdf"
    assert data["llm_provider"] == "ollama"
    assert data["llm_model"] == "llama3"


def test_rag_answer_endpoint_empty_query():
    """Test POST /api/v1/rag/answer returns HTTP 400 for empty query."""
    response = client.post("/api/v1/rag/answer", json={"query": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"].lower()


def test_rag_answer_endpoint_no_vectorstore(tmp_path, monkeypatch):
    """Test POST /api/v1/rag/answer handles missing vector store returning grounded fallback answer."""
    non_existent = tmp_path / "empty_non_existent_dir_answer"

    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_db_path", non_existent)
    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_store", None)
    monkeypatch.setattr(workflow.retrieval_service, "vector_db_path", non_existent)
    monkeypatch.setattr(workflow.retrieval_service, "vector_store", None)

    response = client.post("/api/v1/rag/answer", json={"query": "Any question"})
    assert response.status_code == 200
    data = response.json()
    assert "couldn't find enough information" in data["answer"].lower()
    assert data["retrieved_count"] == 0
    assert data["included_count"] == 0


def test_rag_answer_endpoint_llm_failure_preserves_sources(monkeypatch):
    """Test that when LLM service fails (e.g. Ollama offline), retrieval sources are preserved in response."""
    from app.services.llm_service import LLMServiceError

    pdf_bytes = generate_answer_test_pdf_bytes(
        title="Employee Handbook 2026",
        text_content="Employees receive 20 days of paid vacation leave annually.",
    )
    client.post("/api/v1/documents/index", files={"file": ("handbook.pdf", pdf_bytes, "application/pdf")})

    failing_llm = MagicMock()
    failing_llm.generate_answer.side_effect = LLMServiceError(
        "Could not connect to Ollama server at 'http://localhost:11434'. Please ensure Ollama is installed and running."
    )
    monkeypatch.setattr(workflow, "llm_service", failing_llm)
    monkeypatch.setattr(chat.rag_service, "llm_service", failing_llm)

    payload = {"query": "What is the annual paid vacation leave allowance?"}
    response = client.post("/api/v1/rag/answer", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["generation_status"] == "failed"
    assert "LLM generation service is currently unavailable" in data["answer"]
    assert "Could not connect to Ollama server" in data["generation_error"]
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["original_filename"] == "handbook.pdf"
    assert data["retrieved_count"] >= 1
    assert data["included_count"] >= 1


def test_rag_answer_multi_project_question_complex_routing(monkeypatch):
    """Regression test: multi-project summary query routes to complex, allocates top_k=6, and returns grounded answer."""
    from app.core.config import settings

    pdf_bytes = generate_answer_test_pdf_bytes(
        title="Engineering Projects Portfolio",
        text_content=(
            "PROJECT Alpha: Autonomous drone navigation system for warehouse inventory management. "
            "PROJECT Beta: Real-time fraud detection pipeline using graph neural networks."
        ),
    )
    client.post("/api/v1/documents/index", files={"file": ("projects_portfolio.pdf", pdf_bytes, "application/pdf")})

    mock_llm = MagicMock()
    mock_llm.generate_answer.return_value = (
        "1. PROJECT Alpha: Autonomous drone navigation system for warehouse inventory.\n"
        "2. PROJECT Beta: Real-time fraud detection pipeline using graph neural networks."
    )
    monkeypatch.setattr(workflow, "llm_service", mock_llm)
    monkeypatch.setattr(chat.rag_service, "llm_service", mock_llm)

    payload = {
        "query": "Summarize all the projects mentioned in my resume. Include each project name and its main purpose. If the resume does not provide enough details, say so."
    }
    response = client.post("/api/v1/rag/answer", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["route"] == "complex"
    assert data["top_k"] == settings.RERANKER_TOP_K_COMPLEX
    assert "PROJECT Alpha" in data["answer"]
    assert "PROJECT Beta" in data["answer"]
    assert len(data["sources"]) >= 1


def test_rag_answer_factual_question_grounding(monkeypatch):
    """Regression test: direct factual query routes to simple route and retrieves exact facts."""
    pdf_bytes = generate_answer_test_pdf_bytes(
        title="Academic Profile",
        text_content="Degree: Bachelor of Technology in Information Technology. CGPA: 8.39 out of 10.",
    )
    client.post("/api/v1/documents/index", files={"file": ("academic_profile.pdf", pdf_bytes, "application/pdf")})

    mock_llm = MagicMock()
    mock_llm.generate_answer.return_value = "Mayur's degree is B.Tech in Information Technology with a CGPA of 8.39 [Source 1]."
    monkeypatch.setattr(workflow, "llm_service", mock_llm)
    monkeypatch.setattr(chat.rag_service, "llm_service", mock_llm)

    payload = {"query": "What is Mayur's degree and CGPA?"}
    response = client.post("/api/v1/rag/answer", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["route"] == "simple"
    assert data["top_k"] == 4
    assert "8.39" in data["answer"]
    assert len(data["sources"]) >= 1


def test_rag_answer_unanswerable_question_grounding(monkeypatch):
    """Regression test: query asking about unmentioned facts returns grounded fallback answer without hallucinations."""
    pdf_bytes = generate_answer_test_pdf_bytes(
        title="Leave Policy",
        text_content="Annual leave is 20 days per year for full-time employees.",
    )
    client.post("/api/v1/documents/index", files={"file": ("leave_doc.pdf", pdf_bytes, "application/pdf")})

    mock_llm = MagicMock()
    mock_llm.generate_answer.return_value = "I couldn't find enough information in the provided documents to answer this question."
    monkeypatch.setattr(workflow, "llm_service", mock_llm)
    monkeypatch.setattr(chat.rag_service, "llm_service", mock_llm)

    payload = {"query": "What is the secret rocket propulsion code for Mars Mission?"}
    response = client.post("/api/v1/rag/answer", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "couldn't find enough information" in data["answer"].lower()


def test_rag_prompts_include_single_item_clarification_rule():
    """Verify that SYSTEM_RAG_INSTRUCTIONS contains Rule 8 for plural requests with single items."""
    from app.prompts.rag_prompts import SYSTEM_RAG_INSTRUCTIONS, build_rag_prompt

    assert "8. If the question asks for a list or summary of multiple items" in SYSTEM_RAG_INSTRUCTIONS
    prompt = build_rag_prompt("Summarize all projects", "Context: Only one project.")
    assert "8. If the question asks for a list or summary of multiple items" in prompt

