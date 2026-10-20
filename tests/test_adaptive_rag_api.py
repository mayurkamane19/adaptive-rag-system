"""Integration test suite for Adaptive RAG LangGraph REST API endpoint."""

import io
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.main import app
from app.api.routes import documents, chat
from app.graph import workflow

client = TestClient(app)


def generate_adaptive_test_pdf_bytes(title: str, text_content: str) -> bytes:
    """Generates an in-memory PDF file for adaptive RAG API tests."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(100, 750, title)
    c.drawString(100, 700, text_content)
    c.showPage()
    c.save()
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def override_vectorstore_dir(tmp_path, monkeypatch):
    """Fixture ensuring adaptive RAG API tests execute against an isolated temporary vectorstore directory."""
    temp_db_path = tmp_path / "adaptive_api_test_vectorstore"

    monkeypatch.setattr(documents.rag_service.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(documents.rag_service.retrieval_service, "vector_store", None)

    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_store", None)

    monkeypatch.setattr(workflow.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(workflow.retrieval_service, "vector_store", None)


def test_adaptive_rag_endpoint_simple_query():
    """Test POST /api/v1/rag/adaptive-context routes simple queries to simple branch."""
    pdf_bytes = generate_adaptive_test_pdf_bytes(
        title="Leave Policy 2026",
        text_content="Full-time employees receive 20 days of annual paid vacation leave.",
    )
    # Index document
    client.post("/api/v1/documents/index", files={"file": ("leave_policy.pdf", pdf_bytes, "application/pdf")})

    # Request simple adaptive RAG context
    payload = {"query": "What is the annual leave allowance?"}
    response = client.post("/api/v1/rag/adaptive-context", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "simple"
    assert data["top_k"] == 4
    assert data["status"] == "success"
    assert "20 days of annual paid vacation leave" in data["context"]
    assert data["sources"][0]["original_filename"] == "leave_policy.pdf"


def test_adaptive_rag_endpoint_complex_query():
    """Test POST /api/v1/rag/adaptive-context routes complex queries to complex branch with expanded top_k depth."""
    pdf_bytes = generate_adaptive_test_pdf_bytes(
        title="Workplace Policies 2026",
        text_content="Compare remote work security guidelines versus office network rules.",
    )
    # Index document
    client.post("/api/v1/documents/index", files={"file": ("workplace_policies.pdf", pdf_bytes, "application/pdf")})

    # Request complex adaptive RAG context
    payload = {"query": "Compare remote work security guidelines versus office network rules."}
    response = client.post("/api/v1/rag/adaptive-context", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "complex"
    assert data["top_k"] in (8, 6, 5, 4)
    assert data["status"] == "success"
    assert "Complex query detected" in data["route_reason"]
    assert data["route_confidence"] >= 0.60


def test_adaptive_rag_endpoint_empty_query():
    """Test POST /api/v1/rag/adaptive-context returns HTTP 400 for empty queries."""
    response = client.post("/api/v1/rag/adaptive-context", json={"query": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"].lower()


def test_adaptive_rag_endpoint_missing_vectorstore(tmp_path, monkeypatch):
    """Test POST /api/v1/rag/adaptive-context handles missing vector store gracefully."""
    non_existent = tmp_path / "non_existent_adaptive_dir"

    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_db_path", non_existent)
    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_store", None)
    monkeypatch.setattr(workflow.retrieval_service, "vector_db_path", non_existent)
    monkeypatch.setattr(workflow.retrieval_service, "vector_store", None)

    response = client.post("/api/v1/rag/adaptive-context", json={"query": "Any query"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "no_results"
    assert data["context"] == ""
