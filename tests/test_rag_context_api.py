"""Integration test suite for RAG context FastAPI endpoint."""

import io
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.main import app
from app.api.routes import documents, chat

client = TestClient(app)


def generate_context_test_pdf_bytes(title: str, text_content: str) -> bytes:
    """Generates an in-memory PDF file for context integration tests."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(100, 750, title)
    c.drawString(100, 700, text_content)
    c.showPage()
    c.save()
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def override_vectorstore_dir(tmp_path, monkeypatch):
    """Fixture ensuring tests run in an isolated temporary directory."""
    temp_db_path = tmp_path / "context_api_test_vectorstore"
    temp_docs_path = tmp_path / "context_api_test_documents"

    monkeypatch.setattr(documents.rag_service.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(documents.rag_service.retrieval_service, "vector_store", None)

    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_store", None)


def test_rag_context_endpoint_success():
    """Test POST /api/v1/rag/context returns structured context and sources payload."""
    pdf_bytes = generate_context_test_pdf_bytes(
        title="Incident Response Plan 2026",
        text_content="All security incidents must be reported to the SOC within 15 minutes of detection.",
    )
    # Index document first
    client.post(
        "/api/v1/documents/index",
        files={"file": ("incident_plan.pdf", pdf_bytes, "application/pdf")},
    )

    # Request RAG context
    payload = {"query": "How quickly must security incidents be reported?", "top_k": 2}
    response = client.post("/api/v1/rag/context", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["query"] == payload["query"]
    assert data["retrieved_count"] >= 1
    assert data["included_count"] >= 1
    assert len(data["sources"]) >= 1

    assert "[Source 1]" in data["context"]
    assert "Incident Response Plan 2026" in data["context"] or "reported to the SOC" in data["context"]
    assert data["sources"][0]["original_filename"] == "incident_plan.pdf"


def test_rag_context_endpoint_empty_query():
    """Test POST /api/v1/rag/context returns HTTP 400 when query string is empty."""
    response = client.post("/api/v1/rag/context", json={"query": "   ", "top_k": 2})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"].lower()


def test_rag_context_endpoint_invalid_top_k():
    """Test POST /api/v1/rag/context returns HTTP 422 when top_k is outside allowed bounds."""
    response = client.post("/api/v1/rag/context", json={"query": "Valid query", "top_k": 100})
    assert response.status_code == 422


def test_rag_context_endpoint_no_vectorstore(monkeypatch, tmp_path):
    """Test POST /api/v1/rag/context returns status no_results when no vector store exists."""
    non_existent = tmp_path / "empty_non_existent_dir"
    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_db_path", non_existent)
    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_store", None)

    response = client.post("/api/v1/rag/context", json={"query": "What is the policy?", "top_k": 2})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "no_results"
    assert data["context"] == ""
    assert data["retrieved_count"] == 0
    assert data["included_count"] == 0
