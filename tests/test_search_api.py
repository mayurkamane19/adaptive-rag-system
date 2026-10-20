"""Integration test suite for document indexing and semantic search FastAPI endpoints."""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.main import app
from app.api.routes import documents, chat
from app.services.retrieval_service import RetrievalService

client = TestClient(app)


def generate_search_test_pdf_bytes(title: str, text_content: str) -> bytes:
    """Generates a valid in-memory PDF file for indexing API tests."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(100, 750, title)
    c.drawString(100, 700, text_content)
    c.showPage()
    c.save()
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def override_vectorstore_dir(tmp_path, monkeypatch):
    """Fixture ensuring vectorstore tests operate in a temporary isolated directory."""
    temp_db_path = tmp_path / "api_test_vectorstore"
    temp_docs_path = tmp_path / "api_test_documents"

    monkeypatch.setattr(documents.rag_service.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(documents.rag_service.retrieval_service, "vector_store", None)

    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_store", None)


def test_index_document_endpoint():
    """Test POST /api/v1/documents/index ingests PDF and indexes chunks into FAISS."""
    pdf_bytes = generate_search_test_pdf_bytes(
        title="Cybersecurity Standards 2026",
        text_content="Multi-factor authentication is mandatory across all enterprise systems.",
    )
    files = {"file": ("cyber_sec.pdf", pdf_bytes, "application/pdf")}

    response = client.post("/api/v1/documents/index", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("indexed", "duplicate")
    assert data["filename"] == "cyber_sec.pdf"
    assert data["page_count"] == 1
    assert data["chunk_count"] >= 1


def test_semantic_search_endpoint():
    """Test POST /api/v1/search retrieves matching chunks after indexing."""
    # First index a document
    pdf_bytes = generate_search_test_pdf_bytes(
        title="Enterprise Security Policy",
        text_content="All remote employee connections must pass through corporate VPN tunnels.",
    )
    client.post(
        "/api/v1/documents/index",
        files={"file": ("security_policy.pdf", pdf_bytes, "application/pdf")},
    )

    # Search query
    payload = {"query": "How should remote employees connect to corporate networks?", "top_k": 2}
    response = client.post("/api/v1/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["query"] == payload["query"]
    assert len(data["results"]) >= 1

    first_result = data["results"][0]
    assert "corporate VPN tunnels" in first_result["content"]
    assert first_result["original_filename"] == "security_policy.pdf"
    assert first_result["page"] == 1


def test_semantic_search_empty_query():
    """Test POST /api/v1/search returns HTTP 400 for empty or whitespace queries."""
    response = client.post("/api/v1/search", json={"query": "   ", "top_k": 2})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"].lower()


def test_semantic_search_missing_vectorstore(monkeypatch, tmp_path):
    """Test POST /api/v1/search returns HTTP 404 when no vector store exists."""
    non_existent = tmp_path / "empty_dir_no_store"
    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_db_path", non_existent)
    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_store", None)

    response = client.post("/api/v1/search", json={"query": "Anything", "top_k": 2})
    assert response.status_code == 404
    assert "does not exist" in response.json()["detail"].lower()
