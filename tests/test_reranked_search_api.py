"""Integration test suite for two-stage re-ranked search REST API endpoint."""

import io
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.main import app
from app.api.routes import documents, chat
from app.graph import workflow

client = TestClient(app)


def generate_rerank_test_pdf_bytes(title: str, text_content: str) -> bytes:
    """Generates an in-memory PDF file for re-ranked search API tests."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(100, 750, title)
    c.drawString(100, 700, text_content)
    c.showPage()
    c.save()
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def override_vectorstore_dir_and_reranker(tmp_path, monkeypatch):
    """Fixture ensuring tests run against an isolated vectorstore and a mocked re-ranker service."""
    temp_db_path = tmp_path / "reranked_search_test_vectorstore"

    monkeypatch.setattr(documents.rag_service.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(documents.rag_service.retrieval_service, "vector_store", None)

    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_store", None)

    monkeypatch.setattr(workflow.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(workflow.retrieval_service, "vector_store", None)

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda query, documents, top_k: {
        "status": "success",
        "candidate_count": len(documents),
        "reranked_count": min(len(documents), top_k),
        "reranker_enabled": True,
        "reranker_model": "BAAI/bge-reranker-base",
        "error": None,
        "reranked_documents": [
            {
                "rank": idx,
                "rerank_score": round(0.95 - (idx * 0.1), 4),
                "retrieval_score": round(0.4 + (idx * 0.05), 4),
                "content": f"Chunk content {idx}",
                "source": "data/documents/handbook.pdf",
                "page": 1,
                "chunk_id": f"c_{idx}",
                "original_filename": "handbook.pdf",
            }
            for idx in range(1, min(len(documents), top_k) + 1)
        ],
    }
    monkeypatch.setattr(chat.rag_service, "reranker_service", mock_reranker)


def test_reranked_search_endpoint_success():
    """Test POST /api/v1/search/reranked returns ordered two-stage re-ranked search results."""
    pdf_bytes = generate_rerank_test_pdf_bytes(
        title="Company Handbook",
        text_content="All employees receive 20 days paid leave annually.",
    )
    client.post("/api/v1/documents/index", files={"file": ("handbook.pdf", pdf_bytes, "application/pdf")})

    payload = {"query": "paid leave allowance", "top_k": 2, "candidate_k": 5}
    response = client.post("/api/v1/search/reranked", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["query"] == payload["query"]
    assert data["candidate_count"] > 0
    assert data["reranked_count"] <= payload["top_k"]
    assert data["reranker_enabled"] is True
    assert data["reranker_status"] == "success"
    assert len(data["results"]) > 0
    assert data["results"][0]["rank"] == 1
    assert "rerank_score" in data["results"][0]


def test_reranked_search_endpoint_empty_query():
    """Test POST /api/v1/search/reranked returns HTTP 400 for empty query."""
    response = client.post("/api/v1/search/reranked", json={"query": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"].lower()
