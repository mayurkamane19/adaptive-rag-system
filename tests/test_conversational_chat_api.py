"""Integration test suite for multi-turn conversational RAG REST API endpoints."""

import io
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.main import app
from app.api.routes import documents, chat
from app.graph import workflow

client = TestClient(app)


def generate_chat_test_pdf_bytes(title: str, text_content: str) -> bytes:
    """Generates an in-memory PDF file for conversational chat API tests."""
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
    temp_db_path = tmp_path / "conversational_api_test_vectorstore"

    monkeypatch.setattr(documents.rag_service.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(documents.rag_service.retrieval_service, "vector_store", None)

    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_store", None)

    monkeypatch.setattr(workflow.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(workflow.retrieval_service, "vector_store", None)

    mock_llm = MagicMock()
    mock_llm.generate_answer.return_value = "Employees receive 20 days of paid annual leave."
    monkeypatch.setattr(workflow, "llm_service", mock_llm)
    monkeypatch.setattr(workflow.query_rewriter, "llm_service", mock_llm)
    monkeypatch.setattr(chat.rag_service, "llm_service", mock_llm)

    temp_conv_path = tmp_path / "conversational_api_test_conversations"
    monkeypatch.setattr(workflow.conversation_service, "storage_path", temp_conv_path)
    monkeypatch.setattr(workflow.conversation_service, "_sessions", {})
    monkeypatch.setattr(chat.rag_service.conversation_service, "storage_path", temp_conv_path)
    monkeypatch.setattr(chat.rag_service.conversation_service, "_sessions", {})


def test_conversational_chat_endpoint_first_turn():
    """Test POST /api/v1/rag/chat for first turn in a new session."""
    pdf_bytes = generate_chat_test_pdf_bytes(
        title="Leave Policy 2026",
        text_content="Employees receive 20 days of paid annual leave.",
    )
    client.post("/api/v1/documents/index", files={"file": ("leave_policy.pdf", pdf_bytes, "application/pdf")})

    session_id = "test-session-conv-1"
    payload = {"query": "What is the annual leave policy?", "session_id": session_id}
    response = client.post("/api/v1/rag/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["query"] == payload["query"]
    assert data["was_rewritten"] is False
    assert "20 days of paid annual leave" in data["answer"]
    assert len(data["sources"]) >= 1


def test_conversational_chat_history_retrieval_and_clear():
    """Test GET /api/v1/rag/chat/{session_id}/history and DELETE /api/v1/rag/chat/{session_id}."""
    session_id = "test-session-history-1"

    # Send a turn to populate history
    payload = {"query": "What is the paid leave allowance?", "session_id": session_id}
    client.post("/api/v1/rag/chat", json=payload)

    # Retrieve history
    hist_response = client.get(f"/api/v1/rag/chat/{session_id}/history")
    assert hist_response.status_code == 200
    hist_data = hist_response.json()
    assert hist_data["session_id"] == session_id
    assert hist_data["message_count"] == 2
    assert hist_data["messages"][0]["role"] == "user"
    assert hist_data["messages"][1]["role"] == "assistant"

    # Clear history
    del_response = client.delete(f"/api/v1/rag/chat/{session_id}")
    assert del_response.status_code == 200
    del_data = del_response.json()
    assert del_data["cleared"] is True

    # Confirm history is now empty
    hist_after = client.get(f"/api/v1/rag/chat/{session_id}/history")
    assert hist_after.status_code == 200
    assert hist_after.json()["message_count"] == 0


def test_conversational_chat_empty_query():
    """Test POST /api/v1/rag/chat returns HTTP 400 for empty query."""
    response = client.post("/api/v1/rag/chat", json={"query": "   ", "session_id": "s1"})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"].lower()


def test_conversational_chat_llm_failure_preserves_sources_and_skips_history_poisoning(monkeypatch):
    """Test that LLM failure preserves sources and does not save error turns to conversation history."""
    from app.services.llm_service import LLMServiceError

    pdf_bytes = generate_chat_test_pdf_bytes(
        title="Leave Policy 2026",
        text_content="Employees receive 20 days of paid annual leave.",
    )
    client.post("/api/v1/documents/index", files={"file": ("leave_policy.pdf", pdf_bytes, "application/pdf")})

    failing_llm = MagicMock()
    failing_llm.generate_answer.side_effect = LLMServiceError(
        "Could not connect to Ollama server at 'http://localhost:11434'."
    )
    monkeypatch.setattr(workflow, "llm_service", failing_llm)
    monkeypatch.setattr(chat.rag_service, "llm_service", failing_llm)

    session_id = "test-fail-session"
    payload = {"query": "What is the annual leave allowance?", "session_id": session_id}
    response = client.post("/api/v1/rag/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["generation_status"] == "failed"
    assert "LLM generation service is currently unavailable" in data["answer"]
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["original_filename"] == "leave_policy.pdf"

    # Verify conversation history was NOT poisoned with the error message
    hist_response = client.get(f"/api/v1/rag/chat/{session_id}/history")
    assert hist_response.status_code == 200
    assert hist_response.json()["message_count"] == 0
