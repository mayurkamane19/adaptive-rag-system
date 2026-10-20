"""Phase 11 Release Readiness, Stability, Grounding, and Document Integrity Test Suite."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.main import app
from app.core.config import settings
from app.api.routes import documents, chat
from app.graph import workflow
from app.services.llm_service import LLMServiceError

client = TestClient(app)


def generate_pdf_bytes(title: str, body: str) -> bytes:
    """Generates a valid in-memory PDF file as bytes."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 750, title)
    c.drawString(100, 700, body)
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture(autouse=True)
def isolate_test_environment(tmp_path, monkeypatch):
    """Ensures each test operates on an isolated vector store and safe mocks."""
    temp_db_path = tmp_path / "phase11_test_vectorstore"
    temp_conv_path = tmp_path / "phase11_test_conversations"
    temp_conv_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(documents.rag_service.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(documents.rag_service.retrieval_service, "vector_store", None)

    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(chat.rag_service.retrieval_service, "vector_store", None)

    monkeypatch.setattr(workflow.retrieval_service, "vector_db_path", temp_db_path)
    monkeypatch.setattr(workflow.retrieval_service, "vector_store", None)

    monkeypatch.setattr(chat.rag_service.conversation_service, "storage_path", temp_conv_path)

    mock_llm = MagicMock()
    mock_llm.generate_answer.return_value = "Verified release answer [Source 1]."
    monkeypatch.setattr(workflow, "llm_service", mock_llm)
    monkeypatch.setattr(chat.rag_service, "llm_service", mock_llm)


# -----------------------------------------------------------------------------
# 1. Document & Index Integrity
# -----------------------------------------------------------------------------

def test_duplicate_document_indexing_detection():
    """Verify that attempting to re-index the same document name returns duplicate status and preserves the index."""
    pdf_bytes = generate_pdf_bytes("Doc A", "Content for document A.")
    file_tuple = ("doc_a.pdf", pdf_bytes, "application/pdf")

    # Initial index
    res1 = client.post("/api/v1/documents/index", files={"file": file_tuple})
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] in ("indexed", "success")
    initial_chunks = data1["chunk_count"]
    assert initial_chunks >= 1

    # Attempt to index exact same document again
    res2 = client.post("/api/v1/documents/index", files={"file": file_tuple})
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "duplicate"
    assert "already been indexed" in data2["message"].lower()


def test_unsupported_file_extension_rejected():
    """Verify that uploading non-PDF extensions (.txt, .docx) is rejected with HTTP 400."""
    res_txt = client.post(
        "/api/v1/documents/index",
        files={"file": ("notes.txt", b"Plain text notes", "text/plain")},
    )
    assert res_txt.status_code == 400
    assert "unsupported file format" in res_txt.json()["detail"].lower()

    res_docx = client.post(
        "/api/v1/documents/upload",
        files={"file": ("memo.docx", b"PK\x03\x04...", "application/vnd.openxmlformats-officedocument")},
    )
    assert res_docx.status_code == 400
    assert "unsupported file format" in res_docx.json()["detail"].lower()


def test_empty_file_upload_rejected():
    """Verify that zero-byte uploads are rejected with HTTP 400."""
    res = client.post("/api/v1/documents/upload", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


# -----------------------------------------------------------------------------
# 2. API Input Validation & Error Handling
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/v1/search",
        "/api/v1/search/reranked",
        "/api/v1/rag/context",
        "/api/v1/rag/adaptive-context",
        "/api/v1/rag/answer",
        "/api/v1/rag/chat",
    ],
)
def test_empty_query_rejected_across_all_endpoints(endpoint):
    """Verify that all retrieval and generation endpoints reject empty/whitespace queries with HTTP 400."""
    res = client.post(endpoint, json={"query": "   "})
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower() or "cannot be empty" in res.json()["detail"].lower()


def test_conversational_chat_session_deletion():
    """Verify clearing a conversation session via DELETE /api/v1/rag/chat/{session_id}."""
    session_id = "test-session-delete-me"
    chat.rag_service.conversation_service.add_message(session_id, "user", "Hello there")
    assert chat.rag_service.conversation_service.session_exists(session_id) is True

    res = client.delete(f"/api/v1/rag/chat/{session_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == session_id
    assert data["cleared"] is True
    assert "cleared" in data["message"].lower()
    assert chat.rag_service.conversation_service.session_exists(session_id) is False


# -----------------------------------------------------------------------------
# 3. Grounding & Insufficient Information Handling
# -----------------------------------------------------------------------------

def test_unanswerable_question_fallback_behavior(monkeypatch):
    """Verify that when context does not contain sufficient facts, the grounding rule produces the standard fallback."""
    pdf_bytes = generate_pdf_bytes("Staff Roster", "Alice is Director of Engineering.")
    client.post("/api/v1/documents/index", files={"file": ("staff.pdf", pdf_bytes, "application/pdf")})

    fallback_phrase = "I couldn't find enough information in the provided documents to answer this question."
    mock_llm = MagicMock()
    mock_llm.generate_answer.return_value = fallback_phrase
    monkeypatch.setattr(workflow, "llm_service", mock_llm)
    monkeypatch.setattr(chat.rag_service, "llm_service", mock_llm)

    res = client.post("/api/v1/rag/answer", json={"query": "What is the annual operating budget of NASA?"})
    assert res.status_code == 200
    data = res.json()
    assert "couldn't find enough information" in data["answer"].lower()
    assert data["status"] == "success"
    assert len(data["sources"]) >= 1


def test_rag_generation_failure_does_not_mask_as_success(monkeypatch):
    """Verify that when Ollama is unreachable, generation_status is 'failed' and error is actionable."""
    failing_llm = MagicMock()
    failing_llm.generate_answer.side_effect = LLMServiceError(
        "Could not connect to Ollama server at 'http://localhost:11434'. Please ensure Ollama is installed and running."
    )
    monkeypatch.setattr(workflow, "llm_service", failing_llm)
    monkeypatch.setattr(chat.rag_service, "llm_service", failing_llm)

    pdf_bytes = generate_pdf_bytes("Security", "Remote work requires VPN.")
    client.post("/api/v1/documents/index", files={"file": ("vpn.pdf", pdf_bytes, "application/pdf")})

    res = client.post("/api/v1/rag/answer", json={"query": "Is VPN required?"})
    assert res.status_code == 200
    data = res.json()
    assert data["generation_status"] == "failed"
    assert "LLM generation service is currently unavailable" in data["answer"]
    assert "Could not connect to Ollama" in data["generation_error"]
    assert len(data["sources"]) >= 1


# -----------------------------------------------------------------------------
# 4. Health & Liveness Probes
# -----------------------------------------------------------------------------

def test_health_endpoints_structure():
    """Verify /health, /health/live, and /health/ready provide complete and accurate contracts."""
    res_live = client.get("/health/live")
    assert res_live.status_code == 200
    assert res_live.json()["status"] == "alive"

    res_ready = client.get("/health/ready")
    assert res_ready.status_code == 200
    data_ready = res_ready.json()
    assert data_ready["status"] in ("ready", "degraded")
    assert "components" in data_ready
    assert "vector_store" in data_ready["components"]
    assert "reranker" in data_ready["components"]
    assert "llm_service" in data_ready["components"]
