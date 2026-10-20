"""Unit test suite for document upload FastAPI API endpoint."""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.main import app

client = TestClient(app)


def generate_test_pdf_bytes(title: str = "Test PDF Document") -> bytes:
    """Generates a valid in-memory PDF file as bytes for API testing."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(100, 750, title)
    c.drawString(100, 700, "This is paragraph content inside the test PDF file.")
    c.showPage()
    c.save()
    return buffer.getvalue()


def test_upload_pdf_success():
    """Test POST /api/v1/documents/upload with a valid PDF file."""
    pdf_bytes = generate_test_pdf_bytes("API Test Document")
    files = {"file": ("api_test_doc.pdf", pdf_bytes, "application/pdf")}

    response = client.post("/api/v1/documents/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "api_test_doc.pdf"
    assert data["page_count"] == 1
    assert data["chunk_count"] >= 1
    assert data["file_size_bytes"] == len(pdf_bytes)
    assert len(data["chunks_summary"]) == data["chunk_count"]

    # Cleanup saved test file from data/documents/ if present
    saved_path = Path(data["file_path"])
    if saved_path.exists():
        saved_path.unlink()


def test_upload_invalid_extension():
    """Test POST /api/v1/documents/upload with an unsupported file extension returns HTTP 400."""
    files = {"file": ("report.txt", b"Plain text content", "text/plain")}

    response = client.post("/api/v1/documents/upload", files=files)

    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file format" in data["detail"] or "Only .pdf files are supported" in data["detail"]


def test_upload_empty_file():
    """Test POST /api/v1/documents/upload with an empty 0-byte file returns HTTP 400."""
    files = {"file": ("empty_test.pdf", b"", "application/pdf")}

    response = client.post("/api/v1/documents/upload", files=files)

    assert response.status_code == 400
    data = response.json()
    assert "empty" in data["detail"].lower()


def test_upload_blank_pdf_yields_warning(tmp_path):
    """Regression test: uploading a blank PDF with 0 extractable characters returns warning status and 0 chunks."""
    import pypdf

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=595, height=842)
    blank_bytes = io.BytesIO()
    writer.write(blank_bytes)
    blank_data = blank_bytes.getvalue()

    files = {"file": ("blank_regression_doc.pdf", blank_data, "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "warning"
    assert data["page_count"] == 1
    assert data["chunk_count"] == 0
    assert len(data["chunks_summary"]) == 0


def test_index_blank_pdf_yields_warning():
    """Regression test: indexing a blank PDF returns warning status and 0 chunk count."""
    import pypdf

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=595, height=842)
    blank_bytes = io.BytesIO()
    writer.write(blank_bytes)
    blank_data = blank_bytes.getvalue()

    files = {"file": ("blank_index_doc.pdf", blank_data, "application/pdf")}
    response = client.post("/api/v1/documents/index", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "warning"
    assert data["chunk_count"] == 0
    assert "No document chunks provided for indexing" in data["message"]


def test_index_duplicate_returns_duplicate_status_and_zero_chunks():
    """Regression test: duplicate upload returns duplicate status and accurately reports 0 chunks indexed."""
    pdf_bytes = generate_test_pdf_bytes("Duplicate Test Document")
    unique_name = "test_duplicate_isolation_doc.pdf"
    files1 = {"file": (unique_name, pdf_bytes, "application/pdf")}

    # First index: should succeed
    res1 = client.post("/api/v1/documents/index", files=files1)
    assert res1.status_code == 200
    data1 = res1.json()
    # It might be indexed or already duplicate if run multiple times
    if data1["status"] in ("indexed", "success"):
        assert data1["chunk_count"] >= 1

    # Second index with exact same filename: must return duplicate and 0 chunks
    files2 = {"file": (unique_name, pdf_bytes, "application/pdf")}
    res2 = client.post("/api/v1/documents/index", files=files2)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "duplicate"
    assert data2["chunk_count"] == 0
    assert "already been indexed" in data2["message"]

