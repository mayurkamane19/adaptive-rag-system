"""Unit test suite for DocumentIngestionService, PDF extraction, text cleaning, and chunking."""

import os
import pytest
from pathlib import Path
from reportlab.pdfgen import canvas
from langchain_core.documents import Document

from app.services.ingestion_service import (
    DocumentIngestionService,
    InvalidFileError,
    DocumentProcessingError,
)


@pytest.fixture
def temp_pdf_file(tmp_path) -> Path:
    """Fixture to generate a temporary valid 2-page PDF file using ReportLab."""
    pdf_path = tmp_path / "sample_test_doc.pdf"
    c = canvas.Canvas(str(pdf_path))
    
    # Page 1
    c.drawString(100, 750, "Sample Document Title - Page 1")
    c.drawString(100, 700, "This is the first paragraph of test text for document ingestion testing.")
    c.showPage()
    
    # Page 2
    c.drawString(100, 750, "Sample Document Title - Page 2")
    c.drawString(100, 700, "This is the second page of test text. Section two contains additional details.")
    c.save()
    
    return pdf_path


@pytest.fixture
def empty_file(tmp_path) -> Path:
    """Fixture to generate a 0-byte file."""
    empty_path = tmp_path / "empty.pdf"
    empty_path.touch()
    return empty_path


@pytest.fixture
def text_file(tmp_path) -> Path:
    """Fixture to generate a non-PDF text file."""
    txt_path = tmp_path / "document.txt"
    txt_path.write_text("This is a plain text file, not a PDF.")
    return txt_path


def test_file_validation_valid(temp_pdf_file):
    """Test validation succeeds for valid existing non-empty PDF file."""
    service = DocumentIngestionService()
    assert service.validate_file(temp_pdf_file) is True


def test_file_validation_invalid_extension(text_file):
    """Test validation raises InvalidFileError for non-PDF extensions."""
    service = DocumentIngestionService()
    with pytest.raises(InvalidFileError, match="Only .pdf files are supported"):
        service.validate_file(text_file)


def test_file_validation_empty_file(empty_file):
    """Test validation raises InvalidFileError for 0-byte files."""
    service = DocumentIngestionService()
    with pytest.raises(InvalidFileError, match="empty \\(0 bytes\\)"):
        service.validate_file(empty_file)


def test_file_validation_non_existent(tmp_path):
    """Test validation raises InvalidFileError for non-existent file path."""
    service = DocumentIngestionService()
    fake_path = tmp_path / "non_existent.pdf"
    with pytest.raises(InvalidFileError, match="does not exist on disk"):
        service.validate_file(fake_path)


def test_text_cleaning_whitespace():
    """Test text cleaning normalizes excessive inline whitespace and paragraph linebreaks."""
    service = DocumentIngestionService()

    raw_text = "This   is   a document.\n\n\nThis is another line."
    cleaned = service.clean_text(raw_text)

    expected = "This is a document.\n\nThis is another line."
    assert cleaned == expected


def test_text_cleaning_normal():
    """Test text cleaning handles clean text without distorting content."""
    service = DocumentIngestionService()
    raw = "Header Section\n\nParagraph sentence one. Paragraph sentence two."
    cleaned = service.clean_text(raw)
    assert cleaned == raw


def test_chunking_metadata_and_uniqueness():
    """Test document chunking produces unique IDs, correct page numbers, and preserved metadata."""
    service = DocumentIngestionService(chunk_size=100, chunk_overlap=20)

    docs = [
        Document(
            page_content="Page 1 text content " * 10,
            metadata={"source": "test_path.pdf", "page": 0},
        ),
        Document(
            page_content="Page 2 text content " * 10,
            metadata={"source": "test_path.pdf", "page": 1},
        ),
    ]

    chunks = service.split_documents(docs, original_filename="my_test_report.pdf")

    assert len(chunks) > 0

    chunk_ids = set()
    for chunk in chunks:
        meta = chunk.metadata
        assert "chunk_id" in meta
        assert "page" in meta
        assert "source" in meta
        assert meta["original_filename"] == "my_test_report.pdf"
        assert meta["page"] in (1, 2)

        # Verify chunk IDs are unique
        assert meta["chunk_id"] not in chunk_ids
        chunk_ids.add(meta["chunk_id"])


def test_full_document_processing_pipeline(temp_pdf_file):
    """Test process_document executes end-to-end PDF loading, cleaning, and chunking."""
    service = DocumentIngestionService(chunk_size=200, chunk_overlap=30)
    result = service.process_document(temp_pdf_file, original_filename="sample_test_doc.pdf")

    assert result.status == "success"
    assert result.page_count == 2
    assert result.chunk_count >= 2
    assert len(result.chunks) == result.chunk_count
    assert result.original_filename == "sample_test_doc.pdf"
    assert result.chunks[0].chunk_id.startswith("sample_test_doc_p1_")


def test_process_document_blank_pdf_yields_warning(tmp_path):
    """Regression test: empty/blank PDF with no text and no images returns warning and zero chunks."""
    import pypdf

    blank_pdf = tmp_path / "blank_test.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with open(blank_pdf, "wb") as f:
        writer.write(f)

    service = DocumentIngestionService()
    result = service.process_document(blank_pdf, original_filename="blank_test.pdf")

    assert result.status == "warning"
    assert result.page_count == 1
    assert result.chunk_count == 0
    assert len(result.chunks) == 0
    assert "no readable text chunks" in result.message.lower()


def test_process_document_scanned_pdf_ocr():
    """Regression test: scanned/image-based PDF with embedded images triggers OCR fallback and produces chunks."""
    scanned_path = Path("data/documents/Mayur_Mohan_Kamane_Personal_Profile_Final_1_.pdf")
    if not scanned_path.is_file():
        pytest.skip("Scanned test PDF not found on disk.")

    service = DocumentIngestionService()
    result = service.process_document(scanned_path, original_filename="Mayur_Mohan_Kamane_Personal_Profile_Final_1_.pdf")

    assert result.status == "success"
    assert result.page_count == 1
    assert result.chunk_count >= 1
    assert any("Mayur Mohan Kamane" in c.chunk_id or c.content_length > 0 for c in result.chunks)


def test_ocr_utils_empty_and_missing_handling():
    """Test OCR utility handles empty bytes and missing files safely without raising exceptions."""
    from app.utils.ocr_utils import ocr_image_bytes, ocr_image_file, is_ocr_available

    assert isinstance(is_ocr_available(), bool)
    assert ocr_image_bytes(b"") == ""
    assert ocr_image_file("non_existent_file_path_12345.png") == ""

