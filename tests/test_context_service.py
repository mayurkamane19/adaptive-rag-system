"""Unit test suite for ContextBuilder and RAG context construction logic."""

import pytest
from langchain_core.documents import Document

from app.services.context_service import ContextBuilder


@pytest.fixture
def sample_retrieved_documents() -> list[Document]:
    """Fixture providing sample retrieved Document objects."""
    return [
        Document(
            page_content="All remote employee connections must pass through corporate VPN tunnels.",
            metadata={
                "source": "data/documents/security_policy.pdf",
                "page": 1,
                "chunk_id": "security_policy_p1_001",
                "original_filename": "security_policy.pdf",
            },
        ),
        Document(
            page_content="Employees must use multi-factor authentication when accessing internal portals.",
            metadata={
                "source": "data/documents/security_policy.pdf",
                "page": 2,
                "chunk_id": "security_policy_p2_002",
                "original_filename": "security_policy.pdf",
            },
        ),
    ]


def test_build_context_valid_documents(sample_retrieved_documents):
    """Test ContextBuilder builds structured deterministic context and source references."""
    builder = ContextBuilder(max_context_chars=5000)
    result = builder.build_context(sample_retrieved_documents)

    assert result["retrieved_count"] == 2
    assert result["included_count"] == 2
    assert len(result["sources"]) == 2

    context = result["context"]
    assert "[Source 1]" in context
    assert "Document: security_policy.pdf" in context
    assert "Page: 1" in context
    assert "Chunk ID: security_policy_p1_001" in context
    assert "corporate VPN tunnels" in context
    assert "[Source 2]" in context
    assert "multi-factor authentication" in context


def test_build_context_empty_documents():
    """Test ContextBuilder handles empty document list gracefully."""
    builder = ContextBuilder()
    result = builder.build_context([])

    assert result["context"] == ""
    assert result["sources"] == []
    assert result["retrieved_count"] == 0
    assert result["included_count"] == 0


def test_build_context_filters_empty_page_content():
    """Test ContextBuilder filters out documents with empty or whitespace-only content."""
    builder = ContextBuilder()
    docs = [
        Document(page_content="Valid document content.", metadata={"original_filename": "doc1.pdf", "page": 1}),
        Document(page_content="   \n   ", metadata={"original_filename": "doc2.pdf", "page": 2}),
    ]
    result = builder.build_context(docs)

    assert result["retrieved_count"] == 1
    assert result["included_count"] == 1
    assert "Valid document content." in result["context"]


def test_build_context_size_limit(sample_retrieved_documents):
    """Test ContextBuilder respects max_context_chars limit and stops adding chunks."""
    # Set a tiny max_context_chars limit that fits only the 1st chunk
    builder = ContextBuilder(max_context_chars=180)
    result = builder.build_context(sample_retrieved_documents)

    assert result["retrieved_count"] == 2
    assert result["included_count"] == 1
    assert "[Source 1]" in result["context"]
    assert "[Source 2]" not in result["context"]


def test_source_deduplication():
    """Test build_sources deduplicates identical source metadata entries."""
    builder = ContextBuilder()
    duplicate_docs = [
        Document(page_content="Chunk A", metadata={"original_filename": "report.pdf", "page": 1, "chunk_id": "c1"}),
        Document(page_content="Chunk A duplicate", metadata={"original_filename": "report.pdf", "page": 1, "chunk_id": "c1"}),
    ]

    sources = builder.build_sources(duplicate_docs)
    assert len(sources) == 1
    assert sources[0]["chunk_id"] == "c1"
