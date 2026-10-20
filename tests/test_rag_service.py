"""Unit test suite for RAGService context retrieval methods."""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.services.rag_service import RAGService
from app.services.context_service import ContextBuilder


def test_retrieve_context_success():
    """Test RAGService.retrieve_context processes valid queries and returns structured context."""
    mock_retrieval_service = MagicMock()
    sample_doc = Document(
        page_content="Password policy mandates at least 12 characters including symbols.",
        metadata={
            "source": "data/documents/passwords.pdf",
            "page": 1,
            "chunk_id": "pass_p1_001",
            "original_filename": "passwords.pdf",
        },
    )
    mock_retrieval_service.similarity_search.return_value = [(sample_doc, 0.25)]

    service = RAGService(retrieval_service=mock_retrieval_service)
    result = service.retrieve_context(query="What is the password policy?", top_k=2)

    assert result["status"] == "success"
    assert result["query"] == "What is the password policy?"
    assert result["retrieved_count"] == 1
    assert result["included_count"] == 1
    assert len(result["sources"]) == 1
    assert "[Source 1]" in result["context"]
    assert "at least 12 characters" in result["context"]


def test_retrieve_context_empty_query():
    """Test RAGService.retrieve_context raises ValueError for empty query strings."""
    service = RAGService()
    with pytest.raises(ValueError, match="cannot be empty"):
        service.retrieve_context(query="   ")


def test_retrieve_context_no_results():
    """Test RAGService.retrieve_context handles no-results or missing vector store gracefully."""
    mock_retrieval_service = MagicMock()
    mock_retrieval_service.similarity_search.side_effect = ValueError("Vector store does not exist.")

    service = RAGService(retrieval_service=mock_retrieval_service)
    result = service.retrieve_context(query="Non existent topic query", top_k=4)

    assert result["status"] == "no_results"
    assert result["context"] == ""
    assert result["sources"] == []
    assert result["retrieved_count"] == 0
    assert result["included_count"] == 0
