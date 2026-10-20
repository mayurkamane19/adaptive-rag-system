"""Unit test suite for RetrievalService and FAISS vector store operations."""

import pytest
from pathlib import Path
from langchain_core.documents import Document

from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService


@pytest.fixture(scope="module")
def shared_embedding_service() -> EmbeddingService:
    """Fixture providing a shared embedding service for tests."""
    return EmbeddingService()


@pytest.fixture
def temp_retrieval_service(tmp_path, shared_embedding_service) -> RetrievalService:
    """Fixture providing a RetrievalService pointing to an isolated temp directory."""
    db_path = tmp_path / "test_vectorstore"
    return RetrievalService(vector_db_path=db_path, embedding_service=shared_embedding_service)


@pytest.fixture
def sample_documents() -> list[Document]:
    """Fixture providing sample Document objects with metadata."""
    return [
        Document(
            page_content="The annual profit of the company grew by 25 percent in 2025.",
            metadata={
                "source": "financial_report_2025.pdf",
                "page": 1,
                "chunk_id": "fin2025_p1_001",
                "original_filename": "financial_report_2025.pdf",
            },
        ),
        Document(
            page_content="Employee leave policy requires two weeks notice for vacation requests.",
            metadata={
                "source": "hr_policy.pdf",
                "page": 3,
                "chunk_id": "hr_p3_001",
                "original_filename": "hr_policy.pdf",
            },
        ),
    ]


def test_add_documents_and_persistence(temp_retrieval_service, sample_documents):
    """Test adding documents creates FAISS index and persists it to disk."""
    service = temp_retrieval_service
    doc = sample_documents[0]

    result = service.add_documents([doc], original_filename=doc.metadata["original_filename"])

    assert result["status"] == "indexed"
    assert result["added_chunks"] == 1
    assert service.vector_store_exists() is True

    # Re-initialize service from same path to test loading persistence
    reloaded_service = RetrievalService(
        vector_db_path=service.vector_db_path,
        embedding_service=service.embedding_service,
    )
    assert reloaded_service.vector_store is not None


def test_duplicate_document_handling(temp_retrieval_service, sample_documents):
    """Test adding the same document twice returns duplicate status and skips insertion."""
    service = temp_retrieval_service
    doc = sample_documents[0]
    filename = doc.metadata["original_filename"]

    # First insertion
    res1 = service.add_documents([doc], original_filename=filename)
    assert res1["status"] == "indexed"

    # Second insertion (duplicate)
    res2 = service.add_documents([doc], original_filename=filename)
    assert res2["status"] == "duplicate"
    assert res2["added_chunks"] == 0


def test_similarity_search(temp_retrieval_service, sample_documents):
    """Test similarity search returns relevant document chunks matching query."""
    service = temp_retrieval_service
    service.add_documents(sample_documents, original_filename="batch_import.pdf")

    # Search query relevant to profit/financials
    results = service.similarity_search("How much did company profit grow?", top_k=1)

    assert len(results) == 1
    doc, score = results[0]
    assert "profit of the company grew" in doc.page_content
    assert doc.metadata["original_filename"] == "financial_report_2025.pdf"
    assert doc.metadata["page"] == 1


def test_similarity_search_missing_vectorstore(tmp_path, shared_embedding_service):
    """Test similarity search raises ValueError when no vector store exists on disk."""
    empty_path = tmp_path / "non_existent_vectorstore"
    service = RetrievalService(vector_db_path=empty_path, embedding_service=shared_embedding_service)

    with pytest.raises(ValueError, match="is empty or does not exist"):
        service.similarity_search("Sample query")
