"""Unit test suite for EmbeddingService."""

import pytest
from app.services.embedding_service import EmbeddingService


@pytest.fixture(scope="module")
def embedding_service() -> EmbeddingService:
    """Fixture providing a module-scoped EmbeddingService instance."""
    return EmbeddingService()


def test_embedding_service_initialization(embedding_service):
    """Test EmbeddingService initializes successfully with configured model."""
    assert embedding_service is not None
    assert embedding_service.get_embeddings() is not None


def test_embed_documents(embedding_service):
    """Test embed_documents generates correct vector dimensions for text chunks."""
    texts = [
        "First document text chunk for embedding generation test.",
        "Second document chunk with additional context.",
    ]
    vectors = embedding_service.embed_documents(texts)

    assert len(vectors) == 2
    assert isinstance(vectors[0], list)
    # all-MiniLM-L6-v2 vector dimension is 384
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384


def test_embed_query(embedding_service):
    """Test embed_query generates a valid single embedding vector."""
    query = "What is the document about?"
    vector = embedding_service.embed_query(query)

    assert isinstance(vector, list)
    assert len(vector) == 384


def test_embed_empty_query_raises_error(embedding_service):
    """Test embed_query raises ValueError for empty string inputs."""
    with pytest.raises(ValueError, match="cannot be empty"):
        embedding_service.embed_query("   ")
