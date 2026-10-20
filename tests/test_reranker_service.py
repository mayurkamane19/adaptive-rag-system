"""Unit test suite for RerankerService cross-encoder candidate re-ranking."""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.services.reranker_service import RerankerService


def test_reranker_service_successful_rerank():
    """Test RerankerService correctly scores and orders candidate documents."""
    service = RerankerService(model_name="test-cross-encoder-model")

    mock_model = MagicMock()
    # Return raw scores where document 2 (index 1) gets a higher score than document 1 (index 0)
    mock_model.predict.return_value = [0.15, 0.92, 0.45]
    service._model = mock_model

    candidates = [
        (Document(page_content="Content A", metadata={"original_filename": "docA.pdf", "page": 1}), 0.8),
        (Document(page_content="Content B", metadata={"original_filename": "docB.pdf", "page": 2}), 0.5),
        (Document(page_content="Content C", metadata={"original_filename": "docC.pdf", "page": 3}), 0.3),
    ]

    res = service.rerank(query="test query", documents=candidates, top_k=2)

    assert res["status"] == "success"
    assert res["candidate_count"] == 3
    assert res["reranked_count"] == 2
    assert res["reranker_enabled"] is True

    docs = res["reranked_documents"]
    # Document B should be ranked #1 because its rerank_score is 0.92
    assert docs[0]["content"] == "Content B"
    assert docs[0]["rank"] == 1
    assert docs[0]["rerank_score"] == 0.92
    assert docs[0]["retrieval_score"] == 0.5

    # Document C should be ranked #2 because its rerank_score is 0.45
    assert docs[1]["content"] == "Content C"
    assert docs[1]["rank"] == 2
    assert docs[1]["rerank_score"] == 0.45
    assert docs[1]["retrieval_score"] == 0.3


def test_reranker_service_empty_candidates():
    """Test RerankerService handles empty candidate list gracefully."""
    service = RerankerService()
    res = service.rerank(query="test query", documents=[], top_k=4)

    assert res["status"] == "empty"
    assert res["candidate_count"] == 0
    assert res["reranked_count"] == 0
    assert res["reranked_documents"] == []


def test_reranker_service_fallback_on_model_error():
    """Test RerankerService falls back to FAISS ranking when CrossEncoder fails."""
    service = RerankerService()

    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("Model inference error")
    service._model = mock_model

    candidates = [
        (Document(page_content="First Doc", metadata={"original_filename": "f.pdf", "page": 1}), 0.1),
        (Document(page_content="Second Doc", metadata={"original_filename": "s.pdf", "page": 2}), 0.2),
    ]

    res = service.rerank(query="test query", documents=candidates, top_k=2)

    assert res["status"] == "fallback"
    assert res["candidate_count"] == 2
    assert res["reranker_enabled"] is False
    assert res["error"] is not None

    docs = res["reranked_documents"]
    assert len(docs) == 2
    assert docs[0]["content"] == "First Doc"
    assert docs[0]["rank"] == 1
    assert docs[0]["rerank_score"] is None
