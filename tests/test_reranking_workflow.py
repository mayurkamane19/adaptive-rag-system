"""Unit test suite for LangGraph re-ranking workflow nodes and state transitions."""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.graph import workflow
from app.graph.state import RAGGraphState
from app.core.config import settings


@pytest.fixture(autouse=True)
def mock_retrieval_and_reranker(monkeypatch):
    """Fixture mocking retrieval and re-ranking services for workflow testing."""
    mock_ret = MagicMock()
    mock_ret.similarity_search.return_value = [
        (Document(page_content="Policy doc 1", metadata={"original_filename": "policy.pdf", "page": 1, "chunk_id": "c1"}), 0.6),
        (Document(page_content="Policy doc 2", metadata={"original_filename": "policy.pdf", "page": 2, "chunk_id": "c2"}), 0.4),
        (Document(page_content="Policy doc 3", metadata={"original_filename": "policy.pdf", "page": 3, "chunk_id": "c3"}), 0.2),
    ]
    monkeypatch.setattr(workflow, "retrieval_service", mock_ret)

    mock_rerank = MagicMock()
    mock_rerank.rerank.return_value = {
        "status": "success",
        "candidate_count": 3,
        "reranked_count": 2,
        "reranker_enabled": True,
        "reranker_model": "test-model",
        "error": None,
        "reranked_documents": [
            {
                "document": Document(page_content="Policy doc 2", metadata={"original_filename": "policy.pdf", "page": 2, "chunk_id": "c2", "rank": 1}),
                "content": "Policy doc 2",
                "metadata": {"original_filename": "policy.pdf", "page": 2, "chunk_id": "c2", "rank": 1},
                "retrieval_score": 0.4,
                "rerank_score": 0.95,
                "rank": 1,
            },
            {
                "document": Document(page_content="Policy doc 1", metadata={"original_filename": "policy.pdf", "page": 1, "chunk_id": "c1", "rank": 2}),
                "content": "Policy doc 1",
                "metadata": {"original_filename": "policy.pdf", "page": 1, "chunk_id": "c1", "rank": 2},
                "retrieval_score": 0.6,
                "rerank_score": 0.81,
                "rank": 2,
            },
        ],
    }
    monkeypatch.setattr(workflow, "reranker_service", mock_rerank)


def test_route_query_node_candidate_k_allocation():
    """Test route_query_node sets candidate_k and top_k based on query route."""
    state_simple: RAGGraphState = {"query": "What is annual leave?"}
    res_simple = workflow.route_query_node(state_simple)
    assert res_simple["candidate_k"] == settings.RERANKER_CANDIDATE_K_SIMPLE
    assert res_simple["top_k"] == settings.RERANKER_TOP_K

    state_complex: RAGGraphState = {"query": "Compare remote work security rules versus office network attendance requirements."}
    res_complex = workflow.route_query_node(state_complex)
    assert res_complex["candidate_k"] == settings.RERANKER_CANDIDATE_K_COMPLEX
    assert res_complex["top_k"] == getattr(settings, "RERANKER_TOP_K_COMPLEX", settings.RERANKER_TOP_K)


def test_rerank_documents_node_success():
    """Test rerank_documents_node processes candidate documents and updates state."""
    state: RAGGraphState = {
        "query": "What is annual leave?",
        "standalone_query": "What is annual leave?",
        "candidate_documents": [
            (Document(page_content="Policy doc 1", metadata={"original_filename": "policy.pdf"}), 0.6),
            (Document(page_content="Policy doc 2", metadata={"original_filename": "policy.pdf"}), 0.4),
        ],
        "top_k": 2,
    }

    res = workflow.rerank_documents_node(state)

    assert res["reranker_status"] == "success"
    assert res["candidate_count"] == 3
    assert res["reranked_count"] == 2
    assert len(res["retrieved_documents"]) == 2
    assert res["retrieved_documents"][0][0].page_content == "Policy doc 2"
