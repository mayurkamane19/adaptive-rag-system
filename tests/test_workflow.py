"""Unit test suite for LangGraph Adaptive RAG state graph execution."""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.graph.workflow import build_adaptive_rag_graph, select_retrieval_branch
from app.graph.state import RAGGraphState
from app.core.config import settings


def test_graph_compilation():
    """Test LangGraph adaptive state graph compiles successfully."""
    graph = build_adaptive_rag_graph()
    assert graph is not None


def test_select_retrieval_branch_helper():
    """Test select_retrieval_branch conditional router function."""
    state_simple: RAGGraphState = {"query": "test", "route": "simple"}
    assert select_retrieval_branch(state_simple) == "basic_retrieval"

    state_complex: RAGGraphState = {"query": "test", "route": "complex"}
    assert select_retrieval_branch(state_complex) == "enhanced_retrieval"


def test_workflow_simple_query_execution(monkeypatch):
    """Test full LangGraph execution for a simple query routes through basic_retrieval and assigns candidate_k."""
    mock_retrieval = MagicMock()
    doc = Document(
        page_content="Standard password length is 12 characters.",
        metadata={"original_filename": "passwords.pdf", "page": 1, "chunk_id": "pass_01"},
    )
    mock_retrieval.similarity_search.return_value = [(doc, 0.10)]

    mock_llm = MagicMock()
    mock_llm.generate.return_value = {"answer": "Standard password length is 12 characters.", "status": "success", "error": None}

    monkeypatch.setattr("app.graph.workflow.retrieval_service", mock_retrieval)
    monkeypatch.setattr("app.graph.workflow.llm_service", mock_llm)

    graph = build_adaptive_rag_graph()
    initial_state: RAGGraphState = {"query": "What is the password policy?"}

    final_state = graph.invoke(initial_state)

    expected_candidate_k = settings.RERANKER_CANDIDATE_K_SIMPLE if settings.RERANKER_ENABLED else settings.SIMPLE_TOP_K
    expected_top_k = settings.RERANKER_TOP_K if settings.RERANKER_ENABLED else settings.SIMPLE_TOP_K

    assert final_state["route"] == "simple"
    assert final_state["top_k"] == expected_top_k
    assert final_state["status"] == "success"
    assert "[Source 1]" in final_state["context"]
    mock_retrieval.similarity_search.assert_called_once_with(
        query="What is the password policy?", top_k=expected_candidate_k
    )


def test_workflow_complex_query_execution(monkeypatch):
    """Test full LangGraph execution for a complex query routes through enhanced_retrieval and assigns candidate_k."""
    mock_retrieval = MagicMock()
    doc = Document(
        page_content="Comparing remote VPN access with branch office intranet connections.",
        metadata={"original_filename": "network.pdf", "page": 2, "chunk_id": "net_02"},
    )
    mock_retrieval.similarity_search.return_value = [(doc, 0.15)]

    mock_llm = MagicMock()
    mock_llm.generate.return_value = {"answer": "VPN access comparison answer.", "status": "success", "error": None}

    monkeypatch.setattr("app.graph.workflow.retrieval_service", mock_retrieval)
    monkeypatch.setattr("app.graph.workflow.llm_service", mock_llm)

    graph = build_adaptive_rag_graph()
    initial_state: RAGGraphState = {"query": "Compare remote VPN access vs branch office intranet connections."}

    final_state = graph.invoke(initial_state)

    expected_candidate_k = settings.RERANKER_CANDIDATE_K_COMPLEX if settings.RERANKER_ENABLED else settings.COMPLEX_TOP_K
    expected_top_k = getattr(settings, "RERANKER_TOP_K_COMPLEX", settings.RERANKER_TOP_K) if settings.RERANKER_ENABLED else settings.COMPLEX_TOP_K

    assert final_state["route"] == "complex"
    assert final_state["top_k"] == expected_top_k
    assert final_state["status"] == "success"
    assert "[Source 1]" in final_state["context"]
    mock_retrieval.similarity_search.assert_called_once_with(
        query="Compare remote VPN access vs branch office intranet connections.", top_k=expected_candidate_k
    )
