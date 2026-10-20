"""Unit test suite for LangGraph generation nodes and end-to-end adaptive execution."""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.graph.workflow import build_adaptive_rag_graph
from app.graph.state import RAGGraphState


def test_generation_workflow_empty_context_skips_llm(monkeypatch):
    """Test workflow skips LLM generation and sets grounded fallback answer when no context is retrieved."""
    mock_retrieval = MagicMock()
    mock_retrieval.similarity_search.return_value = []
    monkeypatch.setattr("app.graph.workflow.retrieval_service", mock_retrieval)

    mock_llm = MagicMock()
    monkeypatch.setattr("app.graph.workflow.llm_service", mock_llm)

    graph = build_adaptive_rag_graph()
    initial_state: RAGGraphState = {"query": "Non existent query topics"}

    final_state = graph.invoke(initial_state)

    assert final_state["generation_status"] == "skipped_empty_context"
    assert "couldn't find enough information" in final_state["answer"]
    # Verify LLM was NOT called
    mock_llm.generate_answer.assert_not_called()


def test_generation_workflow_successful_llm_execution(monkeypatch):
    """Test workflow executes prompt node and LLM answer node when valid context is retrieved."""
    mock_retrieval = MagicMock()
    doc = Document(
        page_content="Remote work VPN connections require multi-factor authentication.",
        metadata={"original_filename": "security.pdf", "page": 1, "chunk_id": "sec_01"},
    )
    mock_retrieval.similarity_search.return_value = [(doc, 0.15)]
    monkeypatch.setattr("app.graph.workflow.retrieval_service", mock_retrieval)

    mock_llm = MagicMock()
    mock_llm.generate_answer.return_value = "Remote VPN connections require multi-factor authentication [Source 1]."
    monkeypatch.setattr("app.graph.workflow.llm_service", mock_llm)

    graph = build_adaptive_rag_graph()
    initial_state: RAGGraphState = {"query": "What are the remote work security rules?"}

    final_state = graph.invoke(initial_state)

    assert final_state["generation_status"] == "success"
    assert final_state["answer"] == "Remote VPN connections require multi-factor authentication [Source 1]."
    assert final_state["prompt"] != ""
    mock_llm.generate_answer.assert_called_once()
