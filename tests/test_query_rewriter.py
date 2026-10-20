"""Unit test suite for QueryRewriter service."""

import pytest
from unittest.mock import MagicMock
from app.services.query_rewriter import QueryRewriter
from app.models.message import ConversationMessage


def test_query_rewriter_no_history():
    """Test QueryRewriter returns original query when history is empty."""
    mock_llm = MagicMock()
    rewriter = QueryRewriter(llm_service=mock_llm)

    standalone, was_rewritten = rewriter.rewrite_query("Can I carry it forward?", history=[])
    assert standalone == "Can I carry it forward?"
    assert was_rewritten is False
    mock_llm.generate_answer.assert_not_called()


def test_query_rewriter_successful_rewrite():
    """Test QueryRewriter successfully reformulates follow-up using LLM response."""
    mock_llm = MagicMock()
    mock_llm.generate_answer.return_value = "Can employees carry forward unused annual leave?"
    rewriter = QueryRewriter(llm_service=mock_llm)

    history = [
        ConversationMessage(role="user", content="What is the annual leave allowance?"),
        ConversationMessage(role="assistant", content="Employees receive 20 days of paid annual leave."),
    ]

    standalone, was_rewritten = rewriter.rewrite_query("Can I carry it forward?", history=history)
    assert standalone == "Can employees carry forward unused annual leave?"
    assert was_rewritten is True
    mock_llm.generate_answer.assert_called_once()


def test_query_rewriter_llm_failure_fallback():
    """Test QueryRewriter gracefully falls back to original query when LLM raises exception."""
    mock_llm = MagicMock()
    mock_llm.generate_answer.side_effect = RuntimeError("Ollama service unreachable")
    rewriter = QueryRewriter(llm_service=mock_llm)

    history = [
        ConversationMessage(role="user", content="What is the remote work policy?"),
    ]

    standalone, was_rewritten = rewriter.rewrite_query("How many days per week?", history=history)
    assert standalone == "How many days per week?"
    assert was_rewritten is False
