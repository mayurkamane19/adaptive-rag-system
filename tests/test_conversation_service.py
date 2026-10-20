"""Unit test suite for ConversationService in-memory session management."""

import pytest
from app.services.conversation_service import ConversationService
from app.models.message import ConversationMessage


def test_conversation_service_create_and_exists(tmp_path):
    """Test creating sessions and checking session existence."""
    service = ConversationService(storage_path=tmp_path)
    session_id = service.create_session("session-test-1")
    assert session_id == "session-test-1"
    assert service.session_exists("session-test-1") is True
    assert service.session_exists("non-existent-session") is False


def test_conversation_service_add_and_get_history(tmp_path):
    """Test adding messages and retrieving session history."""
    service = ConversationService(storage_path=tmp_path)
    session_id = "session-test-2"
    service.add_message(session_id, "user", "What is the policy for leave?")
    service.add_message(session_id, "assistant", "Employees receive 20 days paid leave.")

    history = service.get_history(session_id)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "What is the policy for leave?"
    assert history[1].role == "assistant"
    assert history[1].content == "Employees receive 20 days paid leave."


def test_conversation_service_history_limit(tmp_path):
    """Test history bounding when message count exceeds limit."""
    service = ConversationService(storage_path=tmp_path)
    session_id = "session-test-limit"
    for i in range(15):
        service.add_message(session_id, "user", f"Question {i}")
        service.add_message(session_id, "assistant", f"Answer {i}")

    history = service.get_history(session_id, limit=6)
    assert len(history) == 6
    assert history[0].content == "Question 12"
    assert history[-1].content == "Answer 14"


def test_conversation_service_clear_session(tmp_path):
    """Test clearing conversation session history."""
    service = ConversationService(storage_path=tmp_path)
    session_id = "session-test-clear"
    service.add_message(session_id, "user", "Hello")
    assert service.session_exists(session_id) is True

    cleared = service.clear_session(session_id)
    assert cleared is True
    assert service.session_exists(session_id) is False
    assert len(service.get_history(session_id)) == 0


def test_conversation_service_empty_message_error(tmp_path):
    """Test ValueError raised when trying to add empty message."""
    service = ConversationService(storage_path=tmp_path)
    with pytest.raises(ValueError):
        service.add_message("session-test-err", "user", "   ")
