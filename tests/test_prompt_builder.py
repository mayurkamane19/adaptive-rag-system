"""Unit test suite for custom RAG prompt construction."""

import pytest
from app.prompts.rag_prompts import build_rag_prompt, SYSTEM_RAG_INSTRUCTIONS


def test_build_rag_prompt_structure():
    """Test build_rag_prompt includes grounding instructions, context, and query."""
    query = "What is the security policy?"
    context = "[Source 1]\nDocument: policy.pdf\nPage: 1\n\nAll remote connections require VPN."

    prompt = build_rag_prompt(query=query, context=context)

    assert SYSTEM_RAG_INSTRUCTIONS in prompt
    assert "--- CONTEXT START ---" in prompt
    assert context in prompt
    assert "USER QUESTION:\nWhat is the security policy?" in prompt
    assert "GROUNDED ANSWER:" in prompt


def test_build_rag_prompt_empty_context():
    """Test build_rag_prompt handles empty context string gracefully."""
    prompt = build_rag_prompt(query="Test Question", context="")

    assert "USER QUESTION:\nTest Question" in prompt
    assert "--- CONTEXT START ---" in prompt
