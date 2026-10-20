"""Unit test suite for QueryRouter deterministic heuristic classification."""

import pytest
from app.services.query_router import QueryRouter, RouteDecision


@pytest.fixture
def router() -> QueryRouter:
    """Fixture providing a QueryRouter instance."""
    return QueryRouter()


def test_simple_queries(router):
    """Test straightforward, single-concept questions are classified as simple route."""
    simple_queries = [
        "What is the company's leave policy?",
        "Where is the office located?",
        "What is the password policy?",
        "How do I submit an expense report?",
    ]

    for q in simple_queries:
        decision = router.classify_query(q)
        assert isinstance(decision, RouteDecision)
        assert decision.route == "simple"
        assert decision.confidence >= 0.50
        assert "simple" in decision.reason.lower() or "straightforward" in decision.reason.lower()


def test_complex_queries(router):
    """Test queries with comparison, analytical phrasing, or multiple clauses are classified as complex route."""
    complex_queries = [
        "Compare the company's remote work and office work policies.",
        "Why is the security policy important and what are its main advantages?",
        "Analyze the relationship between access control and incident response.",
        "What is the difference between sick leave versus vacation days and how do they impact benefits?",
    ]

    for q in complex_queries:
        decision = router.classify_query(q)
        assert isinstance(decision, RouteDecision)
        assert decision.route == "complex"
        assert decision.confidence >= 0.60
        assert "Complex query detected" in decision.reason


def test_empty_query_default(router):
    """Test empty or whitespace-only queries default to simple route."""
    decision = router.classify_query("   ")
    assert decision.route == "simple"
    assert decision.confidence == 0.50
