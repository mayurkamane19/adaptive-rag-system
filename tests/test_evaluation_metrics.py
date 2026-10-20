"""Unit tests for deterministic RAG evaluation metrics calculation."""

import pytest
from app.services.evaluation_service import EvaluationService


class TestEvaluationMetrics:
    """Test suite for deterministic metric calculation utilities."""

    def test_normalize_text(self):
        """Tests text tokenization and punctuation stripping."""
        text = "Hello, World! This is a TEST: 123."
        tokens = EvaluationService.normalize_text(text)
        assert tokens == ["hello", "world", "this", "is", "a", "test", "123"]

    def test_normalize_text_empty(self):
        """Tests empty text normalization."""
        assert EvaluationService.normalize_text("") == []
        assert EvaluationService.normalize_text("   ") == []

    def test_compute_token_f1_exact_match(self):
        """Tests token F1 calculation when prediction matches ground truth exactly."""
        pred = "The annual leave allowance is 25 days."
        gold = "The annual leave allowance is 25 days."
        f1, prec, rec = EvaluationService.compute_token_f1(pred, gold)
        assert f1 == 1.0
        assert prec == 1.0
        assert rec == 1.0

    def test_compute_token_f1_both_empty(self):
        """Tests token F1 calculation when both strings are empty."""
        f1, prec, rec = EvaluationService.compute_token_f1("", "")
        assert f1 == 1.0
        assert prec == 1.0
        assert rec == 1.0

    def test_compute_token_f1_one_empty(self):
        """Tests token F1 calculation when one string is empty."""
        f1, prec, rec = EvaluationService.compute_token_f1("some text", "")
        assert f1 == 0.0
        assert prec == 0.0
        assert rec == 0.0

        f1, prec, rec = EvaluationService.compute_token_f1("", "some text")
        assert f1 == 0.0
        assert prec == 0.0
        assert rec == 0.0

    def test_compute_token_f1_disjoint(self):
        """Tests token F1 calculation with zero token overlap."""
        pred = "Apples oranges bananas"
        gold = "Cars motorcycles bicycles"
        f1, prec, rec = EvaluationService.compute_token_f1(pred, gold)
        assert f1 == 0.0
        assert prec == 0.0
        assert rec == 0.0

    def test_compute_token_f1_partial_overlap(self):
        """Tests token F1 with partial token overlap."""
        pred = "employees receive 25 days leave"
        gold = "full-time employees receive 25 days paid annual leave"
        f1, prec, rec = EvaluationService.compute_token_f1(pred, gold)
        assert 0.0 < f1 < 1.0
        assert 0.0 < prec <= 1.0
        assert 0.0 < rec < 1.0

    def test_compute_hit_rate_found(self):
        """Tests Hit Rate (Recall@K) when expected source is retrieved."""
        retrieved = ["data/documents/handbook.pdf", "data/documents/leave_policy.pdf"]
        expected = ["leave_policy.pdf"]
        hit_rate = EvaluationService.compute_hit_rate(retrieved, expected)
        assert hit_rate == 1.0

    def test_compute_hit_rate_not_found(self):
        """Tests Hit Rate when expected source is not in retrieved sources."""
        retrieved = ["data/documents/handbook.pdf"]
        expected = ["leave_policy.pdf"]
        hit_rate = EvaluationService.compute_hit_rate(retrieved, expected)
        assert hit_rate == 0.0

    def test_compute_hit_rate_no_expected(self):
        """Tests Hit Rate when question has no expected sources."""
        retrieved = ["data/documents/handbook.pdf"]
        expected = []
        hit_rate = EvaluationService.compute_hit_rate(retrieved, expected)
        assert hit_rate is None

    def test_compute_hit_rate_empty_retrieved(self):
        """Tests Hit Rate when no sources were retrieved."""
        retrieved = []
        expected = ["leave_policy.pdf"]
        hit_rate = EvaluationService.compute_hit_rate(retrieved, expected)
        assert hit_rate == 0.0

    def test_compute_mrr_first_rank(self):
        """Tests MRR when first retrieved document matches."""
        retrieved = ["leave_policy.pdf", "handbook.pdf"]
        expected = ["leave_policy.pdf"]
        mrr = EvaluationService.compute_mrr(retrieved, expected)
        assert mrr == 1.0

    def test_compute_mrr_second_rank(self):
        """Tests MRR when second retrieved document matches."""
        retrieved = ["handbook.pdf", "leave_policy.pdf"]
        expected = ["leave_policy.pdf"]
        mrr = EvaluationService.compute_mrr(retrieved, expected)
        assert mrr == 0.5

    def test_compute_mrr_third_rank(self):
        """Tests MRR when third retrieved document matches."""
        retrieved = ["doc1.pdf", "doc2.pdf", "leave_policy.pdf"]
        expected = ["leave_policy.pdf"]
        mrr = EvaluationService.compute_mrr(retrieved, expected)
        assert mrr == 0.3333

    def test_compute_mrr_not_found(self):
        """Tests MRR when expected source is absent."""
        retrieved = ["doc1.pdf", "doc2.pdf"]
        expected = ["leave_policy.pdf"]
        mrr = EvaluationService.compute_mrr(retrieved, expected)
        assert mrr == 0.0

    def test_compute_mrr_no_expected(self):
        """Tests MRR when expected sources list is empty."""
        retrieved = ["doc1.pdf"]
        expected = []
        mrr = EvaluationService.compute_mrr(retrieved, expected)
        assert mrr is None

    def test_is_standard_refusal_patterns(self):
        """Tests standard refusal detector across various standard phrasing."""
        assert EvaluationService.is_standard_refusal(
            "I couldn't find enough information in the provided documents to answer this question."
        ) is True
        assert EvaluationService.is_standard_refusal(
            "I could not find enough information to answer."
        ) is True
        assert EvaluationService.is_standard_refusal(
            "There is not enough information in the provided documents."
        ) is True

    def test_is_standard_refusal_non_refusal(self):
        """Tests standard refusal detector returns False for informative answers."""
        assert EvaluationService.is_standard_refusal(
            "Employees are entitled to 25 days of paid annual leave."
        ) is False
        assert EvaluationService.is_standard_refusal("") is False

    def test_compute_grounding_score_full(self):
        """Tests lexical grounding score when all answer content words exist in context."""
        context = "Employees receive twenty-five days of paid annual vacation each year."
        answer = "Employees receive twenty-five days of annual vacation."
        score = EvaluationService.compute_grounding_score(answer, context)
        assert score == 1.0

    def test_compute_grounding_score_refusal_empty_context(self):
        """Tests lexical grounding score for refusal answer when context is empty."""
        refusal = "I couldn't find enough information in the provided documents to answer this question."
        score = EvaluationService.compute_grounding_score(refusal, "")
        assert score == 1.0

    def test_compute_grounding_score_unsupported(self):
        """Tests lexical grounding score when answer hallucinated tokens not present in context."""
        context = "The office opens at 9am."
        answer = "Employees receive cryptocurrency bonuses during holidays."
        score = EvaluationService.compute_grounding_score(answer, context)
        assert score == 0.0
