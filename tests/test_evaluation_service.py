"""Unit and integration tests for EvaluationService and benchmark runner."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.schemas.evaluation import EvaluationSample, EvaluationRunResponse
from app.schemas.chat import SourceReference
from app.services.evaluation_service import EvaluationService
from app.services.rag_service import RAGService
from app.core.config import settings


class TestEvaluationService:
    """Test suite for EvaluationService operations."""

    def test_load_dataset_success(self):
        """Tests loading valid default evaluation benchmark dataset from disk."""
        service = EvaluationService()
        samples = service.load_dataset()
        assert len(samples) >= 8
        assert all(isinstance(s, EvaluationSample) for s in samples)
        assert any(s.category == "unanswerable" for s in samples)
        assert any(s.category == "complex" for s in samples)

    def test_load_dataset_file_not_found(self):
        """Tests that loading a non-existent dataset raises FileNotFoundError."""
        service = EvaluationService()
        with pytest.raises(FileNotFoundError):
            service.load_dataset("data/evaluation/non_existent_file.json")

    def test_load_dataset_path_traversal_rejected(self):
        """Tests that path outside workspace triggers security ValueError."""
        service = EvaluationService()
        with pytest.raises(ValueError, match="within the application workspace"):
            service.load_dataset("../../outside_dataset.json")

    def test_evaluate_sample_answerable(self):
        """Tests single-turn answerable sample evaluation with mocked RAGService."""
        mock_rag = MagicMock(spec=RAGService)
        mock_rag.generate_rag_answer.return_value = {
            "query": "How many days of annual leave do full-time employees receive?",
            "route": "simple",
            "answer": "Full-time employees receive 25 days of paid annual leave.",
            "sources": [
                {
                    "source": "data/documents/leave_policy.pdf",
                    "page": 1,
                    "chunk_id": "chunk-1",
                    "original_filename": "leave_policy.pdf",
                }
            ],
            "context": "Employees are entitled to 25 days of paid annual leave.",
            "retrieved_count": 1,
            "included_count": 1,
            "status": "success",
            "llm_provider": "ollama",
            "llm_model": "llama3",
            "timings": {"retrieval_ms": 10.5, "rerank_ms": 8.2, "generation_ms": 120.0, "total_ms": 138.7},
        }

        service = EvaluationService(rag_service=mock_rag)
        sample = EvaluationSample(
            id="eval-001",
            question="How many days of annual leave do full-time employees receive?",
            expected_answer="Full-time employees are entitled to 25 days of paid annual leave per calendar year.",
            category="simple",
            is_answerable=True,
            expected_sources=["leave_policy.pdf"],
        )

        res = service.evaluate_sample(sample)

        assert res.id == "eval-001"
        assert res.status == "success"
        assert res.hit_rate == 1.0
        assert res.reciprocal_rank == 1.0
        assert res.answer_f1 > 0.5
        assert res.grounding_score > 0.0
        assert res.is_refusal is False
        assert res.refusal_correct is True
        assert res.empty_context is False
        assert res.missing_sources is False
        assert "retrieval_ms" in res.timings

    def test_evaluate_sample_unanswerable_refusal(self):
        """Tests unanswerable question evaluation with standard grounded refusal."""
        mock_rag = MagicMock(spec=RAGService)
        mock_rag.generate_rag_answer.return_value = {
            "query": "What is the policy for company cryptocurrency grants?",
            "route": "simple",
            "answer": "I couldn't find enough information in the provided documents to answer this question.",
            "sources": [],
            "context": "",
            "retrieved_count": 0,
            "included_count": 0,
            "status": "no_results",
            "llm_provider": "ollama",
            "llm_model": "llama3",
            "timings": {"retrieval_ms": 5.0, "rerank_ms": 2.0, "prompt_ms": 0.5, "total_ms": 7.5},
        }

        service = EvaluationService(rag_service=mock_rag)
        sample = EvaluationSample(
            id="eval-004",
            question="What is the policy for company cryptocurrency grants?",
            expected_answer="I couldn't find enough information in the provided documents to answer this question.",
            category="unanswerable",
            is_answerable=False,
            expected_sources=[],
        )

        res = service.evaluate_sample(sample)

        assert res.id == "eval-004"
        assert res.status == "success"
        assert res.is_refusal is True
        assert res.refusal_correct is True
        assert res.grounding_score == 1.0  # Refusal with empty context is 100% faithful
        assert res.empty_context is True
        assert res.missing_sources is False

    def test_evaluate_sample_conversational(self):
        """Tests conversational multi-turn sample routing to conversational_rag_chat."""
        mock_rag = MagicMock(spec=RAGService)
        mock_rag.conversational_rag_chat.return_value = {
            "session_id": "eval-session-eval-007",
            "query": "Can I carry it over to next year?",
            "standalone_query": "Can employees carry over unused annual leave to the next year?",
            "was_rewritten": True,
            "answer": "Employees may carry over up to 5 unused annual leave days into the next calendar year.",
            "sources": [
                {
                    "source": "data/documents/leave_policy.pdf",
                    "page": 2,
                    "chunk_id": "chunk-3",
                    "original_filename": "leave_policy.pdf",
                }
            ],
            "context": "Employees may carry over a maximum of 5 days of unused annual leave.",
            "retrieved_count": 1,
            "included_count": 1,
            "route": "simple",
            "status": "success",
            "llm_provider": "ollama",
            "llm_model": "llama3",
            "timings": {"rewrite_ms": 15.0, "retrieval_ms": 10.0, "total_ms": 25.0},
        }

        service = EvaluationService(rag_service=mock_rag)
        sample = EvaluationSample(
            id="eval-007",
            question="Can I carry it over to next year?",
            expected_answer="Employees may carry over a maximum of 5 days of unused annual leave into the next calendar year.",
            category="conversational",
            is_answerable=True,
            expected_sources=["leave_policy.pdf"],
            conversation_history=[
                {"role": "user", "content": "How many annual leave days do I get?"},
                {"role": "assistant", "content": "You get 25 days of annual leave."},
            ],
        )

        res = service.evaluate_sample(sample)

        mock_rag.conversational_rag_chat.assert_called_once()
        assert res.id == "eval-007"
        assert res.hit_rate == 1.0
        assert res.answer_f1 > 0.5

    def test_evaluate_sample_error_handling(self):
        """Tests that service errors during sample processing produce failed result object."""
        mock_rag = MagicMock(spec=RAGService)
        mock_rag.generate_rag_answer.side_effect = RuntimeError("FAISS index unavailable")

        service = EvaluationService(rag_service=mock_rag)
        sample = EvaluationSample(
            id="eval-err",
            question="What is the policy?",
            expected_answer="Policy details",
            category="simple",
            is_answerable=True,
            expected_sources=["policy.pdf"],
        )

        res = service.evaluate_sample(sample)

        assert res.status == "failed"
        assert "FAISS index unavailable" in res.error
        assert res.answer_f1 == 0.0

    def test_run_evaluation_aggregates_summary(self):
        """Tests full run_evaluation aggregate metric computation over multiple samples."""
        mock_rag = MagicMock(spec=RAGService)
        mock_rag.generate_rag_answer.return_value = {
            "query": "Q",
            "route": "simple",
            "answer": "25 days leave",
            "sources": [{"original_filename": "leave_policy.pdf"}],
            "context": "25 days leave",
            "timings": {"retrieval_ms": 10.0, "total_ms": 20.0},
        }

        service = EvaluationService(rag_service=mock_rag)
        custom_samples = [
            EvaluationSample(
                id="s1",
                question="Q1",
                expected_answer="25 days leave",
                category="simple",
                is_answerable=True,
                expected_sources=["leave_policy.pdf"],
            ),
            EvaluationSample(
                id="s2",
                question="Q2",
                expected_answer="25 days leave",
                category="simple",
                is_answerable=True,
                expected_sources=["leave_policy.pdf"],
            ),
        ]

        resp = service.run_evaluation(custom_samples=custom_samples)

        assert isinstance(resp, EvaluationRunResponse)
        assert resp.status == "success"
        assert resp.summary.total_samples == 2
        assert resp.summary.successful_runs == 2
        assert resp.summary.failed_runs == 0
        assert resp.summary.mean_hit_rate == 1.0
        assert resp.summary.mean_mrr == 1.0
        assert resp.summary.mean_answer_f1 == 1.0
        assert resp.summary.mean_grounding_score == 1.0
        assert len(resp.sample_results) == 2

    def test_run_evaluation_reranker_toggle(self):
        """Tests that use_reranker flag is honored during evaluation and restored afterwards."""
        mock_rag = MagicMock(spec=RAGService)
        mock_rag.generate_rag_answer.return_value = {
            "query": "Q",
            "route": "simple",
            "answer": "answer",
            "sources": [],
            "context": "",
            "timings": {"total_ms": 10.0},
        }

        service = EvaluationService(rag_service=mock_rag)
        original_setting = settings.RERANKER_ENABLED

        sample = EvaluationSample(
            id="s1",
            question="Q",
            expected_answer="A",
            category="simple",
            is_answerable=True,
        )

        service.run_evaluation(custom_samples=[sample], use_reranker=not original_setting)
        # Verify settings was restored to its original value
        assert settings.RERANKER_ENABLED == original_setting
