"""Integration tests for RAG evaluation API endpoints."""

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.schemas.evaluation import EvaluationRunResponse, EvaluationMetricsSummary, SampleEvaluationResult

client = TestClient(app)


class TestEvaluationAPI:
    """Test suite for evaluation REST endpoints."""

    def test_get_evaluation_dataset_default(self):
        """Tests GET /api/v1/evaluation/dataset successfully returns benchmark dataset."""
        response = client.get("/api/v1/evaluation/dataset")
        assert response.status_code == 200
        data = response.json()
        assert "dataset_path" in data
        assert "total_samples" in data
        assert data["total_samples"] >= 8
        assert "samples" in data
        assert len(data["samples"]) == data["total_samples"]

    def test_get_evaluation_dataset_not_found(self):
        """Tests GET /api/v1/evaluation/dataset with nonexistent path returns 404."""
        response = client.get(
            "/api/v1/evaluation/dataset",
            params={"dataset_path": "data/evaluation/nonexistent.json"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_evaluation_dataset_path_traversal(self):
        """Tests GET /api/v1/evaluation/dataset with path traversal returns 400."""
        response = client.get(
            "/api/v1/evaluation/dataset",
            params={"dataset_path": "../../secret.json"},
        )
        assert response.status_code == 400
        assert "workspace" in response.json()["detail"].lower()

    @patch("app.api.routes.evaluation.evaluation_service.run_evaluation")
    def test_post_evaluation_run_success(self, mock_run):
        """Tests POST /api/v1/evaluation/run returns 200 and structured response."""
        mock_summary = EvaluationMetricsSummary(
            total_samples=1,
            successful_runs=1,
            failed_runs=0,
            mean_hit_rate=1.0,
            mean_mrr=1.0,
            mean_answer_f1=0.85,
            mean_grounding_score=0.90,
            refusal_accuracy=1.0,
            empty_context_count=0,
            missing_sources_count=0,
            mean_latencies_ms={"retrieval_ms": 12.0, "total_ms": 150.0},
        )
        mock_sample_res = SampleEvaluationResult(
            id="test-1",
            question="What is the leave policy?",
            category="simple",
            is_answerable=True,
            generated_answer="25 days annual leave",
            expected_answer="25 days annual leave",
            route="simple",
            retrieved_sources=["leave_policy.pdf"],
            expected_sources=["leave_policy.pdf"],
            hit_rate=1.0,
            reciprocal_rank=1.0,
            answer_f1=1.0,
            answer_precision=1.0,
            answer_recall=1.0,
            grounding_score=1.0,
            is_refusal=False,
            refusal_correct=True,
            empty_context=False,
            missing_sources=False,
            timings={"total_ms": 150.0},
            status="success",
        )
        mock_run.return_value = EvaluationRunResponse(
            summary=mock_summary,
            sample_results=[mock_sample_res],
            errors=[],
            status="success",
        )

        payload = {
            "dataset_path": "data/evaluation/eval_dataset.json",
            "use_reranker": True,
        }
        response = client.post("/api/v1/evaluation/run", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["summary"]["total_samples"] == 1
        assert data["summary"]["mean_hit_rate"] == 1.0
        assert data["summary"]["mean_mrr"] == 1.0
        assert data["summary"]["mean_answer_f1"] == 0.85
        assert len(data["sample_results"]) == 1
        assert data["sample_results"][0]["id"] == "test-1"

    @patch("app.api.routes.evaluation.evaluation_service.run_evaluation")
    def test_post_evaluation_run_file_not_found(self, mock_run):
        """Tests POST /api/v1/evaluation/run maps FileNotFoundError to 404."""
        mock_run.side_effect = FileNotFoundError("Dataset not found at data/missing.json")

        response = client.post(
            "/api/v1/evaluation/run",
            json={"dataset_path": "data/missing.json"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @patch("app.api.routes.evaluation.evaluation_service.run_evaluation")
    def test_post_evaluation_run_bad_request(self, mock_run):
        """Tests POST /api/v1/evaluation/run maps ValueError to 400."""
        mock_run.side_effect = ValueError("Invalid dataset format")

        response = client.post(
            "/api/v1/evaluation/run",
            json={"dataset_path": "data/invalid.json"},
        )
        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()
