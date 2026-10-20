"""Automated test suite for Phase 10 production hardening, security, health, and storage persistence."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.core.config import settings
from app.core.security import verify_api_key, InMemoryRateLimiter
from app.services.conversation_service import ConversationService

client = TestClient(app)


class TestHealthAndReadiness:
    """Test suite for health, liveness, and readiness probe endpoints."""

    def test_legacy_health_endpoint(self):
        """Tests that GET /health remains functional and backward-compatible."""
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "service" in data

    def test_liveness_probe(self):
        """Tests GET /health/live returns process alive status."""
        res = client.get("/health/live")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "alive"
        assert data["service"] == settings.APP_NAME

    def test_readiness_probe_structure(self):
        """Tests GET /health/ready returns structured subsystem status."""
        res = client.get("/health/ready")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ("ready", "degraded")
        assert "components" in data
        comps = data["components"]
        assert "vector_store" in comps
        assert "llm_service" in comps
        assert "reranker" in comps
        assert comps["vector_store"]["status"] in ("available", "empty", "unavailable")


class TestSecurityHardening:
    """Test suite for API Key authentication, rate limiting, and request limits."""

    def test_api_key_auth_bypassed_in_dev(self):
        """Tests that when API_KEY is unset or empty, requests pass without header."""
        with patch.object(settings, "API_KEY", ""):
            with patch("app.api.routes.evaluation.evaluation_service.run_evaluation") as mock_run:
                mock_run.return_value = {
                    "summary": {
                        "total_samples": 0,
                        "successful_runs": 0,
                        "failed_runs": 0,
                        "mean_answer_f1": 0.0,
                        "mean_grounding_score": 0.0,
                    },
                    "sample_results": [],
                    "errors": [],
                    "status": "empty",
                }
                # Sensitive endpoint: evaluation run
                res = client.post("/api/v1/evaluation/run", json={"dataset_path": "data/evaluation/eval_dataset.json"})
                # Should not be 401 Unauthorized
                assert res.status_code != 401

    def test_api_key_auth_enforced_when_configured(self):
        """Tests that when API_KEY is set, missing or invalid header returns 401."""
        with patch.object(settings, "API_KEY", "prod-secret-token"):
            # Missing header
            res_missing = client.post("/api/v1/evaluation/run", json={})
            assert res_missing.status_code == 401
            assert "missing" in res_missing.json()["detail"].lower()

            # Invalid header
            res_invalid = client.post(
                "/api/v1/evaluation/run",
                json={},
                headers={"X-API-Key": "wrong-token"},
            )
            assert res_invalid.status_code == 401
            assert "invalid" in res_invalid.json()["detail"].lower()

    def test_api_key_auth_success_with_valid_token(self):
        """Tests that when API_KEY is set, valid X-API-Key header succeeds."""
        with patch.object(settings, "API_KEY", "prod-secret-token"):
            with patch("app.api.routes.evaluation.evaluation_service.run_evaluation") as mock_run:
                mock_run.return_value = {
                    "summary": {
                        "total_samples": 0,
                        "successful_runs": 0,
                        "failed_runs": 0,
                        "mean_answer_f1": 0.0,
                        "mean_grounding_score": 0.0,
                    },
                    "sample_results": [],
                    "errors": [],
                    "status": "empty",
                }
                res = client.post(
                    "/api/v1/evaluation/run",
                    json={},
                    headers={"X-API-Key": "prod-secret-token"},
                )
                assert res.status_code == 200

    def test_in_memory_rate_limiter_exceeded(self):
        """Tests in-memory rate limiter triggers when requests exceed limit."""
        limiter = InMemoryRateLimiter(requests_per_minute=2, enabled=True)
        ip = "192.168.1.100"

        assert limiter.is_rate_limited(ip) is False  # 1st request
        assert limiter.is_rate_limited(ip) is False  # 2nd request
        assert limiter.is_rate_limited(ip) is True   # 3rd request (exceeded limit)

    def test_max_upload_size_limit_enforced(self):
        """Tests that document uploads exceeding MAX_UPLOAD_SIZE_MB return HTTP 413."""
        # Create a payload larger than 1MB when MAX_UPLOAD_SIZE_MB is set to 1
        large_bytes = b"%PDF-1.4\n" + b"X" * (1024 * 1024 + 100)
        files = {"file": ("large.pdf", large_bytes, "application/pdf")}

        with patch.object(settings, "MAX_UPLOAD_SIZE_MB", 1):
            res = client.post("/api/v1/documents/upload", files=files)
            assert res.status_code == 413
            assert "exceeds maximum allowed limit" in res.json()["detail"].lower()


class TestStoragePersistence:
    """Test suite for conversation session disk persistence and restoration."""

    def test_conversation_persists_to_disk_and_restores(self, tmp_path):
        """Tests that messages added to a session are saved to disk and reloaded by new service instances."""
        svc1 = ConversationService(storage_path=tmp_path)
        session_id = "test-persist-session-1"

        svc1.add_message(session_id, "user", "What is the policy?")
        svc1.add_message(session_id, "assistant", "The policy provides 25 days leave.")

        # Verify JSON file was written on disk
        expected_file = tmp_path / f"{session_id}.json"
        assert expected_file.is_file()

        with open(expected_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["content"] == "What is the policy?"

        # Simulate service restart: create a new instance pointing to same storage
        svc2 = ConversationService(storage_path=tmp_path)
        assert svc2.session_exists(session_id) is True

        history = svc2.get_history(session_id)
        assert len(history) == 2
        assert history[0].content == "What is the policy?"
        assert history[1].content == "The policy provides 25 days leave."

    def test_clear_session_removes_disk_file(self, tmp_path):
        """Tests that clear_session purges both in-memory cache and disk file."""
        svc = ConversationService(storage_path=tmp_path)
        session_id = "test-clear-session"

        svc.add_message(session_id, "user", "Hello")
        session_file = tmp_path / f"{session_id}.json"
        assert session_file.is_file()

        cleared = svc.clear_session(session_id)
        assert cleared is True
        assert not session_file.exists()
        assert svc.session_exists(session_id) is False
