"""Unit test suite for health check endpoint."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Test GET /health returns 200 OK and expected JSON structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Adaptive RAG System"
    assert "version" in data
