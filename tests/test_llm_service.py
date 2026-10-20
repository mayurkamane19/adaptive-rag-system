"""Unit test suite for LLMService local generation and Ollama integration."""

import pytest
from unittest.mock import MagicMock, patch
import httpx

from app.services.llm_service import LLMService, LLMServiceError


def test_llm_service_initialization():
    """Test LLMService initializes with default settings."""
    service = LLMService()
    assert service.provider == "ollama"
    assert service.model == "llama3"
    assert service.temperature == 0.2
    assert service.max_tokens == 1000


def test_generate_answer_empty_prompt():
    """Test generate_answer raises ValueError for empty prompt string."""
    service = LLMService()
    with pytest.raises(ValueError, match="cannot be empty"):
        service.generate_answer("   ")


@patch("httpx.Client")
def test_generate_answer_ollama_success(mock_client_class):
    """Test LLMService generates text successfully using a mocked Ollama HTTP client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "The annual vacation allowance is 20 paid days."}
    mock_response.raise_for_status.return_value = None
    mock_client.post.return_value = mock_response
    mock_client_class.return_value.__enter__.return_value = mock_client

    service = LLMService(provider="ollama", model="llama3")
    answer = service.generate_answer("What is the vacation allowance?")

    assert answer == "The annual vacation allowance is 20 paid days."
    mock_client.post.assert_called_once()


@patch("httpx.Client")
def test_generate_answer_ollama_connection_error(mock_client_class):
    """Test LLMService raises LLMServiceError gracefully when Ollama server is unreachable."""
    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.ConnectError("Connection refused")
    mock_client_class.return_value.__enter__.return_value = mock_client

    service = LLMService(provider="ollama", model="llama3")
    with pytest.raises(LLMServiceError, match="Could not connect to Ollama server"):
        service.generate_answer("Sample query")


def test_llm_service_configurable_base_url():
    """Test LLMService respects configurable Ollama base URL."""
    custom_url = "http://192.168.1.100:11434"
    service = LLMService(base_url=custom_url)
    assert service.base_url == "http://192.168.1.100:11434"

    # Test trailing slash normalization
    service_slash = LLMService(base_url="http://localhost:11434/")
    assert service_slash.base_url == "http://localhost:11434"


@patch("httpx.Client")
def test_generate_answer_ollama_model_not_found(mock_client_class):
    """Test LLMService raises informative error when model is not found (HTTP 404)."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "model 'llama3' not found"
    error = httpx.HTTPStatusError("Not Found", request=MagicMock(), response=mock_response)
    mock_client.post.side_effect = error
    mock_client_class.return_value.__enter__.return_value = mock_client

    service = LLMService(provider="ollama", model="llama3")
    with pytest.raises(LLMServiceError, match="Model 'llama3' was not found"):
        service.generate_answer("What is the leave policy?")


@patch("httpx.Client")
def test_generate_answer_ollama_timeout(mock_client_class):
    """Test LLMService raises LLMServiceError on request timeout."""
    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.TimeoutException("Read timed out")
    mock_client_class.return_value.__enter__.return_value = mock_client

    service = LLMService(provider="ollama", model="llama3", timeout=10.0)
    with pytest.raises(LLMServiceError, match="timed out after 10.0s"):
        service.generate_answer("Query that times out")


@patch("httpx.Client")
def test_check_connection_success(mock_client_class):
    """Test check_connection reports connected and model_available when model is present."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [{"name": "llama3:latest"}, {"name": "mistral:latest"}]
    }
    mock_client.get.return_value = mock_response
    mock_client_class.return_value.__enter__.return_value = mock_client

    service = LLMService(provider="ollama", model="llama3")
    result = service.check_connection()

    assert result["connected"] is True
    assert result["status"] == "connected"
    assert result["model_available"] is True


@patch("httpx.Client")
def test_check_connection_model_missing(mock_client_class):
    """Test check_connection reports model_missing when server is up but model is not pulled."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [{"name": "mistral:latest"}]
    }
    mock_client.get.return_value = mock_response
    mock_client_class.return_value.__enter__.return_value = mock_client

    service = LLMService(provider="ollama", model="llama3")
    result = service.check_connection()

    assert result["connected"] is True
    assert result["status"] == "model_missing"
    assert result["model_available"] is False


@patch("httpx.Client")
def test_check_connection_unreachable(mock_client_class):
    """Test check_connection reports unreachable when Ollama server is offline."""
    mock_client = MagicMock()
    mock_client.get.side_effect = httpx.ConnectError("Connection refused")
    mock_client_class.return_value.__enter__.return_value = mock_client

    service = LLMService(provider="ollama", model="llama3")
    result = service.check_connection()

    assert result["connected"] is False
    assert result["status"] == "unreachable"
    assert result["model_available"] is False
    assert "Please ensure Ollama is installed" in result["message"]
