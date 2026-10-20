"""Service module for local LLM text generation abstractions (Ollama integration)."""

import time
import threading
from typing import Optional, Dict, Any
import httpx

from app.core.config import settings
from app.core.logging_config import logger


class LLMServiceError(Exception):
    """Raised when an error occurs during LLM initialization or generation."""
    pass


class LLMService:
    """Service providing provider-agnostic LLM text generation with local Ollama support."""

    _generation_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        provider: str = settings.LLM_PROVIDER,
        model: str = settings.LLM_MODEL,
        temperature: float = settings.LLM_TEMPERATURE,
        max_tokens: int = settings.LLM_MAX_TOKENS,
        base_url: str = settings.OLLAMA_BASE_URL,
        timeout: float = settings.OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.last_generation_metrics: Dict[str, Any] = {}
        logger.info(
            f"Initialized LLMService (provider='{self.provider}', model='{self.model}', "
            f"base_url='{self.base_url}', temp={self.temperature}, max_tokens={self.max_tokens}, timeout={self.timeout}s)"
        )

    def check_connection(self) -> Dict[str, Any]:
        """Probes Ollama server connectivity and model availability."""
        if self.provider != "ollama":
            return {
                "connected": False,
                "status": "unsupported_provider",
                "provider": self.provider,
                "message": f"Provider '{self.provider}' is not supported for live checks.",
            }

        urls_to_try = [f"{self.base_url}/api/tags"]
        if "localhost" in self.base_url:
            urls_to_try.append(f"{self.base_url.replace('localhost', '127.0.0.1')}/api/tags")

        for tags_url in urls_to_try:
            try:
                with httpx.Client(timeout=3.0) as client:
                    res = client.get(tags_url)
                    if res.status_code == 200:
                        data = res.json()
                        models = [m.get("name", "") for m in data.get("models", [])]
                        model_available = any(
                            m == self.model or m.startswith(f"{self.model}:") or self.model in m
                            for m in models
                        )
                        return {
                            "connected": True,
                            "status": "connected" if model_available else "model_missing",
                            "base_url": self.base_url,
                            "model": self.model,
                            "model_available": model_available,
                            "available_models": models,
                            "message": (
                                f"Connected to Ollama. Model '{self.model}' is available."
                                if model_available
                                else f"Connected to Ollama, but model '{self.model}' is not pulled yet."
                            ),
                        }
            except (httpx.ConnectError, httpx.TimeoutException):
                continue
            except Exception as e:
                logger.warning(f"Error checking Ollama connection at {tags_url}: {str(e)}")
                continue

        return {
            "connected": False,
            "status": "unreachable",
            "base_url": self.base_url,
            "model": self.model,
            "model_available": False,
            "message": (
                f"Could not connect to Ollama server at {self.base_url}. "
                f"Please ensure Ollama is installed and running ('ollama serve')."
            ),
        }

    def _extract_ollama_metrics(self, data: Dict[str, Any], wall_duration_s: float, effective_num_predict: int) -> Dict[str, Any]:
        """Extracts Ollama timing and token evaluation metrics from API response payload."""
        prompt_eval_count = data.get("prompt_eval_count", 0)
        prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)
        prompt_eval_ms = round(prompt_eval_duration_ns / 1e6, 2)

        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 0)
        eval_ms = round(eval_duration_ns / 1e6, 2)

        load_duration_ns = data.get("load_duration", 0)
        load_ms = round(load_duration_ns / 1e6, 2)

        total_duration_ns = data.get("total_duration", 0)
        total_ms = round(total_duration_ns / 1e6, 2)

        speed_tok_s = round(eval_count / (eval_duration_ns / 1e9), 2) if eval_duration_ns > 0 else 0.0

        metrics = {
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_ms": prompt_eval_ms,
            "eval_count": eval_count,
            "eval_ms": eval_ms,
            "load_ms": load_ms,
            "total_ms": total_ms,
            "tokens_per_second": speed_tok_s,
            "num_predict": effective_num_predict,
            "wall_duration_s": round(wall_duration_s, 2),
        }
        self.last_generation_metrics = metrics
        return metrics

    def _generate_ollama(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Sends prompt to local Ollama API server and returns generated text response."""
        url = f"{self.base_url}/api/generate"
        effective_num_predict = min(max_tokens if max_tokens is not None else self.max_tokens, 250)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "60m",
            "options": {
                "temperature": self.temperature,
                "num_predict": effective_num_predict,
            },
        }

        t_gen_start = time.perf_counter()
        logger.info(
            f"Sending generation request to Ollama ({url}): model='{self.model}', "
            f"options={{temperature: {self.temperature}, num_predict: {effective_num_predict}}}, "
            f"keep_alive='60m', prompt_chars={len(prompt)}"
        )

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                wall_duration = time.perf_counter() - t_gen_start
                answer = data.get("response", "").strip()
                if not answer:
                    raise LLMServiceError("Ollama returned an empty response.")

                metrics = self._extract_ollama_metrics(data, wall_duration, effective_num_predict)
                logger.info(
                    f"Ollama generation completed in {wall_duration:.2f}s | "
                    f"Prompt: {metrics['prompt_eval_count']} tokens ({metrics['prompt_eval_ms']}ms) | "
                    f"Gen: {metrics['eval_count']} tokens ({metrics['eval_ms']}ms) | "
                    f"Speed: {metrics['tokens_per_second']} tok/s (num_predict={effective_num_predict})"
                )
                return answer

        except httpx.ConnectError as e:
            # Fallback attempt: if connecting via localhost failed, try 127.0.0.1 directly for Windows IPv4 binding
            if "localhost" in self.base_url:
                alt_url = f"{self.base_url.replace('localhost', '127.0.0.1')}/api/generate"
                try:
                    with httpx.Client(timeout=self.timeout) as client:
                        alt_res = client.post(alt_url, json=payload)
                        alt_res.raise_for_status()
                        alt_data = alt_res.json()
                        wall_duration = time.perf_counter() - t_gen_start
                        alt_ans = alt_data.get("response", "").strip()
                        if alt_ans:
                            metrics = self._extract_ollama_metrics(alt_data, wall_duration, effective_num_predict)
                            logger.info(
                                f"Ollama generation succeeded via IPv4 fallback URL in {wall_duration:.2f}s | "
                                f"Prompt: {metrics['prompt_eval_count']} tokens ({metrics['prompt_eval_ms']}ms) | "
                                f"Gen: {metrics['eval_count']} tokens ({metrics['eval_ms']}ms) | "
                                f"Speed: {metrics['tokens_per_second']} tok/s"
                            )
                            return alt_ans
                except Exception:
                    pass

            msg = (
                f"Could not connect to Ollama server at '{self.base_url}'. "
                f"Please ensure Ollama is installed and running. "
                f"On Windows, start Ollama by running 'ollama serve' in PowerShell/Command Prompt "
                f"or launching the Ollama desktop app, and ensure '{self.model}' is downloaded via 'ollama pull {self.model}'."
            )
            logger.error(msg)
            raise LLMServiceError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Ollama generation request timed out after {self.timeout}s for model '{self.model}' at '{self.base_url}'."
            logger.error(msg)
            raise LLMServiceError(msg) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                msg = (
                    f"Model '{self.model}' was not found on Ollama server at '{self.base_url}'. "
                    f"Please run 'ollama pull {self.model}' to download the model."
                )
            else:
                msg = f"Ollama HTTP {e.response.status_code} error from '{self.base_url}': {e.response.text}"
            logger.error(msg)
            raise LLMServiceError(msg) from e
        except Exception as e:
            if isinstance(e, LLMServiceError):
                raise
            logger.error(f"Unexpected error calling Ollama API: {str(e)}")
            raise LLMServiceError(f"LLM generation failed: {str(e)}") from e

    def generate_answer(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generates answer using the configured LLM provider with concurrency protection."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt text cannot be empty.")

        if self.provider != "ollama":
            raise NotImplementedError(
                f"LLM provider '{self.provider}' is not currently implemented. Only 'ollama' is supported in Phase 6."
            )

        lock_acquired = self._generation_lock.acquire(timeout=self.timeout)
        if not lock_acquired:
            msg = f"LLM inference engine busy: another generation task is active. Timed out after {self.timeout}s waiting for lock."
            logger.error(msg)
            raise LLMServiceError(msg)

        try:
            return self._generate_ollama(prompt, max_tokens=max_tokens)
        finally:
            self._generation_lock.release()


llm_service = LLMService()
