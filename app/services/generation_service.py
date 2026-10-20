"""Service module for LLM response generation."""

from typing import List, Dict, Any
from app.core.config import settings
from app.core.logging_config import logger


class GenerationService:
    """Service responsible for LLM prompt orchestration and answer generation."""

    def __init__(
        self,
        provider: str = settings.LLM_PROVIDER,
        model: str = settings.LLM_MODEL,
    ) -> None:
        self.provider = provider
        self.model = model
        logger.info(f"Initialized GenerationService with provider='{self.provider}', model='{self.model}'")

    def generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Generates answer using LLM given query and retrieved contexts. Placeholder for Phase 1."""
        logger.info(f"[Phase 1 Placeholder] Generating answer for query: '{query}' using {len(contexts)} contexts")
        return (
            "[Phase 1 Placeholder] The Adaptive RAG System architecture is set up. "
            "LLM generation will be connected in future phases."
        )
