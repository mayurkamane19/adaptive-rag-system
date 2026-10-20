"""Service module for conversational query rewriting and reference resolution."""

from typing import List, Tuple, Optional
from app.models.message import ConversationMessage
from app.services.llm_service import LLMService
from app.prompts.conversation_prompts import build_query_rewrite_prompt
from app.core.logging_config import logger


class QueryRewriter:
    """Service responsible for transforming follow-up conversational questions into standalone search queries."""

    def __init__(self, llm_service: Optional[LLMService] = None) -> None:
        self.llm_service = llm_service or LLMService()
        logger.info("Initialized QueryRewriter service")

    def rewrite_query(
        self, current_query: str, history: List[ConversationMessage]
    ) -> Tuple[str, bool]:
        """Rewrites follow-up user query into a standalone query using conversation history.

        Returns:
            Tuple[str, bool]: (standalone_query, was_rewritten_flag)
        """
        if not current_query or not current_query.strip():
            return "", False

        clean_query = current_query.strip()

        # If no conversation history is available, return original query
        if not history:
            logger.info(f"No conversation history available. Keeping query as standalone: '{clean_query}'")
            return clean_query, False

        prompt = build_query_rewrite_prompt(clean_query, history)
        logger.info(f"Attempting query rewriting for: '{clean_query}' with {len(history)} history message(s)")

        try:
            raw_rewritten = self.llm_service.generate_answer(prompt, max_tokens=60)
            standalone_query = raw_rewritten.strip().strip('"').strip("'")

            # Validate generated standalone query
            if (
                standalone_query
                and len(standalone_query) > 3
                and standalone_query.lower() != clean_query.lower()
            ):
                logger.info(f"Successfully rewritten query: '{clean_query}' -> '{standalone_query}'")
                return standalone_query, True

            logger.info("Query rewriter returned identical or un-rewritten query. Preserving original.")
            return clean_query, False

        except Exception as e:
            logger.warning(
                f"Query rewriting LLM call failed or unavailable ({str(e)}). Falling back to original query."
            )
            return clean_query, False
