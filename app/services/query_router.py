"""Service module for deterministic query classification and adaptive routing decision making."""

from typing import Literal
from pydantic import BaseModel, Field
from app.core.logging_config import logger


class RouteDecision(BaseModel):
    """Data model representing a query routing decision."""

    route: Literal["simple", "complex"] = Field(..., description="Target routing path ('simple' or 'complex')")
    reason: str = Field(..., description="Explainable deterministic heuristic rationale for decision")
    confidence: float = Field(..., description="Heuristic confidence score for routing decision (0.5 - 0.95)")


class QueryRouter:
    """Deterministic, explainable classifier for routing queries into simple or complex retrieval pathways."""

    COMPLEXITY_KEYWORDS = {
        "comparison": ["compare", "comparison", "difference", "versus", "vs", "contrast", "relationship", "similarities"],
        "analytical": ["why", "explain", "analyze", "analysis", "advantages and disadvantages", "pros and cons", "tradeoffs", "impact", "summary", "summarize", "detailed", "overview"],
        "multi_concept": ["and also", "as well as", "in addition to", "furthermore", "both"],
    }

    def __init__(self) -> None:
        logger.info("Initialized QueryRouter classifier service")

    def classify_query(self, query: str) -> RouteDecision:
        """Classifies an incoming query into 'simple' or 'complex' using transparent heuristic signals."""
        if not query or not query.strip():
            return RouteDecision(
                route="simple",
                reason="Query is empty or whitespace; defaulting to simple route.",
                confidence=0.50,
            )

        clean_query = query.strip().lower()
        words = clean_query.split()
        word_count = len(words)
        question_marks = clean_query.count("?")

        reasons = []
        score = 0.0

        # Check for comparison keywords
        found_comparison = [kw for kw in self.COMPLEXITY_KEYWORDS["comparison"] if kw in clean_query]
        if found_comparison:
            score += 0.35
            reasons.append(f"comparison language ('{', '.join(found_comparison)}')")

        # Check for analytical keywords
        found_analytical = [kw for kw in self.COMPLEXITY_KEYWORDS["analytical"] if kw in clean_query]
        if found_analytical:
            score += 0.30
            reasons.append(f"analytical phrasing ('{', '.join(found_analytical)}')")

        # Check for multi-concept conjunctions
        found_multi = [kw for kw in self.COMPLEXITY_KEYWORDS["multi_concept"] if kw in clean_query]
        if found_multi:
            score += 0.25
            reasons.append(f"multi-concept connectors ('{', '.join(found_multi)}')")

        # Check for query length
        if word_count >= 15 or len(clean_query) >= 90:
            score += 0.25
            reasons.append(f"high query length ({word_count} words)")

        # Check for multiple questions
        if question_marks > 1:
            score += 0.30
            reasons.append(f"multiple question marks ({question_marks})")

        # Classification Thresholding
        if score >= 0.30:
            route = "complex"
            confidence = min(0.95, round(0.65 + (score * 0.25), 2))
            reason_str = f"Complex query detected due to: {', '.join(reasons)}."
        else:
            route = "simple"
            confidence = min(0.90, round(0.70 + ((1.0 - score) * 0.15), 2))
            reason_str = "Straightforward, targeted question; routed to simple retrieval path."

        logger.info(f"Query routing decision: route='{route}', confidence={confidence}, reason='{reason_str}'")
        return RouteDecision(
            route=route,
            reason=reason_str,
            confidence=confidence,
        )
