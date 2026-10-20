"""Prompt templates and prompt engineering utilities for grounded RAG answer generation."""

from app.core.logging_config import logger


SYSTEM_RAG_INSTRUCTIONS = """You are an enterprise AI document assistant. Your task is to answer the user's question based strictly and ONLY on the provided document context below.

CRITICAL GROUNDING RULES:
1. Answer using ONLY the facts and context explicitly provided.
2. Do NOT invent, assume, or extrapolate facts that are not directly supported by the context.
3. If the provided context does NOT contain enough information to answer the question, respond with:
   "I couldn't find enough information in the provided documents to answer this question."
4. Do NOT use external knowledge or pretend to know information outside the provided documents.
5. Start your response immediately with the factual answer. Never use conversational preambles or introductory phrases (e.g., do NOT write "Based on the provided context...", "According to the documents...", or "Here is...").
6. Keep your response direct, structured, and concise so all relevant facts are completely answered without trailing off.
7. Wherever possible, cite source numbers (e.g., [Source 1], [Source 2]) to support your statements.
8. If the question asks for a list or summary of multiple items (e.g., 'all projects', 'all certifications', 'all policies') and only one item is present in the context, explicitly state that only that one item is mentioned in the documents (for example: "Only one project is mentioned in the resume: [Project Name]"), explain its main purpose and key details from the context, and confirm that no other items are listed."""


def build_rag_prompt(query: str, context: str) -> str:
    """Constructs a deterministic, grounded RAG prompt from the user question and retrieved context string."""
    clean_query = query.strip() if query else ""
    clean_context = context.strip() if context else ""

    if not clean_context:
        logger.warning("Empty context passed to build_rag_prompt.")

    prompt = (
        f"{SYSTEM_RAG_INSTRUCTIONS}\n\n"
        f"--- CONTEXT START ---\n"
        f"{clean_context}\n"
        f"--- CONTEXT END ---\n\n"
        f"USER QUESTION:\n{clean_query}\n\n"
        f"GROUNDED ANSWER:"
    )

    return prompt
