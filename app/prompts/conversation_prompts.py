"""Prompts for conversational query rewriting and multi-turn grounded RAG answer generation."""

from typing import List
from app.models.message import ConversationMessage
from app.core.logging_config import logger


QUERY_REWRITE_SYSTEM_PROMPT = """You are an expert search query reformulator. Your task is to reformulate the current user question into a clear, self-contained standalone search query that resolves any ambiguous pronouns or references (such as 'it', 'this', 'that', 'they', 'the policy') using the provided conversation history.

RULES:
1. Preserve the user's core intent exactly.
2. If the current question is already standalone and clear, return it UNCHANGED.
3. Do NOT answer the question.
4. Do NOT invent new topics.
5. Return ONLY the standalone search query without any explanation, markdown, or commentary."""


def build_query_rewrite_prompt(current_query: str, history: List[ConversationMessage]) -> str:
    """Constructs prompt for reformulating a follow-up user query into a standalone search query."""
    formatted_history = []
    for msg in history:
        role_label = "USER" if msg.role == "user" else "ASSISTANT"
        formatted_history.append(f"{role_label}: {msg.content}")

    history_str = "\n".join(formatted_history) if formatted_history else "(No prior conversation history)"

    prompt = (
        f"{QUERY_REWRITE_SYSTEM_PROMPT}\n\n"
        f"--- CONVERSATION HISTORY ---\n"
        f"{history_str}\n"
        f"--- END HISTORY ---\n\n"
        f"CURRENT USER QUESTION:\n{current_query.strip()}\n\n"
        f"STANDALONE SEARCH QUERY:"
    )
    return prompt


CONVERSATIONAL_RAG_SYSTEM_PROMPT = """You are a helpful enterprise document assistant. Answer the user's current question using ONLY the provided document context below. Use the conversation history to understand the user's intent and follow-up references.

CRITICAL GROUNDING RULES:
1. Answer using ONLY the facts explicitly stated in the provided document context.
2. Use conversation history ONLY to understand references (e.g., pronouns), NOT as factual evidence if it conflicts with documents.
3. If the provided document context does NOT contain enough information to answer the question, respond with:
   "I couldn't find enough information in the provided documents to answer this question."
4. Do NOT fabricate or assume facts not supported by the context documents.
5. Maintain conversational continuity while remaining strictly grounded. State facts directly without conversational preambles (e.g., do not begin with "Based on the provided context..." or "According to the documents...").
6. Cite source numbers (e.g., [Source 1], [Source 2]) when possible.
7. If the question asks for a list or summary of multiple items (e.g., 'all projects', 'all certifications', 'all policies') and only one item is present in the context, explicitly state that only that one item is mentioned in the documents (for example: "Only one project is mentioned in the resume: [Project Name]"), explain its main purpose and key details from the context, and confirm that no other items are listed."""


def build_conversational_rag_prompt(
    query: str, context: str, history: List[ConversationMessage]
) -> str:
    """Constructs prompt for generating grounded answer with conversation history and document context."""
    formatted_history = []
    for msg in history:
        role_label = "USER" if msg.role == "user" else "ASSISTANT"
        formatted_history.append(f"{role_label}: {msg.content}")

    history_str = "\n".join(formatted_history) if formatted_history else "(No prior conversation history)"

    prompt = (
        f"{CONVERSATIONAL_RAG_SYSTEM_PROMPT}\n\n"
        f"--- CONVERSATION HISTORY ---\n"
        f"{history_str}\n"
        f"--- END CONVERSATION HISTORY ---\n\n"
        f"--- RETRIEVED DOCUMENT CONTEXT START ---\n"
        f"{context.strip()}\n"
        f"--- RETRIEVED DOCUMENT CONTEXT END ---\n\n"
        f"CURRENT USER QUESTION:\n{query.strip()}\n\n"
        f"GROUNDED ANSWER:"
    )
    return prompt
