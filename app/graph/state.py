"""State definition for LangGraph adaptive RAG workflow execution."""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict


class RAGGraphState(TypedDict):
    """Typed dictionary representing the state flowing through LangGraph nodes."""

    # Query & Classification Attributes
    query: str
    session_id: Optional[str]
    route: Optional[str]
    route_reason: Optional[str]
    route_confidence: Optional[float]
    top_k: Optional[int]

    # Conversational & Session Attributes
    conversation_id: Optional[str]
    conversation_history: Optional[List[Dict[str, Any]]]
    conversation_context: Optional[str]
    standalone_query: Optional[str]
    was_rewritten: Optional[bool]

    # Retrieval & Context Attributes
    candidate_k: Optional[int]
    candidate_documents: Optional[List[Dict[str, Any]]]
    candidate_count: Optional[int]
    reranked_documents: Optional[List[Dict[str, Any]]]
    reranked_count: Optional[int]
    reranker_enabled: Optional[bool]
    reranker_model: Optional[str]
    reranker_status: Optional[str]
    reranker_error: Optional[str]
    retrieved_documents: Optional[List[Dict[str, Any]]]
    context: Optional[str]
    sources: Optional[List[Dict[str, Any]]]
    retrieved_count: Optional[int]
    included_count: Optional[int]
    status: Optional[str]
    error: Optional[str]


    # Generation & LLM Attributes
    prompt: Optional[str]
    answer: Optional[str]
    llm_provider: Optional[str]
    llm_model: Optional[str]
    generation_status: Optional[str]
    generation_error: Optional[str]
    generate_answer: Optional[bool]
    llm_metrics: Optional[Dict[str, Any]]

    # Operational Diagnostics & Profiling
    timings: Optional[Dict[str, Any]]

