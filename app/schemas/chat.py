"""Schemas for chat, semantic search, RAG context retrieval, adaptive routing, and LLM answer generation endpoints."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Schema for chat / RAG query request."""

    query: str = Field(..., min_length=1, max_length=2000, description="User's query or prompt")
    session_id: Optional[str] = Field(default=None, description="Session identifier for multi-turn chat history")
    top_k: Optional[int] = Field(default=4, ge=1, le=20, description="Number of top document chunks to retrieve")


class SourceDocument(BaseModel):
    """Schema representing source documents referenced in answer generation."""

    source_id: str = Field(..., description="Unique document chunk identifier")
    content: str = Field(..., description="Excerpt content from the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the source chunk")


class ChatResponse(BaseModel):
    """Schema for chat / RAG query response."""

    answer: str = Field(..., description="Generated answer from the system")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    sources: List[SourceDocument] = Field(default_factory=list, description="Retrieved source document references")
    status: str = Field(default="success", description="Status of the query operation")


class SearchRequest(BaseModel):
    """Schema for semantic search query request."""

    query: str = Field(..., min_length=1, max_length=2000, description="Search query string")
    top_k: Optional[int] = Field(default=4, ge=1, le=20, description="Number of top matching document chunks to return")


class SearchResult(BaseModel):
    """Schema representing an individual retrieved document chunk result."""

    content: str = Field(..., description="Text content of retrieved document chunk")
    source: str = Field(..., description="File path or source identifier")
    page: int = Field(..., description="Page number of origin")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    original_filename: str = Field(..., description="Original filename of the document")
    score: Optional[float] = Field(default=None, description="Similarity score or distance metric")


class SearchResponse(BaseModel):
    """Schema for semantic search response."""

    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(default_factory=list, description="Top matching document chunks")
    status: str = Field(default="success", description="Status of the search operation")


class RAGContextRequest(BaseModel):
    """Schema for RAG context construction request."""

    query: str = Field(..., min_length=1, max_length=2000, description="User query for context retrieval")
    top_k: Optional[int] = Field(default=4, ge=1, le=20, description="Number of top document chunks to retrieve")


class RetrievedDocument(BaseModel):
    """Schema representing a retrieved document chunk with score and metadata."""

    content: str = Field(..., description="Text content of the retrieved chunk")
    source: str = Field(..., description="File path or source location")
    page: int = Field(..., description="Page number")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    original_filename: str = Field(..., description="Original document filename")
    score: Optional[float] = Field(default=None, description="Relevance distance score")


class SourceReference(BaseModel):
    """Schema for deduplicated source metadata reference."""

    source: str = Field(..., description="File path or source location")
    page: int = Field(..., description="Page number of chunk origin")
    chunk_id: str = Field(..., description="Chunk ID")
    original_filename: str = Field(..., description="Original document filename")


class RAGContextResponse(BaseModel):
    """Schema for constructed RAG context API response."""

    query: str = Field(..., description="Original user query")
    status: str = Field(..., description="Status of retrieval and context construction (success, no_results)")
    context: str = Field(..., description="Constructed deterministic RAG context string")
    sources: List[SourceReference] = Field(default_factory=list, description="Deduplicated list of source reference metadata")
    retrieved_count: int = Field(..., description="Total number of chunks retrieved from FAISS")
    included_count: int = Field(..., description="Number of chunks included within maximum context size limit")


class AdaptiveRAGRequest(BaseModel):
    """Schema for adaptive RAG context routing request."""

    query: str = Field(..., min_length=1, max_length=2000, description="User query for adaptive routing")
    session_id: Optional[str] = Field(default=None, description="Optional session identifier")


class AdaptiveRAGResponse(BaseModel):
    """Schema for adaptive RAG context routing API response."""

    query: str = Field(..., description="Original user query")
    route: str = Field(..., description="Assigned routing branch ('simple' or 'complex')")
    route_reason: str = Field(..., description="Heuristic explanation for routing classification decision")
    route_confidence: float = Field(..., description="Heuristic confidence score (0.5 - 0.95)")
    top_k: int = Field(..., description="Retrieval depth allocated based on route decision")
    candidate_count: Optional[int] = Field(default=0, description="Total candidate chunks retrieved before re-ranking")
    reranked_count: Optional[int] = Field(default=0, description="Chunks remaining after re-ranking")
    reranker_enabled: Optional[bool] = Field(default=True, description="Flag indicating whether re-ranking was enabled")
    reranker_model: Optional[str] = Field(default="BAAI/bge-reranker-base", description="Re-ranker model name")
    reranker_status: Optional[str] = Field(default="success", description="Status of re-ranking operation")
    status: str = Field(..., description="Status of processing ('success' or 'no_results')")
    context: str = Field(..., description="Constructed character-bounded RAG context string")
    sources: List[SourceReference] = Field(default_factory=list, description="Deduplicated list of source reference metadata")
    retrieved_count: int = Field(..., description="Total number of document chunks retrieved")
    included_count: int = Field(..., description="Number of document chunks included within context size limit")
    timings: Optional[Dict[str, float]] = Field(default=None, description="Diagnostic execution latency in milliseconds per stage")


class RAGAnswerRequest(BaseModel):
    """Schema for complete RAG answer generation request."""

    query: str = Field(..., min_length=1, max_length=2000, description="User question for RAG answer generation")
    session_id: Optional[str] = Field(default=None, description="Optional session identifier")


class RAGAnswerResponse(BaseModel):
    """Schema for generated RAG answer API response."""

    query: str = Field(..., description="Original user query")
    route: str = Field(..., description="Assigned adaptive routing path")
    route_reason: str = Field(..., description="Heuristic explanation for routing decision")
    route_confidence: float = Field(..., description="Routing decision confidence score")
    top_k: int = Field(..., description="Retrieval depth used")
    candidate_count: Optional[int] = Field(default=0, description="Total candidate chunks retrieved before re-ranking")
    reranked_count: Optional[int] = Field(default=0, description="Chunks remaining after re-ranking")
    reranker_enabled: Optional[bool] = Field(default=True, description="Flag indicating whether re-ranking was enabled")
    reranker_model: Optional[str] = Field(default="BAAI/bge-reranker-base", description="Re-ranker model name")
    reranker_status: Optional[str] = Field(default="success", description="Status of re-ranking operation")
    answer: str = Field(..., description="Generated natural language answer from LLM grounded in context")
    sources: List[SourceReference] = Field(default_factory=list, description="Deduplicated source citations supporting the answer")
    retrieved_count: int = Field(..., description="Total document chunks retrieved from vector store")
    included_count: int = Field(..., description="Number of document chunks included in context")
    status: str = Field(..., description="Status of overall RAG execution ('success' or 'no_results')")
    llm_provider: str = Field(..., description="LLM provider name (e.g. ollama)")
    llm_model: str = Field(..., description="LLM model name (e.g. llama3)")
    generation_status: Optional[str] = Field(default="success", description="Status of LLM generation step")
    generation_error: Optional[str] = Field(default=None, description="Detailed error message if LLM generation failed")
    timings: Optional[Dict[str, float]] = Field(default=None, description="Diagnostic execution latency in milliseconds per stage")


class ConversationalChatRequest(BaseModel):
    """Schema for multi-turn conversational RAG request."""

    query: str = Field(..., min_length=1, max_length=2000, description="User question or follow-up prompt")
    session_id: Optional[str] = Field(default=None, description="Optional session identifier for history tracking")


class ConversationalChatResponse(BaseModel):
    """Schema for multi-turn conversational RAG response."""

    session_id: str = Field(..., description="Active session identifier")
    query: str = Field(..., description="Original user query")
    standalone_query: str = Field(..., description="Reformulated standalone search query used for retrieval")
    was_rewritten: bool = Field(..., description="Flag indicating whether query was rewritten using history")
    answer: str = Field(..., description="Generated grounded natural language answer")
    route: str = Field(..., description="Assigned adaptive routing path")
    route_reason: str = Field(..., description="Explanation for routing decision")
    route_confidence: float = Field(..., description="Routing confidence score")
    top_k: int = Field(..., description="Retrieval depth used")
    candidate_count: Optional[int] = Field(default=0, description="Total candidate chunks retrieved before re-ranking")
    reranked_count: Optional[int] = Field(default=0, description="Chunks remaining after re-ranking")
    reranker_enabled: Optional[bool] = Field(default=True, description="Flag indicating whether re-ranking was enabled")
    reranker_model: Optional[str] = Field(default="BAAI/bge-reranker-base", description="Re-ranker model name")
    reranker_status: Optional[str] = Field(default="success", description="Status of re-ranking operation")
    sources: List[SourceReference] = Field(default_factory=list, description="Deduplicated source references")
    retrieved_count: int = Field(..., description="Total document chunks retrieved")
    included_count: int = Field(..., description="Document chunks included in context")
    status: str = Field(..., description="Operation status ('success' or 'no_results')")
    llm_provider: str = Field(..., description="LLM provider name")
    llm_model: str = Field(..., description="LLM model name")
    generation_status: Optional[str] = Field(default="success", description="Status of LLM generation step")
    generation_error: Optional[str] = Field(default=None, description="Detailed error message if LLM generation failed")
    timings: Optional[Dict[str, float]] = Field(default=None, description="Diagnostic execution latency in milliseconds per stage")


class RerankedSearchRequest(BaseModel):
    """Schema for two-stage re-ranked search request."""

    query: str = Field(..., min_length=1, max_length=2000, description="Search query string")
    top_k: Optional[int] = Field(default=4, ge=1, le=20, description="Final top_k re-ranked document chunks to return")
    candidate_k: Optional[int] = Field(default=8, ge=1, le=50, description="FAISS candidate pool depth before re-ranking")


class RerankedSearchResult(BaseModel):
    """Schema representing an individual re-ranked document chunk result."""

    rank: int = Field(..., description="Final 1-based rank position after re-ranking")
    rerank_score: Optional[float] = Field(default=None, description="Cross-encoder relevance score")
    retrieval_score: float = Field(..., description="Original FAISS distance or similarity score")
    content: str = Field(..., description="Text content of the document chunk")
    source: str = Field(..., description="File path or source location")
    page: int = Field(..., description="Page number of origin")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    original_filename: str = Field(..., description="Original document filename")


class RerankedSearchResponse(BaseModel):
    """Schema for two-stage re-ranked search API response."""

    query: str = Field(..., description="Original search query")
    candidate_count: int = Field(..., description="Number of candidate chunks retrieved from FAISS")
    reranked_count: int = Field(..., description="Number of chunks returned after re-ranking")
    reranker_enabled: bool = Field(..., description="Flag indicating whether re-ranking was active")
    reranker_model: str = Field(..., description="Name of the Cross-Encoder model used")
    reranker_status: str = Field(..., description="Status of re-ranking operation ('success', 'fallback', or 'disabled')")
    results: List[RerankedSearchResult] = Field(default_factory=list, description="Ordered re-ranked document results")
    status: str = Field(default="success", description="Overall API operation status")


class ConversationMessageSchema(BaseModel):
    """Schema representing an individual message in session history."""

    role: str = Field(..., description="Role of the sender ('user' or 'assistant')")
    content: str = Field(..., description="Text content of the message")
    timestamp: str = Field(..., description="Timestamp of message creation")


class ConversationHistoryResponse(BaseModel):
    """Schema for conversation session history endpoint response."""

    session_id: str = Field(..., description="Session identifier")
    messages: List[ConversationMessageSchema] = Field(default_factory=list, description="Ordered list of historical messages")
    message_count: int = Field(..., description="Total count of messages in session history")


class ClearSessionResponse(BaseModel):
    """Schema for clear conversation session endpoint response."""

    session_id: str = Field(..., description="Session identifier")
    cleared: bool = Field(..., description="Boolean status indicating whether session was cleared")
    message: str = Field(..., description="Status message detailing operation outcome")


