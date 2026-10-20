"""Chat, search, retrieval, adaptive routing, and LLM answer generation endpoints router."""

from fastapi import APIRouter, HTTPException, status, Depends
from starlette.concurrency import run_in_threadpool
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SourceDocument,
    SearchRequest,
    SearchResponse,
    SearchResult,
    RAGContextRequest,
    RAGContextResponse,
    SourceReference,
    AdaptiveRAGRequest,
    AdaptiveRAGResponse,
    RAGAnswerRequest,
    RAGAnswerResponse,
    ConversationalChatRequest,
    ConversationalChatResponse,
    ConversationMessageSchema,
    ConversationHistoryResponse,
    ClearSessionResponse,
    RerankedSearchRequest,
    RerankedSearchResponse,
    RerankedSearchResult,
)
from app.services.rag_service import RAGService, rag_service
from app.core.security import verify_api_key
from app.core.logging_config import logger

router = APIRouter(tags=["Chat, Search, Context & LLM Generation"])
rag_service = rag_service


@router.post("/api/v1/chat/query", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_query(request: ChatRequest) -> ChatResponse:
    """Endpoint for handling RAG chat queries."""
    logger.info(f"Received chat query request: query='{request.query}' (session_id={request.session_id})")

    try:
        rag_res = await run_in_threadpool(
            rag_service.generate_rag_answer,
            query=request.query,
            session_id=request.session_id,
        )

        sources = [
            SourceDocument(
                source_id=s.chunk_id,
                content=f"Page {s.page} from {s.original_filename}",
                metadata={
                    "source": s.source,
                    "page": s.page,
                    "original_filename": s.original_filename,
                },
            )
            for s in rag_res.get("sources", [])
        ]

        return ChatResponse(
            answer=rag_res["answer"],
            session_id=request.session_id,
            sources=sources,
            status=rag_res["status"],
        )

    except Exception as e:
        logger.error(f"Error in chat_query endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat query: {str(e)}",
        )


@router.post("/api/v1/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def semantic_search(request: SearchRequest) -> SearchResponse:
    """Performs raw vector similarity search against the FAISS vector store and returns matching document chunks."""
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query string cannot be empty.",
        )

    logger.info(f"Received semantic search request: query='{request.query}' (top_k={request.top_k})")

    try:
        results = await run_in_threadpool(
            rag_service.retrieve,
            query=request.query,
            top_k=request.top_k or 4,
        )

        search_results = [
            SearchResult(
                content=res["content"],
                source=res["source"],
                page=res["page"],
                chunk_id=res["chunk_id"],
                original_filename=res["original_filename"],
                score=res.get("score"),
            )
            for res in results
        ]

        return SearchResponse(
            query=request.query,
            results=search_results,
            status="success",
        )

    except ValueError as e:
        logger.warning(f"Semantic search error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error during semantic search for query '{request.query}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while performing semantic search.",
        )


@router.post("/api/v1/search/reranked", response_model=RerankedSearchResponse, status_code=status.HTTP_200_OK)
async def reranked_search(request: RerankedSearchRequest) -> RerankedSearchResponse:
    """Executes two-stage retrieval: FAISS candidate search followed by Cross-Encoder re-ranking."""
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query string cannot be empty.",
        )

    logger.info(f"Received re-ranked search request: query='{request.query}' (candidate_k={request.candidate_k}, top_k={request.top_k})")

    try:
        res = await run_in_threadpool(
            rag_service.reranked_search,
            query=request.query,
            top_k=request.top_k or 4,
            candidate_k=request.candidate_k or 8,
        )

        results = [
            RerankedSearchResult(
                rank=r["rank"],
                rerank_score=r.get("rerank_score"),
                retrieval_score=r["retrieval_score"],
                content=r["content"],
                source=r["source"],
                page=r["page"],
                chunk_id=r["chunk_id"],
                original_filename=r["original_filename"],
            )
            for r in res.get("results", [])
        ]

        return RerankedSearchResponse(
            query=res["query"],
            candidate_count=res["candidate_count"],
            reranked_count=res["reranked_count"],
            reranker_enabled=res["reranker_enabled"],
            reranker_model=res["reranker_model"],
            reranker_status=res["reranker_status"],
            results=results,
            status="success",
        )

    except ValueError as e:
        logger.warning(f"Re-ranked search error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error during re-ranked search for query '{request.query}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while performing re-ranked search.",
        )


@router.post("/api/v1/rag/context", response_model=RAGContextResponse, status_code=status.HTTP_200_OK)
async def get_rag_context(request: RAGContextRequest) -> RAGContextResponse:
    """Retrieves matching vector document chunks and constructs a structured character-bounded RAG context."""
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    logger.info(f"Received RAG context request: query='{request.query}' (top_k={request.top_k})")

    try:
        context_result = await run_in_threadpool(
            rag_service.retrieve_context,
            query=request.query,
            top_k=request.top_k or 4,
        )

        sources = [
            SourceReference(
                source=src["source"],
                page=src["page"],
                chunk_id=src["chunk_id"],
                original_filename=src["original_filename"],
            )
            for src in context_result.get("sources", [])
        ]

        return RAGContextResponse(
            query=context_result["query"],
            status=context_result["status"],
            context=context_result["context"],
            sources=sources,
            retrieved_count=context_result["retrieved_count"],
            included_count=context_result["included_count"],
        )

    except ValueError as e:
        logger.warning(f"RAG context retrieval validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error constructing RAG context for query '{request.query}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while constructing the RAG context.",
        )


@router.post("/api/v1/rag/adaptive-context", response_model=AdaptiveRAGResponse, status_code=status.HTTP_200_OK)
async def get_adaptive_rag_context(request: AdaptiveRAGRequest) -> AdaptiveRAGResponse:
    """Executes stateful LangGraph adaptive routing and re-ranking, dynamically retrieving candidates and formatting RAG context."""
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    logger.info(f"Received Adaptive RAG context request: query='{request.query}' (session_id={request.session_id})")

    try:
        adaptive_result = await run_in_threadpool(
            rag_service.adaptive_retrieve_context,
            query=request.query,
            session_id=request.session_id,
        )

        sources = [
            SourceReference(
                source=src["source"],
                page=src["page"],
                chunk_id=src["chunk_id"],
                original_filename=src["original_filename"],
            )
            for src in adaptive_result.get("sources", [])
        ]

        return AdaptiveRAGResponse(
            query=adaptive_result["query"],
            route=adaptive_result["route"],
            route_reason=adaptive_result["route_reason"],
            route_confidence=adaptive_result["route_confidence"],
            top_k=adaptive_result["top_k"],
            candidate_count=adaptive_result.get("candidate_count", 0),
            reranked_count=adaptive_result.get("reranked_count", 0),
            reranker_enabled=adaptive_result.get("reranker_enabled", True),
            reranker_model=adaptive_result.get("reranker_model", "BAAI/bge-reranker-base"),
            reranker_status=adaptive_result.get("reranker_status", "disabled"),
            status=adaptive_result["status"],
            context=adaptive_result["context"],
            sources=sources,
            retrieved_count=adaptive_result["retrieved_count"],
            included_count=adaptive_result["included_count"],
            timings=adaptive_result.get("timings"),
        )

    except ValueError as e:
        logger.warning(f"Adaptive RAG context validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error in adaptive RAG context workflow for query '{request.query}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during adaptive RAG workflow execution.",
        )


@router.post("/api/v1/rag/answer", response_model=RAGAnswerResponse, status_code=status.HTTP_200_OK)
async def generate_rag_answer(request: RAGAnswerRequest) -> RAGAnswerResponse:
    """Executes end-to-end Adaptive RAG pipeline: Query Router -> FAISS Retrieval -> Cross-Encoder Re-Ranking -> Context Construction -> Custom Prompt -> LLM Answer Generation."""
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    logger.info(f"Received RAG answer generation request: query='{request.query}' (session_id={request.session_id})")

    try:
        rag_result = await run_in_threadpool(
            rag_service.generate_rag_answer,
            query=request.query,
            session_id=request.session_id,
        )

        sources = [
            SourceReference(
                source=src["source"],
                page=src["page"],
                chunk_id=src["chunk_id"],
                original_filename=src["original_filename"],
            )
            for src in rag_result.get("sources", [])
        ]

        return RAGAnswerResponse(
            query=rag_result["query"],
            route=rag_result["route"],
            route_reason=rag_result["route_reason"],
            route_confidence=rag_result["route_confidence"],
            top_k=rag_result["top_k"],
            candidate_count=rag_result.get("candidate_count", 0),
            reranked_count=rag_result.get("reranked_count", 0),
            reranker_enabled=rag_result.get("reranker_enabled", True),
            reranker_model=rag_result.get("reranker_model", "BAAI/bge-reranker-base"),
            reranker_status=rag_result.get("reranker_status", "disabled"),
            answer=rag_result["answer"],
            sources=sources,
            retrieved_count=rag_result["retrieved_count"],
            included_count=rag_result["included_count"],
            status=rag_result["status"],
            llm_provider=rag_result["llm_provider"],
            llm_model=rag_result["llm_model"],
            generation_status=rag_result.get("generation_status", "success"),
            generation_error=rag_result.get("generation_error"),
            timings=rag_result.get("timings"),
        )

    except ValueError as e:
        logger.warning(f"RAG answer validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error during RAG answer generation for query '{request.query}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during RAG answer generation.",
        )


@router.post("/api/v1/rag/chat", response_model=ConversationalChatResponse, status_code=status.HTTP_200_OK)
async def conversational_chat(request: ConversationalChatRequest) -> ConversationalChatResponse:
    """Executes multi-turn conversational RAG pipeline with query rewriting, FAISS candidate retrieval, Cross-Encoder re-ranking, and session history tracking."""
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    logger.info(f"Received conversational chat request: query='{request.query}' (session_id={request.session_id})")

    try:
        chat_res = await run_in_threadpool(
            rag_service.conversational_rag_chat,
            query=request.query,
            session_id=request.session_id,
        )

        sources = [
            SourceReference(
                source=src["source"],
                page=src["page"],
                chunk_id=src["chunk_id"],
                original_filename=src["original_filename"],
            )
            for src in chat_res.get("sources", [])
        ]

        return ConversationalChatResponse(
            session_id=chat_res["session_id"],
            query=chat_res["query"],
            standalone_query=chat_res["standalone_query"],
            was_rewritten=chat_res["was_rewritten"],
            answer=chat_res["answer"],
            route=chat_res["route"],
            route_reason=chat_res["route_reason"],
            route_confidence=chat_res["route_confidence"],
            top_k=chat_res["top_k"],
            candidate_count=chat_res.get("candidate_count", 0),
            reranked_count=chat_res.get("reranked_count", 0),
            reranker_enabled=chat_res.get("reranker_enabled", True),
            reranker_model=chat_res.get("reranker_model", "BAAI/bge-reranker-base"),
            reranker_status=chat_res.get("reranker_status", "disabled"),
            sources=sources,
            retrieved_count=chat_res["retrieved_count"],
            included_count=chat_res["included_count"],
            status=chat_res["status"],
            llm_provider=chat_res["llm_provider"],
            llm_model=chat_res["llm_model"],
            generation_status=chat_res.get("generation_status", "success"),
            generation_error=chat_res.get("generation_error"),
            timings=chat_res.get("timings"),
        )

    except ValueError as e:
        logger.warning(f"Conversational chat validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error during conversational chat for query '{request.query}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during conversational RAG execution.",
        )


@router.get("/api/v1/rag/chat/{session_id}/history", response_model=ConversationHistoryResponse, status_code=status.HTTP_200_OK)
async def get_chat_history(session_id: str) -> ConversationHistoryResponse:
    """Retrieves session history for a specific conversation session ID."""
    if not session_id or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID parameter cannot be empty.",
        )

    clean_id = session_id.strip()
    logger.info(f"Received request to fetch history for session: '{clean_id}'")

    try:
        history_raw = await run_in_threadpool(rag_service.get_conversation_history, clean_id)
        messages = [
            ConversationMessageSchema(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg["timestamp"],
            )
            for msg in history_raw
        ]

        return ConversationHistoryResponse(
            session_id=clean_id,
            messages=messages,
            message_count=len(messages),
        )

    except Exception as e:
        logger.error(f"Error fetching chat history for session '{clean_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation session history.",
        )


@router.delete(
    "/api/v1/rag/chat/{session_id}",
    response_model=ClearSessionResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def clear_chat_history(session_id: str) -> ClearSessionResponse:
    """Clears history for a specific conversation session ID."""
    if not session_id or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID parameter cannot be empty.",
        )

    clean_id = session_id.strip()
    logger.info(f"Received request to clear session history for: '{clean_id}'")

    try:
        cleared = await run_in_threadpool(rag_service.clear_conversation_session, clean_id)
        msg = f"Session '{clean_id}' successfully cleared." if cleared else f"Session '{clean_id}' was not found or already empty."

        return ClearSessionResponse(
            session_id=clean_id,
            cleared=cleared,
            message=msg,
        )

    except Exception as e:
        logger.error(f"Error clearing chat history for session '{clean_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear conversation session history.",
        )


