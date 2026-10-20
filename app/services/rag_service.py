"""Service module orchestrating end-to-end RAG workflows: ingestion, vector embedding, retrieval, context construction, LangGraph adaptive routing, and LLM answer generation."""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from app.services.ingestion_service import DocumentIngestionService
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService, retrieval_service as default_retrieval_service
from app.services.context_service import ContextBuilder
from app.services.llm_service import LLMService
from app.services.conversation_service import ConversationService, conversation_service as default_conversation_service
from app.services.reranker_service import RerankerService, reranker_service as default_reranker_service
from app.graph.workflow import adaptive_rag_graph, build_adaptive_rag_graph
from app.graph.state import RAGGraphState
from app.core.config import settings
from app.core.logging_config import logger


class RAGService:
    """Façade service orchestrating document processing, vector indexing, retrieval, context building, adaptive routing, and LLM answer generation."""

    def __init__(
        self,
        ingestion_service: Optional[DocumentIngestionService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        retrieval_service: Optional[RetrievalService] = None,
        context_builder: Optional[ContextBuilder] = None,
        llm_service: Optional[LLMService] = None,
        conversation_service: Optional[ConversationService] = None,
        reranker_service: Optional[RerankerService] = None,
        graph=None,
    ) -> None:
        self.ingestion_service = ingestion_service or DocumentIngestionService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.retrieval_service = retrieval_service or default_retrieval_service
        self.context_builder = context_builder or ContextBuilder()
        self.llm_service = llm_service or LLMService()
        self.conversation_service = conversation_service or default_conversation_service
        self.reranker_service = reranker_service or default_reranker_service
        self.graph = graph or adaptive_rag_graph

        logger.info("Initialized RAGService façade with LLM answer generation, LangGraph workflow engine, and CrossEncoder re-ranking")

    def reload_vector_store(self) -> None:
        """Forces vector store reload across the underlying retrieval service."""
        self.retrieval_service.reload_vector_store()

    def index_document(
        self, file_path: Union[str, Path], original_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Processes a PDF document (extract, clean, chunk) and indexes its embeddings into FAISS."""
        path = Path(file_path)
        orig_name = original_filename or path.name

        logger.info(f"Starting document indexing workflow for: '{orig_name}'")

        # 1. Process document using IngestionService
        ingestion_result = self.ingestion_service.process_document(path, original_filename=orig_name)

        # 2. Extract LangChain Document objects from processing step
        pages = self.ingestion_service.load_document(path)
        chunks = self.ingestion_service.split_documents(pages, original_filename=orig_name)

        # 3. Add chunks to FAISS vector store
        if not chunks:
            logger.warning(f"No document chunks available to index for: '{orig_name}'")
            return {
                "filename": ingestion_result.filename,
                "original_filename": orig_name,
                "status": "warning",
                "page_count": ingestion_result.page_count,
                "chunk_count": 0,
                "message": f"No document chunks provided for indexing from '{orig_name}'. The document may be empty or unreadable.",
            }

        index_result = self.retrieval_service.add_documents(chunks, original_filename=orig_name)

        if index_result["status"] == "duplicate":
            indexed_chunks = 0
        elif index_result["status"] in ("indexed", "success"):
            indexed_chunks = index_result.get("added_chunks", len(chunks))
        else:
            indexed_chunks = 0

        return {
            "filename": ingestion_result.filename,
            "original_filename": orig_name,
            "status": index_result["status"],
            "page_count": ingestion_result.page_count,
            "chunk_count": indexed_chunks,
            "message": index_result["message"],
        }

    def retrieve(self, query: str, top_k: int = settings.TOP_K) -> List[Dict[str, Any]]:
        """Executes raw FAISS semantic search against the vector store and returns matching chunk payloads."""
        logger.info(f"RAGService processing raw FAISS retrieval query: '{query}' (top_k={top_k})")
        results_with_score = self.retrieval_service.similarity_search(query=query, top_k=top_k)

        formatted_results = []
        for doc, score in results_with_score:
            meta = doc.metadata or {}
            formatted_results.append(
                {
                    "content": doc.page_content,
                    "source": meta.get("source", ""),
                    "page": meta.get("page", 1),
                    "chunk_id": meta.get("chunk_id", ""),
                    "original_filename": meta.get("original_filename", ""),
                    "score": round(float(score), 4),
                }
            )

        return formatted_results

    def reranked_search(
        self,
        query: str,
        top_k: int = settings.RERANKER_TOP_K,
        candidate_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Executes two-stage retrieval: FAISS candidate search followed by Cross-Encoder re-ranking."""
        if not query or not query.strip():
            raise ValueError("Search query string cannot be empty.")

        clean_query = query.strip()
        cand_k = candidate_k or settings.RERANKER_CANDIDATE_K_SIMPLE
        logger.info(f"RAGService initiating two-stage re-ranked search for query: '{clean_query}' (cand_k={cand_k}, top_k={top_k})")

        candidates = self.retrieval_service.similarity_search(query=clean_query, top_k=cand_k)
        rerank_result = self.reranker_service.rerank(query=clean_query, documents=candidates, top_k=top_k)

        return {
            "query": clean_query,
            "candidate_count": rerank_result["candidate_count"],
            "reranked_count": rerank_result["reranked_count"],
            "reranker_enabled": rerank_result["reranker_enabled"],
            "reranker_model": rerank_result["reranker_model"],
            "reranker_status": rerank_result["status"],
            "results": rerank_result["reranked_documents"],
            "status": "success",
        }

    def retrieve_context(self, query: str, top_k: int = settings.TOP_K) -> Dict[str, Any]:
        """Performs vector retrieval and builds a structured, character-bounded RAG context without calling an LLM."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        clean_query = query.strip()
        logger.info(f"RAGService building retrieval context for query: '{clean_query}' (top_k={top_k})")

        try:
            results_with_score = self.retrieval_service.similarity_search(query=clean_query, top_k=top_k)
        except ValueError as e:
            logger.warning(f"Vector retrieval failed during context build: {str(e)}")
            return {
                "query": clean_query,
                "status": "no_results",
                "context": "",
                "sources": [],
                "retrieved_count": 0,
                "included_count": 0,
            }

        if not results_with_score:
            return {
                "query": clean_query,
                "status": "no_results",
                "context": "",
                "sources": [],
                "retrieved_count": 0,
                "included_count": 0,
            }

        context_payload = self.context_builder.build_context(results_with_score)

        return {
            "query": clean_query,
            "status": "success" if context_payload["retrieved_count"] > 0 else "no_results",
            "context": context_payload["context"],
            "sources": context_payload["sources"],
            "retrieved_count": context_payload["retrieved_count"],
            "included_count": context_payload["included_count"],
        }

    def adaptive_retrieve_context(
        self, query: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes the compiled LangGraph workflow to classify query complexity, dynamically route, retrieve candidates, re-rank, and construct RAG context."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        clean_query = query.strip()
        logger.info(f"RAGService initiating adaptive LangGraph workflow for query: '{clean_query}'")

        initial_state: RAGGraphState = {
            "query": clean_query,
            "session_id": session_id,
            "route": None,
            "route_reason": None,
            "route_confidence": None,
            "top_k": None,
            "candidate_k": None,
            "candidate_documents": None,
            "candidate_count": 0,
            "reranked_documents": None,
            "reranked_count": 0,
            "reranker_enabled": None,
            "reranker_model": None,
            "reranker_status": None,
            "reranker_error": None,
            "retrieved_documents": None,
            "context": None,
            "sources": None,
            "retrieved_count": 0,
            "included_count": 0,
            "status": None,
            "error": None,
            "prompt": None,
            "answer": None,
            "llm_provider": None,
            "llm_model": None,
            "generation_status": None,
            "generation_error": None,
            "generate_answer": False,
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "query": clean_query,
            "route": final_state.get("route", "simple"),
            "route_reason": final_state.get("route_reason", "Default simple retrieval route."),
            "route_confidence": final_state.get("route_confidence", 0.70),
            "top_k": final_state.get("top_k", settings.RERANKER_TOP_K),
            "candidate_count": final_state.get("candidate_count", 0),
            "reranked_count": final_state.get("reranked_count", 0),
            "reranker_enabled": final_state.get("reranker_enabled", settings.RERANKER_ENABLED),
            "reranker_model": final_state.get("reranker_model", settings.RERANKER_MODEL),
            "reranker_status": final_state.get("reranker_status", "disabled"),
            "reranked_documents": final_state.get("reranked_documents", []),
            "status": final_state.get("status", "no_results"),
            "context": final_state.get("context", ""),
            "sources": final_state.get("sources", []),
            "retrieved_count": final_state.get("retrieved_count", 0),
            "included_count": final_state.get("included_count", 0),
            "timings": final_state.get("timings", {}),
        }

    def generate_rag_answer(
        self, query: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes the complete Adaptive RAG LangGraph pipeline to generate a grounded LLM answer with re-ranking and source citations."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        clean_query = query.strip()
        logger.info(f"RAGService initiating end-to-end adaptive RAG answer generation for query: '{clean_query}'")

        initial_state: RAGGraphState = {
            "query": clean_query,
            "session_id": session_id,
            "route": None,
            "route_reason": None,
            "route_confidence": None,
            "top_k": None,
            "candidate_k": None,
            "candidate_documents": None,
            "candidate_count": 0,
            "reranked_documents": None,
            "reranked_count": 0,
            "reranker_enabled": None,
            "reranker_model": None,
            "reranker_status": None,
            "reranker_error": None,
            "retrieved_documents": None,
            "context": None,
            "sources": None,
            "retrieved_count": 0,
            "included_count": 0,
            "status": None,
            "error": None,
            "prompt": None,
            "answer": None,
            "llm_provider": None,
            "llm_model": None,
            "generation_status": None,
            "generation_error": None,
            "generate_answer": True,
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "query": clean_query,
            "route": final_state.get("route", "simple"),
            "route_reason": final_state.get("route_reason", "Default simple route."),
            "route_confidence": final_state.get("route_confidence", 0.70),
            "top_k": final_state.get("top_k", settings.RERANKER_TOP_K),
            "candidate_count": final_state.get("candidate_count", 0),
            "reranked_count": final_state.get("reranked_count", 0),
            "reranker_enabled": final_state.get("reranker_enabled", settings.RERANKER_ENABLED),
            "reranker_model": final_state.get("reranker_model", settings.RERANKER_MODEL),
            "reranker_status": final_state.get("reranker_status", "disabled"),
            "reranked_documents": final_state.get("reranked_documents", []),
            "answer": final_state.get("answer", "I couldn't find enough information in the provided documents to answer this question."),
            "sources": final_state.get("sources", []),
            "retrieved_count": final_state.get("retrieved_count", 0),
            "included_count": final_state.get("included_count", 0),
            "status": final_state.get("status", "no_results"),
            "llm_provider": final_state.get("llm_provider", settings.LLM_PROVIDER),
            "llm_model": final_state.get("llm_model", settings.LLM_MODEL),
            "generation_status": final_state.get("generation_status", "completed"),
            "generation_error": final_state.get("generation_error"),
            "llm_metrics": final_state.get("llm_metrics", {}),
            "timings": final_state.get("timings", {}),
        }

    def conversational_rag_chat(
        self, query: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes multi-turn conversational RAG pipeline with query rewriting, re-ranking, and session history tracking."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        clean_query = query.strip()
        logger.info(
            f"RAGService initiating conversational RAG chat for query: '{clean_query}' (session_id='{session_id}')"
        )

        initial_state: RAGGraphState = {
            "query": clean_query,
            "session_id": session_id,
            "conversation_id": session_id,
            "route": None,
            "route_reason": None,
            "route_confidence": None,
            "top_k": None,
            "candidate_k": None,
            "candidate_documents": None,
            "candidate_count": 0,
            "reranked_documents": None,
            "reranked_count": 0,
            "reranker_enabled": None,
            "reranker_model": None,
            "reranker_status": None,
            "reranker_error": None,
            "conversation_history": None,
            "standalone_query": None,
            "was_rewritten": None,
            "retrieved_documents": None,
            "context": None,
            "sources": None,
            "retrieved_count": 0,
            "included_count": 0,
            "status": None,
            "error": None,
            "prompt": None,
            "answer": None,
            "llm_provider": None,
            "llm_model": None,
            "generation_status": None,
            "generation_error": None,
            "generate_answer": True,
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "session_id": final_state.get("session_id") or final_state.get("conversation_id") or session_id,
            "query": clean_query,
            "standalone_query": final_state.get("standalone_query", clean_query),
            "was_rewritten": final_state.get("was_rewritten", False),
            "answer": final_state.get(
                "answer",
                "I couldn't find enough information in the provided documents to answer this question.",
            ),
            "route": final_state.get("route", "simple"),
            "route_reason": final_state.get("route_reason", "Default simple route."),
            "route_confidence": final_state.get("route_confidence", 0.70),
            "top_k": final_state.get("top_k", settings.RERANKER_TOP_K),
            "candidate_count": final_state.get("candidate_count", 0),
            "reranked_count": final_state.get("reranked_count", 0),
            "reranker_enabled": final_state.get("reranker_enabled", settings.RERANKER_ENABLED),
            "reranker_model": final_state.get("reranker_model", settings.RERANKER_MODEL),
            "reranker_status": final_state.get("reranker_status", "disabled"),
            "reranked_documents": final_state.get("reranked_documents", []),
            "sources": final_state.get("sources", []),
            "retrieved_count": final_state.get("retrieved_count", 0),
            "included_count": final_state.get("included_count", 0),
            "status": final_state.get("status", "no_results"),
            "llm_provider": final_state.get("llm_provider", settings.LLM_PROVIDER),
            "llm_model": final_state.get("llm_model", settings.LLM_MODEL),
            "generation_status": final_state.get("generation_status", "completed"),
            "generation_error": final_state.get("generation_error"),
            "llm_metrics": final_state.get("llm_metrics", {}),
            "timings": final_state.get("timings", {}),
        }

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves history for a conversation session."""
        messages = self.conversation_service.get_history(session_id)
        return [msg.model_dump() for msg in messages]

    def clear_conversation_session(self, session_id: str) -> bool:
        """Clears conversation session history."""
        return self.conversation_service.clear_session(session_id)


rag_service = RAGService()


