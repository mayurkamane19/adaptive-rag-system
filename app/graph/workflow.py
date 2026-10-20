"""LangGraph workflow definition for stateful Adaptive RAG execution with LLM answer generation."""

from typing import Dict, Any, Optional
import time
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from app.graph.state import RAGGraphState
from app.services.query_router import QueryRouter
from app.services.retrieval_service import RetrievalService, retrieval_service
from app.services.context_service import ContextBuilder
from app.services.llm_service import LLMService, LLMServiceError
from app.services.conversation_service import ConversationService, conversation_service
from app.services.query_rewriter import QueryRewriter
from app.services.reranker_service import RerankerService, reranker_service
from app.prompts.rag_prompts import build_rag_prompt
from app.prompts.conversation_prompts import build_conversational_rag_prompt
from app.core.config import settings
from app.core.logging_config import logger


# Module-level service instances for efficiency
query_router = QueryRouter()
retrieval_service = retrieval_service
context_builder = ContextBuilder()
llm_service = LLMService()
query_rewriter = QueryRewriter(llm_service=llm_service)


def load_conversation_node(state: RAGGraphState) -> RAGGraphState:
    """Node: Loads recent conversation history from session store if session_id is provided."""
    t0 = time.perf_counter()
    if state.get("timings") is None:
        state["timings"] = {}

    session_id = state.get("session_id") or state.get("conversation_id")
    if session_id:
        clean_id = conversation_service.create_session(session_id)
        history_msgs = conversation_service.get_history(clean_id)
        state["conversation_id"] = clean_id
        state["session_id"] = clean_id
        state["conversation_history"] = history_msgs
        logger.info(f"[Graph Node: load_conversation] Loaded {len(history_msgs)} history message(s) for session '{clean_id}'")
    else:
        state["conversation_history"] = []
        state["conversation_id"] = None

    state["timings"]["load_conversation_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return state


def rewrite_query_node(state: RAGGraphState) -> RAGGraphState:
    """Node: Reformulates follow-up user query into standalone search query using history."""
    t0 = time.perf_counter()
    if state.get("timings") is None:
        state["timings"] = {}

    query = state.get("query", "").strip()
    history = state.get("conversation_history") or []

    if history:
        standalone, rewritten = query_rewriter.rewrite_query(query, history)
        state["standalone_query"] = standalone
        state["was_rewritten"] = rewritten
    else:
        state["standalone_query"] = query
        state["was_rewritten"] = False

    logger.info(
        f"[Graph Node: rewrite_query] standalone_query='{state['standalone_query']}', was_rewritten={state['was_rewritten']}"
    )
    state["timings"]["rewrite_query_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return state


def analyze_query_node(state: RAGGraphState) -> RAGGraphState:
    """Node: Validates and normalizes input query in workflow state."""
    t0 = time.perf_counter()
    if state.get("timings") is None:
        state["timings"] = {}

    raw_query = state.get("query", "")
    clean_query = raw_query.strip() if raw_query else ""
    logger.info(f"[Graph Node: analyze_query] Query: '{clean_query}'")

    state["query"] = clean_query
    if not state.get("standalone_query"):
        state["standalone_query"] = clean_query
    state["status"] = "processing"
    state["timings"]["analyze_query_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return state


def route_query_node(state: RAGGraphState) -> RAGGraphState:
    """Node: Evaluates query complexity and assigns target retrieval branch, candidate pool size, and top_k depth."""
    t0 = time.perf_counter()
    if state.get("timings") is None:
        state["timings"] = {}

    query_to_route = state.get("standalone_query") or state.get("query", "")
    decision = query_router.classify_query(query_to_route)

    state["route"] = decision.route
    state["route_reason"] = decision.reason
    state["route_confidence"] = decision.confidence

    if decision.route == "complex":
        state["candidate_k"] = settings.RERANKER_CANDIDATE_K_COMPLEX
        state["top_k"] = settings.RERANKER_TOP_K_COMPLEX
    else:
        state["candidate_k"] = settings.RERANKER_CANDIDATE_K_SIMPLE
        state["top_k"] = settings.RERANKER_TOP_K

    logger.info(
        f"[Graph Node: route_query] Assigned route='{state['route']}', "
        f"candidate_k={state['candidate_k']}, top_k={state['top_k']}, confidence={state['route_confidence']}"
    )
    state["timings"]["routing_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return state


def select_retrieval_branch(state: RAGGraphState) -> str:
    """Conditional edge router function returning branch node name."""
    route = state.get("route", "simple")
    if route == "complex":
        return "enhanced_retrieval"
    return "basic_retrieval"


def basic_retrieval_node(state: RAGGraphState) -> RAGGraphState:
    """Node: Executes standard FAISS vector retrieval for simple candidate pool."""
    t0 = time.perf_counter()
    if state.get("timings") is None:
        state["timings"] = {}

    search_query = state.get("standalone_query") or state.get("query", "")
    candidate_k = state.get("candidate_k", settings.RERANKER_CANDIDATE_K_SIMPLE)
    logger.info(f"[Graph Node: basic_retrieval] Retrieving top_{candidate_k} candidate documents for search_query='{search_query}'")

    try:
        results_with_score = retrieval_service.similarity_search(query=search_query, top_k=candidate_k)
        state["candidate_documents"] = results_with_score
        state["candidate_count"] = len(results_with_score)
        state["retrieved_documents"] = results_with_score
    except Exception as e:
        logger.warning(f"[Graph Node: basic_retrieval] Search exception: {str(e)}")
        state["candidate_documents"] = []
        state["candidate_count"] = 0
        state["retrieved_documents"] = []
        state["error"] = str(e)

    state["timings"]["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return state


def enhanced_retrieval_node(state: RAGGraphState) -> RAGGraphState:
    """Node: Executes enhanced FAISS vector retrieval for complex candidate pool."""
    t0 = time.perf_counter()
    if state.get("timings") is None:
        state["timings"] = {}

    search_query = state.get("standalone_query") or state.get("query", "")
    candidate_k = state.get("candidate_k", settings.RERANKER_CANDIDATE_K_COMPLEX)
    logger.info(f"[Graph Node: enhanced_retrieval] Retrieving top_{candidate_k} candidate documents for search_query='{search_query}'")

    try:
        results_with_score = retrieval_service.similarity_search(query=search_query, top_k=candidate_k)
        state["candidate_documents"] = results_with_score
        state["candidate_count"] = len(results_with_score)
        state["retrieved_documents"] = results_with_score
    except Exception as e:
        logger.warning(f"[Graph Node: enhanced_retrieval] Search exception: {str(e)}")
        state["candidate_documents"] = []
        state["candidate_count"] = 0
        state["retrieved_documents"] = []
        state["error"] = str(e)

    state["timings"]["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return state


def rerank_documents_node(state: RAGGraphState) -> RAGGraphState:
    """Node: Re-ranks candidate retrieved documents using Cross-Encoder model."""
    t0 = time.perf_counter()
    if state.get("timings") is None:
        state["timings"] = {}

    search_query = state.get("standalone_query") or state.get("query", "")
    candidate_docs = state.get("candidate_documents", [])
    top_k = state.get("top_k", settings.RERANKER_TOP_K)

    if not candidate_docs:
        logger.info("[Graph Node: rerank_documents] Candidate pool is empty. Skipping re-ranking.")
        state["reranked_documents"] = []
        state["reranked_count"] = 0
        state["candidate_count"] = 0
        state["reranker_enabled"] = settings.RERANKER_ENABLED
        state["reranker_model"] = settings.RERANKER_MODEL
        state["reranker_status"] = "empty"
        state["reranker_error"] = None
        state["retrieved_documents"] = []
        state["timings"]["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return state

    if not settings.RERANKER_ENABLED:
        logger.info("[Graph Node: rerank_documents] Re-ranker disabled via configuration. Using FAISS ranking.")
        formatted_docs = []
        for idx, item in enumerate(candidate_docs[:top_k], 1):
            if isinstance(item, tuple):
                doc, score = item
                ret_score = float(score)
            elif isinstance(item, dict):
                content = item.get("content", "")
                metadata = item.get("metadata", {})
                ret_score = float(item.get("retrieval_score", 0.0))
                doc = Document(page_content=content, metadata=metadata)
            else:
                doc = item
                ret_score = 0.0

            meta = dict(doc.metadata or {})
            meta["rank"] = idx
            meta["retrieval_score"] = round(ret_score, 4)
            meta["rerank_score"] = None
            formatted_docs.append(
                {
                    "document": doc,
                    "content": doc.page_content,
                    "metadata": meta,
                    "source": meta.get("source", ""),
                    "page": meta.get("page", 1),
                    "chunk_id": meta.get("chunk_id", ""),
                    "original_filename": meta.get("original_filename", ""),
                    "retrieval_score": round(ret_score, 4),
                    "rerank_score": None,
                    "rank": idx,
                }
            )
        state["reranked_documents"] = formatted_docs
        state["reranked_count"] = len(formatted_docs)
        state["candidate_count"] = len(candidate_docs)
        state["reranker_enabled"] = False
        state["reranker_model"] = settings.RERANKER_MODEL
        state["reranker_status"] = "disabled"
        state["reranker_error"] = None
    else:
        res = reranker_service.rerank(query=search_query, documents=candidate_docs, top_k=top_k)
        state["reranked_documents"] = res["reranked_documents"]
        state["reranked_count"] = res["reranked_count"]
        state["candidate_count"] = res["candidate_count"]
        state["reranker_enabled"] = res["reranker_enabled"]
        state["reranker_model"] = res["reranker_model"]
        state["reranker_status"] = res["status"]
        state["reranker_error"] = res.get("error")

    # Format reranked documents for build_context_node
    reranked_list = state.get("reranked_documents", [])
    context_docs = []
    for item in reranked_list:
        doc_obj = item["document"] if "document" in item else Document(page_content=item["content"], metadata=item.get("metadata", {}))
        score = item.get("rerank_score") if item.get("rerank_score") is not None else item.get("retrieval_score", 0.0)
        context_docs.append((doc_obj, score))

    state["retrieved_documents"] = context_docs
    logger.info(f"[Graph Node: rerank_documents] Re-ranked {len(context_docs)} document(s) for context construction.")
    state["timings"]["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return state


PIPELINE_STAGE_KEYS = (
    "load_conversation_ms",
    "rewrite_query_ms",
    "analyze_query_ms",
    "routing_ms",
    "retrieval_ms",
    "rerank_ms",
    "context_ms",
    "prompt_ms",
    "generation_ms",
    "save_conversation_ms",
)


def build_context_node(state: RAGGraphState) -> RAGGraphState:
    """Node: Constructs structured character-bounded RAG context from retrieved documents."""
    t0 = time.perf_counter()
    if state.get("timings") is None:
        state["timings"] = {}

    docs = state.get("retrieved_documents", [])
    logger.info(f"[Graph Node: build_context] Processing {len(docs) if docs else 0} document chunk(s)")

    context_payload = context_builder.build_context(docs or [])

    state["context"] = context_payload["context"]
    state["sources"] = context_payload["sources"]
    state["retrieved_count"] = context_payload["retrieved_count"]
    state["included_count"] = context_payload["included_count"]
    state["status"] = "success" if context_payload["retrieved_count"] > 0 else "no_results"

    logger.info(f"[Graph Node: build_context] Completed context build (status='{state['status']}')")
    state["timings"]["context_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    if state.get("generate_answer") is False:
        state["timings"]["total_ms"] = round(
            sum(
                state["timings"].get(k, 0.0)
                for k in PIPELINE_STAGE_KEYS
                if isinstance(state["timings"].get(k), (int, float))
            ),
            2,
        )
    return state


def generate_prompt_node(state: RAGGraphState) -> RAGGraphState:
    """Node: Constructs custom RAG prompt payload or handles empty context grounding."""
    t0 = time.perf_counter()
    if state.get("timings") is None:
        state["timings"] = {}

    context = state.get("context", "")
    retrieved_cnt = state.get("retrieved_count", 0)

    # Empty context handling: Avoid calling LLM if no documents retrieved
    if retrieved_cnt == 0 or not context or not context.strip():
        logger.info("[Graph Node: generate_prompt] No context retrieved. Assigning grounded fallback answer.")
        state["prompt"] = ""
        state["answer"] = "I couldn't find enough information in the provided documents to answer this question."
        state["generation_status"] = "skipped_empty_context"
        state["timings"]["prompt_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return state

    query = state.get("query", "")
    history = state.get("conversation_history") or []

    if history:
        prompt = build_conversational_rag_prompt(query=query, context=context, history=history)
    else:
        prompt = build_rag_prompt(query=query, context=context)

    state["prompt"] = prompt
    state["generation_status"] = "prompt_created"
    logger.info("[Graph Node: generate_prompt] Successfully constructed RAG prompt.")
    state["timings"]["prompt_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return state


def generate_answer_node(state: RAGGraphState) -> RAGGraphState:
    """Node: Calls LLM Service to generate final grounded answer based on prompt."""
    t0 = time.perf_counter()
    if state.get("timings") is None:
        state["timings"] = {}

    state["llm_provider"] = settings.LLM_PROVIDER
    state["llm_model"] = settings.LLM_MODEL

    # If prompt generation was skipped due to empty context, keep fallback answer
    if state.get("generation_status") == "skipped_empty_context":
        logger.info("[Graph Node: generate_answer] Skipping LLM execution due to empty context.")
        state["timings"]["generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return state

    prompt = state.get("prompt", "")
    if not prompt:
        state["answer"] = "I couldn't find enough information in the provided documents to answer this question."
        state["generation_status"] = "skipped_empty_prompt"
        state["timings"]["generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return state

    logger.info(f"[Graph Node: generate_answer] Calling LLMService ({settings.LLM_PROVIDER}/{settings.LLM_MODEL})...")

    route = state.get("route", "simple")
    gen_tokens = 220 if route == "simple" else 250

    try:
        answer = llm_service.generate_answer(prompt, max_tokens=gen_tokens)
        state["answer"] = answer
        state["generation_status"] = "success"
        metrics = getattr(llm_service, "last_generation_metrics", {})
        if metrics:
            state["llm_metrics"] = metrics
            state["timings"]["prompt_eval_ms"] = metrics.get("prompt_eval_ms", 0.0)
            state["timings"]["eval_ms"] = metrics.get("eval_ms", 0.0)
            state["timings"]["prompt_eval_count"] = metrics.get("prompt_eval_count", 0)
            state["timings"]["eval_count"] = metrics.get("eval_count", 0)
            state["timings"]["tokens_per_second"] = metrics.get("tokens_per_second", 0.0)
        logger.info("[Graph Node: generate_answer] Answer generation succeeded.")
    except Exception as e:
        logger.error(f"[Graph Node: generate_answer] LLM generation failed: {str(e)}")
        state["answer"] = f"LLM generation service is currently unavailable: {str(e)}"
        state["generation_status"] = "failed"
        state["generation_error"] = str(e)

    state["timings"]["generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return state


def save_conversation_node(state: RAGGraphState) -> RAGGraphState:
    """Node: Appends current turn user query and assistant response to conversation session history."""
    t0 = time.perf_counter()
    if state.get("timings") is None:
        state["timings"] = {}

    session_id = state.get("session_id") or state.get("conversation_id")
    query = state.get("query", "").strip()
    answer = state.get("answer", "")

    if session_id and query and answer and state.get("generation_status") != "failed":
        try:
            conversation_service.add_message(session_id, "user", query)
            conversation_service.add_message(session_id, "assistant", answer)
            logger.info(f"[Graph Node: save_conversation] Saved user and assistant turns for session '{session_id}'")
        except Exception as e:
            logger.warning(f"[Graph Node: save_conversation] Error saving conversation turn: {str(e)}")

    state["timings"]["save_conversation_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    state["timings"]["total_ms"] = round(
        sum(
            state["timings"].get(k, 0.0)
            for k in PIPELINE_STAGE_KEYS
            if isinstance(state["timings"].get(k), (int, float))
        ),
        2,
    )
    return state


def should_generate_answer(state: RAGGraphState) -> str:
    """Conditional edge router: proceeds to generate_prompt only when generate_answer is True."""
    if state.get("generate_answer", True) is False:
        logger.info("[Graph Edge: build_context] generate_answer is False. Short-circuiting workflow directly to END.")
        return "end"
    return "generate_prompt"


def build_adaptive_rag_graph():
    """Builds and compiles the executable LangGraph state graph for Adaptive RAG with LLM answer generation and multi-turn memory."""
    workflow = StateGraph(RAGGraphState)

    # Add workflow nodes
    workflow.add_node("load_conversation", load_conversation_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("route_query", route_query_node)
    workflow.add_node("basic_retrieval", basic_retrieval_node)
    workflow.add_node("enhanced_retrieval", enhanced_retrieval_node)
    workflow.add_node("rerank_documents", rerank_documents_node)
    workflow.add_node("build_context", build_context_node)
    workflow.add_node("generate_prompt", generate_prompt_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("save_conversation", save_conversation_node)

    # Define entry flow
    workflow.set_entry_point("load_conversation")
    workflow.add_edge("load_conversation", "rewrite_query")
    workflow.add_edge("rewrite_query", "analyze_query")
    workflow.add_edge("analyze_query", "route_query")

    # Add conditional branching edges from route_query node
    workflow.add_conditional_edges(
        "route_query",
        select_retrieval_branch,
        {
            "basic_retrieval": "basic_retrieval",
            "enhanced_retrieval": "enhanced_retrieval",
        },
    )

    # Add convergence edges: candidate retrieval -> rerank_documents -> build_context -> [conditional] -> prompt -> answer -> save -> END
    workflow.add_edge("basic_retrieval", "rerank_documents")
    workflow.add_edge("enhanced_retrieval", "rerank_documents")
    workflow.add_edge("rerank_documents", "build_context")
    workflow.add_conditional_edges(
        "build_context",
        should_generate_answer,
        {
            "generate_prompt": "generate_prompt",
            "end": END,
        },
    )
    workflow.add_edge("generate_prompt", "generate_answer")
    workflow.add_edge("generate_answer", "save_conversation")
    workflow.add_edge("save_conversation", END)

    app_graph = workflow.compile()
    logger.info("Successfully compiled executable LangGraph Adaptive RAG state graph with re-ranking nodes")
    return app_graph


# Pre-compiled application graph instance
adaptive_rag_graph = build_adaptive_rag_graph()

