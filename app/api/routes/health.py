"""Health check, liveness, and readiness probe endpoint router."""

import httpx
from fastapi import APIRouter
from app.core.config import settings
from app.core.logging_config import logger
from app.schemas.common import HealthResponse, LivenessResponse, ReadinessResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Returns the operational health status of the API service."""
    return HealthResponse(
        status="healthy",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
    )


@router.get("/health/live", response_model=LivenessResponse)
def liveness_check() -> LivenessResponse:
    """Confirms that the FastAPI process is alive and responsive."""
    return LivenessResponse(
        status="alive",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
    )


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness_check() -> ReadinessResponse:
    """Readiness probe checking availability of core subsystems (FAISS, LLM, Re-ranker)."""
    # 1. Vector Store probe
    vs_status = "unavailable"
    vs_chunks = 0
    try:
        from app.services.retrieval_service import RetrievalService, retrieval_service
        ret_svc = retrieval_service
        if ret_svc.vector_store_exists():
            vs_status = "available"
            if ret_svc.vector_store and hasattr(ret_svc.vector_store, "docstore"):
                vs_chunks = len(ret_svc.vector_store.docstore._dict)
        else:
            vs_status = "empty"
    except Exception as e:
        logger.warning(f"Readiness probe vector store check error: {str(e)}")
        vs_status = "error"

    # 2. Ollama / LLM probe (non-blocking fast ping)
    llm_status = "unavailable"
    if settings.LLM_PROVIDER.lower() == "ollama":
        try:
            from app.services.llm_service import llm_service
            conn_info = llm_service.check_connection()
            llm_status = conn_info.get("status", "unavailable")
        except Exception:
            llm_status = "unreachable"
    else:
        llm_status = "configured"

    # 3. Re-ranker probe
    reranker_status = "enabled" if settings.RERANKER_ENABLED else "disabled"

    overall_status = "ready" if llm_status in ("connected", "configured") else "degraded"

    return ReadinessResponse(
        status=overall_status,
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        components={
            "vector_store": {
                "status": vs_status,
                "chunks_indexed": vs_chunks,
            },
            "llm_service": {
                "provider": settings.LLM_PROVIDER,
                "model": settings.LLM_MODEL,
                "status": llm_status,
            },
            "reranker": {
                "enabled": settings.RERANKER_ENABLED,
                "model": settings.RERANKER_MODEL,
                "status": reranker_status,
            },
        },
    )
