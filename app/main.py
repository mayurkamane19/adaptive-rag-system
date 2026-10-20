"""Main FastAPI application module."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging_config import logger
from app.api.routes import health, documents, chat, evaluation


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for application startup and shutdown tasks."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} (Debug: {settings.DEBUG})")
    try:
        from app.services.embedding_service import EmbeddingService
        from app.services.reranker_service import reranker_service
        from app.services.retrieval_service import retrieval_service

        logger.info("Pre-warming embedding and re-ranking models in memory...")
        EmbeddingService.prewarm()
        if settings.RERANKER_ENABLED:
            reranker_service.prewarm()
        retrieval_service.load_vector_store()
        logger.info("Startup model pre-warming and vector store loading completed successfully.")
    except Exception as e:
        logger.warning(f"Startup pre-warming encountered a non-fatal warning: {e}")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise-grade Adaptive Retrieval-Augmented Generation (RAG) System backend API.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Parse allowed CORS origins from settings
allowed_origins_list = [
    origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",") if origin.strip()
]
if not allowed_origins_list:
    allowed_origins_list = ["*"]

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catches unhandled server exceptions to prevent stack trace leaks in responses."""
    logger.error(
        f"Unhandled server error processing {request.method} {request.url.path}: {str(exc)}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "detail": "An internal server error occurred while processing your request.",
        },
    )


# Register API Routers
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(evaluation.router)


@app.get("/", include_in_schema=False)
def root_redirect():
    """Root endpoint providing quick status check."""
    return {
        "message": f"Welcome to {settings.APP_NAME} API. Access documentation at /docs",
        "status": "online",
    }
