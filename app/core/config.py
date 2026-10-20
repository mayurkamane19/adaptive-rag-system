"""Application configuration module using Pydantic Settings."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables or defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    APP_NAME: str = Field(default="Adaptive RAG System", description="Name of the application")
    APP_VERSION: str = Field(default="1.0.0", description="Version of the application")
    DEBUG: bool = Field(default=True, description="Debug mode flag")
    HOST: str = Field(default="127.0.0.1", description="Host address for server execution")
    PORT: int = Field(default=8000, description="Port number for server execution")

    # Security & Production Hardening Parameters
    ALLOWED_ORIGINS: str = Field(default="*", description="Comma-separated list of allowed CORS origins")
    API_KEY: Optional[str] = Field(default=None, description="Optional API key for securing sensitive endpoints")
    MAX_UPLOAD_SIZE_MB: int = Field(default=25, description="Maximum allowed file upload size in megabytes")
    RATE_LIMIT_ENABLED: bool = Field(default=False, description="Enable request rate limiting")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="Allowed requests per minute per IP")

    # LLM Provider Parameters
    LLM_PROVIDER: str = Field(default="ollama", description="LLM provider name (e.g. ollama, openai)")
    LLM_MODEL: str = Field(default="llama3", description="LLM model name")
    LLM_TEMPERATURE: float = Field(default=0.2, description="Sampling temperature for LLM answer generation")
    LLM_MAX_TOKENS: int = Field(default=1000, description="Maximum tokens for generated LLM response")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", description="Base URL for local Ollama API service")
    OLLAMA_TIMEOUT_SECONDS: float = Field(default=240.0, description="HTTP request timeout in seconds for Ollama API calls")

    # Embedding Model Parameters
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name or path",
    )

    VECTOR_DB_PATH: str = Field(default="data/vectorstore", description="Path for vector database storage")
    DOCUMENTS_PATH: str = Field(default="data/documents", description="Path for uploaded raw documents")
    CONVERSATIONS_PATH: str = Field(default="data/conversations", description="Path for persistent conversation sessions")

    # Document Processing & Chunking Parameters
    CHUNK_SIZE: int = Field(default=1000, description="Character chunk size for document splitting")
    CHUNK_OVERLAP: int = Field(default=200, description="Character overlap between consecutive chunks")

    # Retrieval & Context Parameters
    TOP_K: int = Field(default=4, description="Default number of top relevant document chunks to retrieve")
    SIMPLE_TOP_K: int = Field(default=4, description="Retrieval depth for simple queries")
    COMPLEX_TOP_K: int = Field(default=8, description="Retrieval depth for complex queries")
    MAX_CONTEXT_CHARS: int = Field(default=12000, description="Maximum character limit for constructed RAG context")

    # Re-ranking Parameters
    RERANKER_ENABLED: bool = Field(default=True, description="Enable post-retrieval cross-encoder re-ranking stage")
    RERANKER_MODEL: str = Field(
        default="BAAI/bge-reranker-base",
        description="Sentence Transformers Cross-Encoder model name",
    )
    RERANKER_TOP_K: int = Field(default=4, description="Final number of re-ranked top relevant documents to include in context")
    RERANKER_TOP_K_COMPLEX: int = Field(default=5, description="Final number of re-ranked top relevant documents to include in context for complex queries")
    RERANKER_CANDIDATE_K_SIMPLE: int = Field(default=8, description="FAISS candidate retrieval pool size for simple queries")
    RERANKER_CANDIDATE_K_COMPLEX: int = Field(default=16, description="FAISS candidate retrieval pool size for complex queries")

    # Conversational Memory Parameters
    MAX_CONVERSATION_MESSAGES: int = Field(
        default=10,
        description="Maximum recent conversation messages included in query rewriting and prompts",
    )

    # Evaluation Parameters
    EVALUATION_DATASET_PATH: str = Field(
        default="data/evaluation/eval_dataset.json",
        description="Path to default RAG benchmark evaluation dataset",
    )


settings = Settings()
