"""Service module for generating text vector embeddings using Hugging Face models."""

from typing import List, Optional, Dict, Any
from langchain_core.embeddings import Embeddings

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from app.core.config import settings
from app.core.logging_config import logger


class EmbeddingService:
    """Service for initializing vector embedding models and generating text embeddings."""

    _models_cache: dict[str, Any] = {}

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL) -> None:
        self.model_name = model_name

        if self.model_name in EmbeddingService._models_cache:
            self._embeddings_model = EmbeddingService._models_cache[self.model_name]
            logger.debug(f"Reusing cached HuggingFaceEmbeddings model: '{self.model_name}'")
            return

        logger.info(f"Initializing HuggingFaceEmbeddings model: '{self.model_name}'")
        try:
            try:
                # First try loading with local_files_only to bypass slow remote HTTP checks
                self._embeddings_model = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": "cpu", "local_files_only": True},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception as local_err:
                logger.info(f"Could not load embedding model with local_files_only ({local_err}); checking online repository...")
                self._embeddings_model = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )

            EmbeddingService._models_cache[self.model_name] = self._embeddings_model
            logger.info(f"Successfully loaded and cached embedding model '{self.model_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model '{self.model_name}': {str(e)}")
            raise RuntimeError(f"Embedding model initialization error: {str(e)}") from e

    @classmethod
    def prewarm(cls, model_name: str = settings.EMBEDDING_MODEL) -> "EmbeddingService":
        """Pre-initializes and caches the embedding model in memory."""
        return cls(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a list of document chunk texts."""
        if not texts:
            return []
        logger.debug(f"Generating embeddings for {len(texts)} document chunk(s)")
        return self._embeddings_model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        """Generates an embedding vector for a single search query."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty for embedding generation.")
        logger.debug(f"Generating embedding vector for query: '{query[:50]}...'")
        return self._embeddings_model.embed_query(query)

    def get_embeddings(self) -> Embeddings:
        """Returns the underlying LangChain Embeddings instance for vector store integration."""
        return self._embeddings_model
