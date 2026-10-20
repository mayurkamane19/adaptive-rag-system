"""Service module for managing FAISS vector database indexing, persistence, and semantic search."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.core.config import settings
from app.core.logging_config import logger
from app.services.embedding_service import EmbeddingService
from app.utils.file_utils import ensure_directory_exists


class RetrievalService:
    """Service responsible for FAISS index creation, loading, persistence, document insertion, and similarity search."""

    def __init__(
        self,
        vector_db_path: Union[str, Path] = settings.VECTOR_DB_PATH,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        self.vector_db_path = Path(vector_db_path)
        ensure_directory_exists(self.vector_db_path)
        self.embedding_service = embedding_service or EmbeddingService()
        self._last_loaded_mtime: float = 0.0
        self.vector_store: Optional[FAISS] = self.load_vector_store()

    def vector_store_exists(self) -> bool:
        """Checks whether a valid FAISS index exists on disk."""
        faiss_file = self.vector_db_path / "index.faiss"
        pkl_file = self.vector_db_path / "index.pkl"
        return faiss_file.is_file() and pkl_file.is_file()

    def _get_disk_index_mtime(self) -> float:
        """Returns the latest modification timestamp of the index files on disk."""
        faiss_file = self.vector_db_path / "index.faiss"
        pkl_file = self.vector_db_path / "index.pkl"
        mtime = 0.0
        if faiss_file.is_file():
            mtime = max(mtime, faiss_file.stat().st_mtime)
        if pkl_file.is_file():
            mtime = max(mtime, pkl_file.stat().st_mtime)
        return mtime

    def load_vector_store(self) -> Optional[FAISS]:
        """Loads existing FAISS vector store from disk if available."""
        if not self.vector_store_exists():
            logger.info(f"No existing FAISS vector store found at '{self.vector_db_path}'.")
            self._last_loaded_mtime = 0.0
            return None

        try:
            logger.info(f"Loading FAISS vector store from '{self.vector_db_path}'")
            store = FAISS.load_local(
                folder_path=str(self.vector_db_path),
                embeddings=self.embedding_service.get_embeddings(),
                allow_dangerous_deserialization=True,
            )
            self._last_loaded_mtime = self._get_disk_index_mtime()
            logger.info("Successfully loaded FAISS vector store into memory.")
            return store
        except Exception as e:
            logger.error(f"Error loading FAISS vector store from '{self.vector_db_path}': {str(e)}")
            return None

    def reload_vector_store(self) -> Optional[FAISS]:
        """Explicitly reloads the FAISS vector store from disk."""
        logger.info(f"Explicitly reloading FAISS vector store from '{self.vector_db_path}'")
        self.vector_store = self.load_vector_store()
        return self.vector_store

    def reload_if_disk_updated(self) -> None:
        """Checks if the disk index has changed since it was last loaded, and reloads if necessary."""
        if not self.vector_store_exists():
            if self.vector_store is not None:
                self.vector_store = None
                self._last_loaded_mtime = 0.0
            return

        disk_mtime = self._get_disk_index_mtime()
        if self.vector_store is None:
            self.vector_store = self.load_vector_store()
        elif disk_mtime > 0 and disk_mtime > self._last_loaded_mtime:
            logger.info(
                f"Detected updated FAISS index on disk (disk mtime={disk_mtime} > loaded mtime={self._last_loaded_mtime}). Reloading store into memory."
            )
            self.vector_store = self.load_vector_store()

    def save_vector_store(self) -> None:
        """Persists the current FAISS vector store to disk."""
        if self.vector_store is None:
            logger.warning("Attempted to save vector store, but no store is currently initialized.")
            return

        try:
            ensure_directory_exists(self.vector_db_path)
            self.vector_store.save_local(str(self.vector_db_path))
            self._last_loaded_mtime = self._get_disk_index_mtime()
            logger.info(f"Successfully saved FAISS vector store to '{self.vector_db_path}'")
        except Exception as e:
            logger.error(f"Failed to save FAISS vector store to '{self.vector_db_path}': {str(e)}")
            raise IOError(f"Failed to save vector store: {str(e)}") from e

    def is_duplicate_document(self, original_filename: str) -> bool:
        """Checks if a document with original_filename has already been indexed in the vector store."""
        self.reload_if_disk_updated()

        if self.vector_store is None or not hasattr(self.vector_store, "docstore"):
            return False

        try:
            for doc in self.vector_store.docstore._dict.values():
                if doc.metadata.get("original_filename") == original_filename:
                    return True
        except Exception as e:
            logger.warning(f"Could not inspect docstore metadata for duplicates: {str(e)}")

        return False

    def add_documents(
        self, documents: List[Document], original_filename: str
    ) -> Dict[str, Any]:
        """Adds document chunks to FAISS vector store with duplicate checking and persistence."""
        if not documents:
            return {
                "status": "warning",
                "message": "No document chunks provided for indexing.",
                "added_chunks": 0,
            }

        # Check disk mtime and reload if disk has newer documents before checking duplicates
        self.reload_if_disk_updated()

        # Check for duplicates
        if self.is_duplicate_document(original_filename):
            logger.info(f"Duplicate document detected: '{original_filename}' is already indexed.")
            return {
                "status": "duplicate",
                "message": f"Document '{original_filename}' has already been indexed in the vector store.",
                "added_chunks": 0,
            }

        logger.info(f"Adding {len(documents)} chunk(s) from '{original_filename}' to FAISS store.")

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_service.get_embeddings(),
            )
        else:
            self.vector_store.add_documents(documents)

        self.save_vector_store()

        total_docs = len(self.vector_store.docstore._dict) if hasattr(self.vector_store, "docstore") else len(documents)

        return {
            "status": "indexed",
            "message": f"Successfully indexed {len(documents)} chunk(s) from '{original_filename}'.",
            "added_chunks": len(documents),
            "total_chunks": total_docs,
        }

    def similarity_search(
        self, query: str, top_k: int = settings.TOP_K
    ) -> List[Tuple[Document, float]]:
        """Performs semantic similarity search against the FAISS vector store."""
        if not query or not query.strip():
            raise ValueError("Search query string cannot be empty.")

        self.reload_if_disk_updated()

        if self.vector_store is None:
            logger.warning("Similarity search failed: No FAISS vector store found on disk.")
            raise ValueError("Vector store is empty or does not exist. Please index documents before searching.")

        logger.info(f"Executing similarity search for query: '{query}' (top_k={top_k})")
        results_with_score = self.vector_store.similarity_search_with_score(query, k=top_k)

        logger.info(f"Retrieved {len(results_with_score)} matching chunk(s) from FAISS store.")
        return results_with_score


retrieval_service = RetrievalService()
