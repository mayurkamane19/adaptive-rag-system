import threading
from typing import List, Dict, Any, Tuple, Optional, Union
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging_config import logger


class RerankerService:
    """Service responsible for re-ranking retrieved candidate documents using a Cross-Encoder model."""

    _models_cache: Dict[str, Any] = {}
    _rerank_lock: threading.Lock = threading.Lock()

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = RerankerService._models_cache.get(self.model_name)
        logger.info(f"Initialized RerankerService with model setting: '{self.model_name}'")

    def _get_model(self):
        """Lazy loader for CrossEncoder model instance with local cache fast-path."""
        if self._model is not None:
            return self._model

        if self.model_name in RerankerService._models_cache:
            self._model = RerankerService._models_cache[self.model_name]
            return self._model

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder re-ranking model '{self.model_name}' into memory...")
            try:
                # First attempt to load with local_files_only to bypass slow remote HTTP checks
                self._model = CrossEncoder(self.model_name, local_files_only=True)
            except Exception as local_err:
                logger.info(f"Could not load CrossEncoder with local_files_only ({local_err}); checking online repository...")
                self._model = CrossEncoder(self.model_name)

            RerankerService._models_cache[self.model_name] = self._model
            logger.info(f"Successfully loaded and cached CrossEncoder model '{self.model_name}'.")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model '{self.model_name}': {str(e)}")
            raise RuntimeError(f"Could not load CrossEncoder model '{self.model_name}': {str(e)}") from e
        return self._model

    def prewarm(self) -> None:
        """Pre-loads model into memory so user requests have zero cold-start delay."""
        if settings.RERANKER_ENABLED:
            try:
                self._get_model()
            except Exception as e:
                logger.warning(f"Reranker prewarming warning: {e}")

    def rerank(
        self,
        query: str,
        documents: List[Union[Tuple[Document, float], Dict[str, Any], Document]],
        top_k: int = settings.RERANKER_TOP_K,
    ) -> Dict[str, Any]:
        """Re-ranks candidate documents based on cross-encoder similarity scores against the query.

        Args:
            query: The search or user query.
            documents: Candidate document items (tuples of Document & score, dicts, or Document objects).
            top_k: Number of top re-ranked documents to return.

        Returns:
            Dict containing status, reranked_documents, candidate_count, reranked_count, reranker_enabled, reranker_model, error.
        """
        if not documents:
            logger.info("Empty candidate document list passed to RerankerService. Returning empty payload.")
            return {
                "status": "empty",
                "reranked_documents": [],
                "candidate_count": 0,
                "reranked_count": 0,
                "reranker_enabled": settings.RERANKER_ENABLED,
                "reranker_model": self.model_name,
                "error": None,
            }

        if not query or not query.strip():
            logger.warning("Empty query passed to RerankerService.")
            clean_query = ""
        else:
            clean_query = query.strip()

        candidate_count = len(documents)

        # Normalize candidate items into List[Tuple[Document, float]]
        normalized_candidates: List[Tuple[Document, float]] = []
        for item in documents:
            if isinstance(item, tuple):
                doc, score = item
                normalized_candidates.append((doc, float(score)))
            elif isinstance(item, dict):
                content = item.get("content", "")
                metadata = item.get("metadata", {})
                ret_score = float(item.get("retrieval_score") if item.get("retrieval_score") is not None else item.get("score", 0.0))
                doc = Document(page_content=content, metadata=metadata)
                normalized_candidates.append((doc, ret_score))
            elif isinstance(item, Document):
                normalized_candidates.append((item, 0.0))

        if not clean_query:
            # Fall back directly if query is empty
            fallback_list = []
            for rank_idx, (doc, ret_score) in enumerate(normalized_candidates[:top_k], 1):
                meta = dict(doc.metadata or {})
                meta["rank"] = rank_idx
                meta["retrieval_score"] = round(ret_score, 4)
                meta["rerank_score"] = None
                fallback_list.append(
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
                        "rank": rank_idx,
                    }
                )
            return {
                "status": "fallback",
                "reranked_documents": fallback_list,
                "candidate_count": candidate_count,
                "reranked_count": len(fallback_list),
                "reranker_enabled": True,
                "reranker_model": self.model_name,
                "error": "Query string was empty",
            }

        try:
            model = self._get_model()
            pairs = [(clean_query, doc.page_content) for doc, _ in normalized_candidates]

            logger.info(f"Re-ranking {len(pairs)} candidate document(s) using CrossEncoder model '{self.model_name}'...")
            lock_acquired = self._rerank_lock.acquire(timeout=30.0)
            if not lock_acquired:
                raise TimeoutError("Reranker inference lock acquisition timed out.")
            try:
                raw_scores = model.predict(pairs)
            finally:
                self._rerank_lock.release()

            scored_docs = []
            for (doc, ret_score), r_score in zip(normalized_candidates, raw_scores):
                scored_docs.append((doc, ret_score, float(r_score)))

            # Sort descending by cross-encoder score
            scored_docs.sort(key=lambda x: x[2], reverse=True)

            final_scored = scored_docs[:top_k]

            reranked_list: List[Dict[str, Any]] = []
            for rank_idx, (doc, ret_score, r_score) in enumerate(final_scored, 1):
                meta = dict(doc.metadata or {})
                meta["rank"] = rank_idx
                meta["retrieval_score"] = round(ret_score, 4)
                meta["rerank_score"] = round(r_score, 4)

                reranked_list.append(
                    {
                        "document": doc,
                        "content": doc.page_content,
                        "metadata": meta,
                        "source": meta.get("source", ""),
                        "page": meta.get("page", 1),
                        "chunk_id": meta.get("chunk_id", ""),
                        "original_filename": meta.get("original_filename", ""),
                        "retrieval_score": round(ret_score, 4),
                        "rerank_score": round(r_score, 4),
                        "rank": rank_idx,
                    }
                )

            logger.info(f"Successfully re-ranked candidate pool: returned top_{len(reranked_list)} documents.")
            return {
                "status": "success",
                "reranked_documents": reranked_list,
                "candidate_count": candidate_count,
                "reranked_count": len(reranked_list),
                "reranker_enabled": True,
                "reranker_model": self.model_name,
                "error": None,
            }

        except Exception as e:
            logger.warning(f"Re-ranking model execution failed ({str(e)}). Falling back to FAISS retrieval ranking.")

            fallback_list = []
            for rank_idx, (doc, ret_score) in enumerate(normalized_candidates[:top_k], 1):
                meta = dict(doc.metadata or {})
                meta["rank"] = rank_idx
                meta["retrieval_score"] = round(ret_score, 4)
                meta["rerank_score"] = None

                fallback_list.append(
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
                        "rank": rank_idx,
                    }
                )

            return {
                "status": "fallback",
                "reranked_documents": fallback_list,
                "candidate_count": candidate_count,
                "reranked_count": len(fallback_list),
                "reranker_enabled": False,
                "reranker_model": self.model_name,
                "error": str(e),
            }


reranker_service = RerankerService()
