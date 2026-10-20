"""Service module for managing persistent and in-memory multi-turn conversation sessions and chat history."""

import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Literal, Union

from app.models.message import ConversationMessage
from app.core.config import settings
from app.core.logging_config import logger
from app.utils.file_utils import ensure_directory_exists


class ConversationService:
    """Service providing thread-safe conversation session state management with disk persistence."""

    _instance: Optional["ConversationService"] = None

    def __init__(self, storage_path: Union[str, Path] = settings.CONVERSATIONS_PATH) -> None:
        self.storage_path = Path(storage_path)
        ensure_directory_exists(self.storage_path)
        self._sessions: Dict[str, List[ConversationMessage]] = {}
        logger.info(f"Initialized ConversationService (storage_path='{self.storage_path}')")

    def _get_session_path(self, session_id: str) -> Path:
        """Computes safe sanitized file path for a session identifier."""
        safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id.strip())
        return self.storage_path / f"{safe_id}.json"

    def _save_session_to_disk(self, session_id: str) -> None:
        """Persists session message history to disk as JSON."""
        try:
            path = self._get_session_path(session_id)
            messages = self._sessions.get(session_id, [])
            data = [msg.model_dump() for msg in messages]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Persisted {len(messages)} messages for session '{session_id}' to '{path}'")
        except Exception as e:
            logger.warning(f"Failed to persist session '{session_id}' to disk: {str(e)}")

    def _load_session_from_disk(self, session_id: str) -> Optional[List[ConversationMessage]]:
        """Loads session message history from disk if file exists."""
        path = self._get_session_path(session_id)
        if not path.is_file():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            messages = [ConversationMessage(**item) for item in data]
            self._sessions[session_id] = messages
            logger.info(f"Loaded {len(messages)} messages for session '{session_id}' from disk cache.")
            return messages
        except Exception as e:
            logger.warning(f"Error loading session history from '{path}': {str(e)}")
            return None

    def session_exists(self, session_id: str) -> bool:
        """Checks whether a conversation session exists in memory or on disk."""
        if not session_id or not session_id.strip():
            return False
        clean_id = session_id.strip()
        if clean_id in self._sessions:
            return True
        return self._get_session_path(clean_id).is_file()

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Creates a new conversation session or restores existing session ID from cache/disk."""
        clean_id = session_id.strip() if session_id and session_id.strip() else f"session-{uuid.uuid4().hex[:8]}"

        if clean_id not in self._sessions:
            loaded = self._load_session_from_disk(clean_id)
            if loaded is None:
                self._sessions[clean_id] = []
                self._save_session_to_disk(clean_id)
                logger.info(f"Created new conversation session: '{clean_id}'")

        return clean_id

    def get_history(
        self, session_id: str, limit: Optional[int] = settings.MAX_CONVERSATION_MESSAGES
    ) -> List[ConversationMessage]:
        """Retrieves history for a session, optionally bounded by maximum recent messages limit."""
        if not session_id or not session_id.strip():
            return []

        clean_id = session_id.strip()
        if clean_id not in self._sessions:
            loaded = self._load_session_from_disk(clean_id)
            if loaded is None:
                logger.debug(f"Requested history for non-existent session '{clean_id}'; returning empty list.")
                return []

        history = self._sessions[clean_id]
        if limit and len(history) > limit:
            return history[-limit:]
        return list(history)

    def add_message(
        self, session_id: str, role: Literal["user", "assistant"], content: str
    ) -> ConversationMessage:
        """Appends a user or assistant message to session history and persists to disk."""
        if not content or not content.strip():
            raise ValueError("Message content cannot be empty.")

        clean_id = self.create_session(session_id)
        msg = ConversationMessage(role=role, content=content.strip())
        self._sessions[clean_id].append(msg)
        self._save_session_to_disk(clean_id)
        logger.info(f"Added message (role='{role}', len={len(content)}) to session '{clean_id}'")
        return msg

    def clear_session(self, session_id: str) -> bool:
        """Clears the history of the specified session ID from memory and disk."""
        if not session_id:
            return False

        clean_id = session_id.strip()
        existed = False
        if clean_id in self._sessions:
            del self._sessions[clean_id]
            existed = True

        path = self._get_session_path(clean_id)
        if path.is_file():
            try:
                path.unlink()
                existed = True
            except Exception as e:
                logger.warning(f"Error removing session file '{path}': {str(e)}")

        if existed:
            logger.info(f"Successfully cleared conversation session: '{clean_id}'")
            return True

        logger.info(f"Attempted to clear session '{clean_id}', but it did not exist.")
        return False


# Module-level default conversation service instance
conversation_service = ConversationService()

