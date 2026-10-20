"""Conversation message model for multi-turn chat memory."""

from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """Data model representing a single message in a conversation session."""

    role: Literal["user", "assistant"] = Field(..., description="Role of the message sender ('user' or 'assistant')")
    content: str = Field(..., description="Text content of the message")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of message creation",
    )
