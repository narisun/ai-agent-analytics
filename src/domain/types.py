"""Domain value types — frozen dataclasses and pydantic models.

UserContext replaces the module-level auth ContextVar in platform_sdk.mcp_bridge.
It flows explicitly through ChatService and into LangGraph config["configurable"].
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UserContext:
    """Per-request user identity and auth scope.

    auth_token is never shown in repr/log output.
    """
    user_id: str
    tenant_id: str
    auth_token: str = field(repr=False)

    def __repr__(self) -> str:
        return f"UserContext(user_id={self.user_id!r}, tenant_id={self.tenant_id!r}, auth_token=<redacted>)"


from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request from the HTTP layer."""
    message: str = Field(..., min_length=1)
    conversation_id: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Structured chat response (non-streaming path)."""
    conversation_id: str
    narrative: str
    components: list[dict[str, Any]] = Field(default_factory=list)
    follow_up_suggestions: list[str] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    """Summary entry for the conversation list."""
    conversation_id: str
    title: str
    updated_at: str


class Conversation(BaseModel):
    """Full conversation with messages."""
    conversation_id: str
    title: str
    updated_at: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
