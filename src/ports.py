"""Port Protocols — consumer-owned interfaces.

These Protocols define what ChatService, routes, and nodes depend on.
Adapters in platform-sdk structurally satisfy these Protocols without
importing from analytics-agent (Dependency Inversion).
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from .domain.types import (
    Conversation,
    ConversationSummary,
    UserContext,
)


@runtime_checkable
class ConversationStore(Protocol):
    """Persistence port for conversations. Tenant-scoped via UserContext."""

    async def save(self, convo: Conversation, user_ctx: UserContext) -> None: ...

    async def load(
        self, convo_id: str, user_ctx: UserContext
    ) -> Conversation | None: ...

    async def list(self, user_ctx: UserContext) -> list[ConversationSummary]: ...

    async def delete(self, convo_id: str, user_ctx: UserContext) -> None: ...


@runtime_checkable
class MCPToolsProvider(Protocol):
    """Aggregates MCP bridges behind a single port."""

    async def get_langchain_tools(
        self, user_ctx: UserContext
    ) -> list[BaseTool]: ...


@runtime_checkable
class LLMFactory(Protocol):
    """Builds configured LLM clients."""

    def make_router_llm(self) -> BaseChatModel: ...

    def make_synthesis_llm(self) -> BaseChatModel: ...


@runtime_checkable
class StreamEncoder(Protocol):
    """Encodes graph events into wire bytes (Vercel Data Stream, SSE, etc.).

    Call contract:
      - encode_event per graph event
      - encode_error if the stream aborts
      - finalize once at end
    """

    def encode_event(self, event: dict[str, Any]) -> bytes: ...

    def encode_error(self, err: Exception, *, error_id: str) -> bytes: ...

    def finalize(self) -> bytes: ...


@runtime_checkable
class CompactionModifier(Protocol):
    """Trims long message histories to fit a token budget."""

    def apply(self, messages: list[BaseMessage]) -> list[BaseMessage]: ...


@runtime_checkable
class TelemetryScope(Protocol):
    """Abstract telemetry (spans + events) over OTel / Langfuse / etc."""

    def start_span(self, name: str) -> AbstractContextManager[Any]: ...

    def record_event(self, name: str, **attrs: Any) -> None: ...
