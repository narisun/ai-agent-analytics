"""Conversation persistence layer for the analytics agent.

Provides async conversation storage with support for both PostgreSQL and in-memory backends.
Used by the FastAPI app to persist conversation history for the frontend sidebar.

PostgresConversationStore requires asyncpg and is imported lazily to avoid hard
failures in environments where the package is not installed.
"""

from .conversation_store import ConversationStore
from .memory_store import MemoryConversationStore

try:
    from .postgres_store import PostgresConversationStore
except ImportError:
    PostgresConversationStore = None  # type: ignore[assignment,misc]

__all__ = [
    "ConversationStore",
    "MemoryConversationStore",
    "PostgresConversationStore",
]
