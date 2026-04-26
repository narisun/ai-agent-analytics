"""ConversationService — thin façade over the ConversationStore Protocol.

The store handles persistence and tenant-scoping; this service adds:
- a stable error contract (ConversationNotFound) — routes catch this
  and translate to HTTP 404 via the registered exception handler.
- idempotent delete semantics — deleting a non-existent conversation
  is a no-op rather than an error.

UserContext is a per-call argument (not a constructor field) because
one service instance handles requests from many tenants. ChatService
holds user_ctx per-instance because it is constructed per-request.
"""
from __future__ import annotations

from ..domain.errors import ConversationNotFound
from ..domain.types import Conversation, ConversationSummary, UserContext
from ..ports import ConversationStore


class ConversationService:
    """Tenant-scoped CRUD over conversations."""

    def __init__(self, *, store: ConversationStore) -> None:
        self._store = store

    async def save(self, convo: Conversation, user_ctx: UserContext) -> None:
        await self._store.save(convo, user_ctx)

    async def get(self, conversation_id: str, user_ctx: UserContext) -> Conversation:
        convo = await self._store.load(conversation_id, user_ctx)
        if convo is None:
            raise ConversationNotFound(conversation_id=conversation_id)
        return convo

    async def list(self, user_ctx: UserContext) -> list[ConversationSummary]:
        return await self._store.list(user_ctx)

    async def delete(self, conversation_id: str, user_ctx: UserContext) -> None:
        # Idempotent — store.delete is a no-op for missing keys (per the
        # FakeConversationStore contract; PostgresConversationStore should
        # match). We do NOT raise ConversationNotFound here because callers
        # commonly call delete in cleanup paths where existence is unknown.
        await self._store.delete(conversation_id, user_ctx)
