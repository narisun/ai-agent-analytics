"""In-memory ConversationStore for tests."""
from __future__ import annotations

from src.domain.types import (
    Conversation,
    ConversationSummary,
    UserContext,
)


class FakeConversationStore:
    """Tenant-scoped in-memory store. Not thread-safe."""

    def __init__(self) -> None:
        # key = (tenant_id, conversation_id)
        self._data: dict[tuple[str, str], Conversation] = {}

    async def save(self, convo: Conversation, user_ctx: UserContext) -> None:
        self._data[(user_ctx.tenant_id, convo.conversation_id)] = convo

    async def load(
        self, convo_id: str, user_ctx: UserContext
    ) -> Conversation | None:
        return self._data.get((user_ctx.tenant_id, convo_id))

    async def list(self, user_ctx: UserContext) -> list[ConversationSummary]:
        return [
            ConversationSummary(
                conversation_id=c.conversation_id,
                title=c.title,
                updated_at=c.updated_at,
            )
            for (tenant, _), c in self._data.items()
            if tenant == user_ctx.tenant_id
        ]

    async def delete(self, convo_id: str, user_ctx: UserContext) -> None:
        self._data.pop((user_ctx.tenant_id, convo_id), None)
