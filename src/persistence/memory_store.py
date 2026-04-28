"""In-memory conversation store for development and fallback.

Used when ENVIRONMENT=local or DATABASE_URL is not set.
All data is lost on shutdown.
"""

import uuid
from datetime import datetime

from platform_sdk import get_logger

log = get_logger(__name__)


class MemoryConversationStore:
    """In-memory implementation of conversation persistence.

    Stores conversations and messages in nested dicts for dev/test environments.
    No persistence across application restarts.
    """

    def __init__(self):
        """Initialize the in-memory store."""
        self.conversations: dict[str, dict] = {}
        log.info("memory_conversation_store_initialized")

    async def list_conversations(self, limit: int = 50) -> list[dict]:
        """List conversations sorted by most recently updated.

        Args:
            limit: Maximum number of conversations to return.

        Returns:
            List of conversation dicts (without messages).
        """
        sorted_convs = sorted(
            self.conversations.values(),
            key=lambda c: c["updated_at"],
            reverse=True,
        )
        return [
            {
                "id": c["id"],
                "title": c["title"],
                "created_at": c["created_at"],
                "updated_at": c["updated_at"],
            }
            for c in sorted_convs[:limit]
        ]

    async def get_conversation(self, conversation_id: str) -> dict | None:
        """Get a conversation with full message history.

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            Conversation dict with messages array, or None if not found.
        """
        conv = self.conversations.get(conversation_id)
        if not conv:
            return None

        return {
            "id": conv["id"],
            "title": conv["title"],
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "messages": conv.get("messages", []),
        }

    async def create_conversation(self, conversation_id: str, title: str) -> dict:
        """Create a new conversation.

        Args:
            conversation_id: UUID for the conversation.
            title: Initial conversation title.

        Returns:
            Conversation dict with empty messages array.
        """
        now = datetime.utcnow().isoformat()
        conv = {
            "id": conversation_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self.conversations[conversation_id] = conv
        log.info("conversation_created", conversation_id=conversation_id)
        return conv

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        components: list | None = None,
    ) -> dict:
        """Add a message to a conversation.

        Args:
            conversation_id: UUID of the conversation.
            role: Message role ('user' or 'assistant').
            content: Message text content.
            components: Optional list of UI component dicts.

        Returns:
            Message dict with id, role, content, components, created_at.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role: {role}")

        conv = self.conversations.get(conversation_id)
        if not conv:
            raise ValueError(f"Conversation not found: {conversation_id}")

        now = datetime.utcnow().isoformat()
        message = {
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "components": components or [],
            "created_at": now,
        }

        conv["messages"].append(message)
        conv["updated_at"] = now
        return message

    async def update_title(self, conversation_id: str, title: str) -> None:
        """Update the title of a conversation.

        Args:
            conversation_id: UUID of the conversation.
            title: New conversation title.
        """
        conv = self.conversations.get(conversation_id)
        if not conv:
            raise ValueError(f"Conversation not found: {conversation_id}")

        conv["title"] = title
        conv["updated_at"] = datetime.utcnow().isoformat()

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all its messages.

        Args:
            conversation_id: UUID of the conversation.
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            log.info("conversation_deleted", conversation_id=conversation_id)
