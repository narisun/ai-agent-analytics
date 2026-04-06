"""Abstract conversation persistence interface (Protocol)."""

from typing import Protocol


class ConversationStore(Protocol):
    """Abstract conversation persistence interface.

    Defines the contract for conversation storage backends (PostgreSQL, in-memory, etc).
    All methods are async and must be implemented by concrete stores.
    """

    async def list_conversations(self, limit: int = 50) -> list[dict]:
        """List conversations, sorted by most recently updated.

        Args:
            limit: Maximum number of conversations to return.

        Returns:
            List of conversation dicts with keys: id, title, created_at, updated_at.
        """
        ...

    async def get_conversation(self, conversation_id: str) -> dict | None:
        """Get a single conversation with its full message history.

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            Dict with keys: id, title, created_at, updated_at, messages (list of message dicts).
            None if conversation does not exist.
        """
        ...

    async def create_conversation(self, conversation_id: str, title: str) -> dict:
        """Create a new conversation.

        Args:
            conversation_id: UUID for the new conversation.
            title: Initial conversation title.

        Returns:
            Conversation dict with keys: id, title, created_at, updated_at, messages=[].
        """
        ...

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
            components: Optional list of UI component dicts (charts, KPIs, etc).

        Returns:
            Message dict with keys: id, conversation_id, role, content, components, created_at.
        """
        ...

    async def update_title(self, conversation_id: str, title: str) -> None:
        """Update the title of a conversation.

        Args:
            conversation_id: UUID of the conversation.
            title: New conversation title.
        """
        ...

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all its messages.

        Args:
            conversation_id: UUID of the conversation.
        """
        ...
