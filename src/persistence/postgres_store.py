"""PostgreSQL-backed conversation store for async persistence."""

import asyncpg
from datetime import datetime
from typing import Optional
import json

from platform_sdk import get_logger

log = get_logger(__name__)


class PostgresConversationStore:
    """Async PostgreSQL implementation of conversation persistence.

    Manages conversations and messages using a dedicated connection pool.
    Handles lifecycle via connect()/disconnect() methods.
    """

    def __init__(self, database_url: str):
        """Initialize the PostgreSQL conversation store.

        Args:
            database_url: PostgreSQL connection string (e.g. postgres://user:pass@host/db).
        """
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Establish the connection pool on startup."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=10,
            )
            log.info("postgres_conversation_store_connected")
        except Exception as exc:
            log.error("postgres_conversation_store_connect_failed", error=str(exc))
            raise

    async def disconnect(self) -> None:
        """Close the connection pool on shutdown."""
        if self.pool:
            await self.pool.close()
            log.info("postgres_conversation_store_disconnected")

    async def list_conversations(self, limit: int = 50) -> list[dict]:
        """List conversations sorted by most recently updated.

        Args:
            limit: Maximum number of conversations to return.

        Returns:
            List of conversation dicts (without messages).
        """
        if not self.pool:
            raise RuntimeError("PostgresConversationStore not connected")

        try:
            rows = await self.pool.fetch(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT $1
                """,
                limit,
            )
            return [
                {
                    "id": str(row["id"]),
                    "title": row["title"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                }
                for row in rows
            ]
        except Exception as exc:
            log.error("list_conversations_failed", error=str(exc))
            raise

    async def get_conversation(self, conversation_id: str) -> dict | None:
        """Get a conversation with full message history.

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            Conversation dict with messages array, or None if not found.
        """
        if not self.pool:
            raise RuntimeError("PostgresConversationStore not connected")

        try:
            conv_row = await self.pool.fetchrow(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations
                WHERE id = $1
                """,
                conversation_id,
            )

            if not conv_row:
                return None

            # Fetch messages for this conversation
            msg_rows = await self.pool.fetch(
                """
                SELECT id, conversation_id, role, content, components, created_at
                FROM conversation_messages
                WHERE conversation_id = $1
                ORDER BY created_at ASC
                """,
                conversation_id,
            )

            messages = [
                {
                    "id": str(msg["id"]),
                    "conversation_id": str(msg["conversation_id"]),
                    "role": msg["role"],
                    "content": msg["content"],
                    "components": msg["components"] or [],
                    "created_at": msg["created_at"].isoformat(),
                }
                for msg in msg_rows
            ]

            return {
                "id": str(conv_row["id"]),
                "title": conv_row["title"],
                "created_at": conv_row["created_at"].isoformat(),
                "updated_at": conv_row["updated_at"].isoformat(),
                "messages": messages,
            }
        except Exception as exc:
            log.error("get_conversation_failed", error=str(exc), conversation_id=conversation_id)
            raise

    async def create_conversation(self, conversation_id: str, title: str) -> dict:
        """Create a new conversation.

        Args:
            conversation_id: UUID for the conversation.
            title: Initial conversation title.

        Returns:
            Conversation dict with empty messages array.
        """
        if not self.pool:
            raise RuntimeError("PostgresConversationStore not connected")

        try:
            now = datetime.utcnow()
            row = await self.pool.fetchrow(
                """
                INSERT INTO conversations (id, title, created_at, updated_at)
                VALUES ($1, $2, $3, $4)
                RETURNING id, title, created_at, updated_at
                """,
                conversation_id,
                title,
                now,
                now,
            )

            return {
                "id": str(row["id"]),
                "title": row["title"],
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "messages": [],
            }
        except Exception as exc:
            log.error(
                "create_conversation_failed",
                error=str(exc),
                conversation_id=conversation_id,
                title=title,
            )
            raise

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        components: list | None = None,
    ) -> dict:
        """Add a message to a conversation and update the conversation's updated_at.

        Args:
            conversation_id: UUID of the conversation.
            role: Message role ('user' or 'assistant').
            content: Message text content.
            components: Optional list of UI component dicts.

        Returns:
            Message dict with id, role, content, components, created_at.
        """
        if not self.pool:
            raise RuntimeError("PostgresConversationStore not connected")

        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role: {role}")

        try:
            components_json = json.dumps(components or [])
            now = datetime.utcnow()

            async with self.pool.acquire() as conn:
                # Insert message
                msg_row = await conn.fetchrow(
                    """
                    INSERT INTO conversation_messages
                        (conversation_id, role, content, components, created_at)
                    VALUES ($1, $2, $3, $4::jsonb, $5)
                    RETURNING id, conversation_id, role, content, components, created_at
                    """,
                    conversation_id,
                    role,
                    content,
                    components_json,
                    now,
                )

                # Update conversation's updated_at timestamp
                await conn.execute(
                    """
                    UPDATE conversations
                    SET updated_at = $1
                    WHERE id = $2
                    """,
                    now,
                    conversation_id,
                )

            return {
                "id": str(msg_row["id"]),
                "conversation_id": str(msg_row["conversation_id"]),
                "role": msg_row["role"],
                "content": msg_row["content"],
                "components": msg_row["components"] or [],
                "created_at": msg_row["created_at"].isoformat(),
            }
        except Exception as exc:
            log.error(
                "add_message_failed",
                error=str(exc),
                conversation_id=conversation_id,
                role=role,
            )
            raise

    async def update_title(self, conversation_id: str, title: str) -> None:
        """Update the title of a conversation.

        Args:
            conversation_id: UUID of the conversation.
            title: New conversation title.
        """
        if not self.pool:
            raise RuntimeError("PostgresConversationStore not connected")

        try:
            now = datetime.utcnow()
            await self.pool.execute(
                """
                UPDATE conversations
                SET title = $1, updated_at = $2
                WHERE id = $3
                """,
                title,
                now,
                conversation_id,
            )
        except Exception as exc:
            log.error(
                "update_title_failed",
                error=str(exc),
                conversation_id=conversation_id,
            )
            raise

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all its messages (cascading).

        Args:
            conversation_id: UUID of the conversation.
        """
        if not self.pool:
            raise RuntimeError("PostgresConversationStore not connected")

        try:
            await self.pool.execute(
                """
                DELETE FROM conversations
                WHERE id = $1
                """,
                conversation_id,
            )
        except Exception as exc:
            log.error(
                "delete_conversation_failed",
                error=str(exc),
                conversation_id=conversation_id,
            )
            raise
