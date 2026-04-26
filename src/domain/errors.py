"""Domain-owned exception hierarchy.

Infrastructure adapters (platform-sdk) catch vendor exceptions and re-raise
as one of these types. Domain and service code only ever sees this tree.
"""
from __future__ import annotations


class AnalyticsError(Exception):
    """Base for all analytics-agent domain errors."""


class AuthError(AnalyticsError):
    """Auth context invalid, missing, or denied by OPA."""


class IntentError(AnalyticsError):
    """Base for intent-classification problems."""


class UnknownIntent(IntentError):
    """Router LLM returned an intent the graph cannot route. (P0 fix)"""

    def __init__(self, intent: str) -> None:
        self.intent = intent
        super().__init__(f"Unknown intent from router: {intent!r}")


class ToolsError(AnalyticsError):
    """Base for MCP-tool problems."""


class ToolsUnavailable(ToolsError):
    """No MCP bridges are reachable."""


class UnsupportedSchemaError(ToolsError):
    """MCP tool schema uses unsupported JSON Schema keyword. (P0 fix)"""

    def __init__(self, tool_name: str, keyword: str) -> None:
        self.tool_name = tool_name
        self.keyword = keyword
        super().__init__(
            f"MCP tool {tool_name!r} schema uses unsupported keyword {keyword!r}"
        )


class LLMError(AnalyticsError):
    """Base for LLM-layer problems."""


class LLMUnavailable(LLMError):
    """LLM provider could not be reached."""


class LLMStructuredOutputError(LLMError):
    """LLM returned output that failed schema validation."""


class StoreError(AnalyticsError):
    """Base for conversation-store problems."""


class StoreUnavailable(StoreError):
    """Conversation store could not be reached."""


class ConversationNotFound(StoreError):
    """Requested conversation does not exist."""

    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        super().__init__(f"Conversation not found: {conversation_id!r}")
