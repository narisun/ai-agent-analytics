"""Unit tests for the domain error hierarchy."""
import pytest

from src.domain.errors import (
    AnalyticsError,
    AuthError,
    ConversationNotFound,
    IntentError,
    LLMError,
    LLMStructuredOutputError,
    LLMUnavailable,
    StoreError,
    StoreUnavailable,
    ToolsError,
    ToolsUnavailable,
    UnknownIntent,
    UnsupportedSchemaError,
)


def test_all_subclass_analytics_error():
    for cls in [
        AuthError,
        IntentError,
        UnknownIntent,
        ToolsError,
        ToolsUnavailable,
        UnsupportedSchemaError,
        LLMError,
        LLMUnavailable,
        LLMStructuredOutputError,
        StoreError,
        StoreUnavailable,
        ConversationNotFound,
    ]:
        assert issubclass(cls, AnalyticsError), f"{cls.__name__} is not an AnalyticsError"


def test_unknown_intent_carries_intent_string():
    err = UnknownIntent(intent="bogus")
    assert err.intent == "bogus"
    assert "bogus" in str(err)


def test_unsupported_schema_carries_tool_and_keyword():
    err = UnsupportedSchemaError(tool_name="fetch_x", keyword="$ref")
    assert err.tool_name == "fetch_x"
    assert err.keyword == "$ref"
    assert "fetch_x" in str(err)
    assert "$ref" in str(err)


def test_conversation_not_found_carries_id():
    err = ConversationNotFound(conversation_id="c-123")
    assert err.conversation_id == "c-123"


def test_raising_subclass_catchable_as_base():
    with pytest.raises(AnalyticsError):
        raise UnknownIntent(intent="x")
