"""Unit tests for domain pydantic models."""
import pytest
from pydantic import ValidationError

from src.domain.types import (
    ChatRequest,
    ChatResponse,
    Conversation,
    ConversationSummary,
)


class TestChatRequest:
    def test_parses_minimal_body(self):
        req = ChatRequest(message="hello")
        assert req.message == "hello"
        assert req.conversation_id is None
        assert req.history == []

    def test_rejects_empty_message(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_accepts_history(self):
        req = ChatRequest(
            message="hi",
            history=[{"role": "user", "content": "prev"}],
        )
        assert req.history[0]["role"] == "user"


class TestConversation:
    def test_summary_has_title_and_id(self):
        s = ConversationSummary(conversation_id="c1", title="Hello", updated_at="2026-04-16T00:00:00Z")
        assert s.conversation_id == "c1"


class TestChatResponse:
    def test_shape(self):
        r = ChatResponse(conversation_id="c1", narrative="n", components=[])
        assert r.narrative == "n"
