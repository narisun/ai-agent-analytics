"""Unit tests verifying Protocol seams exist and are structurally satisfiable."""
from __future__ import annotations

from contextlib import contextmanager

from src.ports import (
    CompactionModifier,
    ConversationStore,
    LLMFactory,
    MCPToolsProvider,
    StreamEncoder,
    TelemetryScope,
)


class _FakeStore:
    async def save(self, convo, user_ctx): ...
    async def load(self, convo_id, user_ctx): return None
    async def list(self, user_ctx): return []
    async def delete(self, convo_id, user_ctx): ...


class _FakeToolsProvider:
    async def get_langchain_tools(self, user_ctx): return []


class _FakeLLMFactory:
    def make_router_llm(self): return None
    def make_synthesis_llm(self): return None


class _FakeEncoder:
    def encode_event(self, event): return b""
    def encode_error(self, err, *, error_id): return b""
    def finalize(self): return b""


class _FakeCompaction:
    def apply(self, messages): return messages


class _FakeSpan:
    def record_exception(self, err): ...
    def set_status(self, status): ...


class _FakeTelemetry:
    @contextmanager
    def start_span(self, name):
        yield _FakeSpan()

    def record_event(self, name, **attrs): ...


def test_conversation_store_protocol():
    assert isinstance(_FakeStore(), ConversationStore)


def test_mcp_tools_provider_protocol():
    assert isinstance(_FakeToolsProvider(), MCPToolsProvider)


def test_llm_factory_protocol():
    assert isinstance(_FakeLLMFactory(), LLMFactory)


def test_stream_encoder_protocol():
    assert isinstance(_FakeEncoder(), StreamEncoder)


def test_compaction_modifier_protocol():
    assert isinstance(_FakeCompaction(), CompactionModifier)


def test_telemetry_scope_protocol():
    assert isinstance(_FakeTelemetry(), TelemetryScope)
