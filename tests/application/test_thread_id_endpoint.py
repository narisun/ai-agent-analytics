"""Endpoint-level proof that LangGraph thread_id is tenant-scoped.

Builds AppDependencies with a stub graph and a chat_service_factory that
captures the run_config passed to graph.astream_events. Verifies that
two requests with the same conversation_id but different X-User-Email
headers arrive at the graph with distinct configurable.thread_id
values — Phase 0b's tenant-isolation guarantee, validated through the
new SoC/DI structure.
"""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

# Set INTERNAL_API_KEY so any module that reads it at import time
# does not surface unrelated config errors.
os.environ.setdefault("INTERNAL_API_KEY", "test-internal-api-key")

from src.app import create_app  # noqa: E402
from src.app_dependencies import AppDependencies  # noqa: E402
from src.domain.types import UserContext  # noqa: E402
from src.services.chat_service import ChatService  # noqa: E402
from tests.fakes.fake_conversation_store import FakeConversationStore  # noqa: E402
from tests.fakes.fake_stream_encoder import FakeStreamEncoder  # noqa: E402
from tests.fakes.fake_telemetry import FakeTelemetryScope  # noqa: E402


@pytest.fixture
def captured_configs():
    """List that the fake graph appends every received `config` dict to."""
    return []


@pytest.fixture
def client(captured_configs):
    """TestClient over create_app(deps) with stub graph and ChatService factory."""

    async def fake_astream_events(state, config, version):
        captured_configs.append(config)
        if False:
            yield  # pragma: no cover

    fake_graph = MagicMock()
    fake_graph.astream_events = fake_astream_events

    store = FakeConversationStore()
    # Existing dashboard endpoints expect the legacy add_message/etc methods;
    # the test's path doesn't actually call those (FakeConversationStore has
    # them as protocol methods only), so we only need add_message to exist.
    store.add_message = AsyncMock(return_value=None)
    store.get_conversation = AsyncMock(return_value=None)
    store.create_conversation = AsyncMock(return_value=None)

    config = MagicMock()

    def chat_service_factory(user_ctx: UserContext) -> ChatService:
        return ChatService(
            graph=fake_graph,
            conversation_store=store,
            config=config,
            user_ctx=user_ctx,
            encoder_factory=lambda: FakeStreamEncoder(),
            telemetry=FakeTelemetryScope(),
        )

    deps = AppDependencies(
        config=config,
        graph=fake_graph,
        conversation_store=store,
        mcp_tools_provider=None,
        llm_factory=None,
        telemetry=FakeTelemetryScope(),
        compaction=None,
        encoder_factory=lambda: FakeStreamEncoder(),
        chat_service_factory=chat_service_factory,
    )

    app = create_app(deps)
    yield TestClient(app)


class TestChatEndpointThreadIdIsolation:
    """Cross-user isolation contract for POST /api/v1/analytics/chat."""

    def test_thread_id_includes_user_email(self, client, captured_configs):
        payload = {
            "id": "shared-session-uuid",
            "messages": [{"role": "user", "content": "hi"}],
        }
        client.post(
            "/api/v1/analytics/chat",
            json=payload,
            headers={
                "Authorization": "Bearer test-token",
                "X-User-Email": "alice@example.com",
                "X-User-Role": "manager",
            },
        )
        assert len(captured_configs) == 1
        thread_id = captured_configs[0]["configurable"]["thread_id"]
        assert "alice@example.com" in thread_id
        assert "shared-session-uuid" in thread_id

    def test_two_users_with_same_session_id_get_distinct_thread_ids(self, client, captured_configs):
        payload = {
            "id": "shared-session-uuid",
            "messages": [{"role": "user", "content": "hi"}],
        }
        for email in ("alice@example.com", "bob@example.com"):
            client.post(
                "/api/v1/analytics/chat",
                json=payload,
                headers={
                    "Authorization": "Bearer test-token",
                    "X-User-Email": email,
                    "X-User-Role": "manager",
                },
            )
        assert len(captured_configs) == 2
        thread_ids = [c["configurable"]["thread_id"] for c in captured_configs]
        assert thread_ids[0] != thread_ids[1]
        assert "alice@example.com" in thread_ids[0]
        assert "bob@example.com" in thread_ids[1]


class TestStreamEndpointThreadIdIsolation:
    """Cross-user isolation contract for legacy POST /api/v1/analytics/stream.

    The legacy endpoint uses anonymous fallback when X-User-Email is absent;
    when present, the email feeds make_thread_id and produces tenant-scoped
    thread_ids the same as the /chat path.
    """

    def test_thread_id_includes_user_email(self, client, captured_configs):
        payload = {"session_id": "shared-session-uuid", "message": "hi"}
        client.post(
            "/api/v1/analytics/stream",
            json=payload,
            headers={
                "Authorization": "Bearer test-token",
                "X-User-Email": "alice@example.com",
                "X-User-Role": "manager",
            },
        )
        assert len(captured_configs) == 1
        thread_id = captured_configs[0]["configurable"]["thread_id"]
        assert "alice@example.com" in thread_id
        assert "shared-session-uuid" in thread_id

    def test_two_users_with_same_session_id_get_distinct_thread_ids(self, client, captured_configs):
        payload = {"session_id": "shared-session-uuid", "message": "hi"}
        for email in ("alice@example.com", "bob@example.com"):
            client.post(
                "/api/v1/analytics/stream",
                json=payload,
                headers={
                    "Authorization": "Bearer test-token",
                    "X-User-Email": email,
                    "X-User-Role": "manager",
                },
            )
        assert len(captured_configs) == 2
        thread_ids = [c["configurable"]["thread_id"] for c in captured_configs]
        assert thread_ids[0] != thread_ids[1]
        assert "alice@example.com" in thread_ids[0]
        assert "bob@example.com" in thread_ids[1]
