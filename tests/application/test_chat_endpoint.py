"""Application-tier tests for POST /api/v1/analytics/chat."""
import os
os.environ.setdefault("INTERNAL_API_KEY", "test-internal-api-key")

from contextlib import contextmanager  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.app import create_app  # noqa: E402
from src.app_dependencies import AppDependencies  # noqa: E402
from src.domain.types import UserContext  # noqa: E402
from src.services.chat_service import ChatService  # noqa: E402
from tests.fakes.fake_conversation_store import FakeConversationStore  # noqa: E402
from tests.fakes.fake_stream_encoder import FakeStreamEncoder  # noqa: E402
from tests.fakes.fake_telemetry import FakeTelemetryScope  # noqa: E402


def _stub_graph():
    g = MagicMock()

    async def _astream(*args, **kwargs):
        return
        yield  # unreachable
    g.astream_events = _astream
    return g


def _build_app():
    graph = _stub_graph()
    store = FakeConversationStore()
    store.add_message = AsyncMock(return_value=None)
    store.get_conversation = AsyncMock(return_value=None)
    store.create_conversation = AsyncMock(return_value=None)

    def factory(user_ctx: UserContext) -> ChatService:
        return ChatService(
            graph=graph,
            conversation_store=store,
            config=MagicMock(),
            user_ctx=user_ctx,
            encoder_factory=lambda: FakeStreamEncoder(),
            telemetry=FakeTelemetryScope(),
        )

    deps = AppDependencies(
        config=MagicMock(),
        graph=graph,
        conversation_store=store,
        mcp_tools_provider=None,
        llm_factory=None,
        telemetry=FakeTelemetryScope(),
        compaction=None,
        encoder_factory=lambda: FakeStreamEncoder(),
        chat_service_factory=factory,
    )
    return create_app(deps)


def test_200_with_auth_headers():
    client = TestClient(_build_app())
    resp = client.post(
        "/api/v1/analytics/chat",
        headers={
            "X-User-Email": "alice@example.com",
            "X-User-Role": "manager",
            "Authorization": "Bearer tok",
        },
        json={"id": "c1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200, resp.text


def test_401_without_user_email_header():
    client = TestClient(_build_app())
    resp = client.post(
        "/api/v1/analytics/chat",
        json={"id": "c1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 401, resp.text


def test_422_on_no_user_message():
    client = TestClient(_build_app())
    resp = client.post(
        "/api/v1/analytics/chat",
        headers={"X-User-Email": "alice@example.com"},
        json={"id": "c1", "messages": [{"role": "assistant", "content": "no user msg"}]},
    )
    assert resp.status_code == 422, resp.text
