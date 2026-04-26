"""Pre-stream vs. mid-stream error contract.

A pre-stream failure (auth, body validation) must surface as an HTTP
4xx. A mid-stream failure (after streaming has started) must yield an
encoded error event in a 200 stream — never a 500.
"""
import os
os.environ.setdefault("INTERNAL_API_KEY", "test-internal-api-key")

from contextlib import contextmanager  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from src.app import create_app  # noqa: E402
from src.app_dependencies import AppDependencies  # noqa: E402
from src.domain.errors import ToolsUnavailable  # noqa: E402
from src.domain.types import UserContext  # noqa: E402
from src.services.chat_service import ChatService  # noqa: E402
from tests.fakes.fake_conversation_store import FakeConversationStore  # noqa: E402
from tests.fakes.fake_stream_encoder import FakeStreamEncoder  # noqa: E402
from tests.fakes.fake_telemetry import FakeTelemetryScope  # noqa: E402


def _graph_that_raises_midstream():
    g = MagicMock()

    async def _stream(*a, **k):
        yield {"event": "on_chain_start", "name": "intent_router"}
        raise ToolsUnavailable("all MCP bridges unreachable")
    g.astream_events = _stream
    return g


def _build_app_with_failing_graph():
    graph = _graph_that_raises_midstream()
    store = FakeConversationStore()
    store.add_message = AsyncMock(return_value=None)
    store.get_conversation = AsyncMock(return_value=None)
    store.create_conversation = AsyncMock(return_value=None)

    def factory(ctx: UserContext) -> ChatService:
        return ChatService(
            graph=graph,
            conversation_store=store,
            config=MagicMock(),
            user_ctx=ctx,
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


def test_pre_stream_auth_error_returns_401():
    """Missing X-User-Email is a pre-stream auth failure → HTTP 401, not 200 stream."""
    client = TestClient(_build_app_with_failing_graph())
    resp = client.post(
        "/api/v1/analytics/chat",
        json={"id": "c1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 401
    body = resp.json()
    assert body.get("type") == "auth"
    assert body.get("error_id")  # non-empty


def test_mid_stream_error_returns_200_with_encoded_error():
    """Once streaming has started, mid-stream errors return 200 with an
    encoded error event in the body — never an HTTP 500."""
    client = TestClient(_build_app_with_failing_graph())
    resp = client.post(
        "/api/v1/analytics/chat",
        headers={"X-User-Email": "alice@example.com"},
        json={"id": "c1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.content
    # The fake stream encoder serialises errors as JSON with {"error": "..."}
    assert b"error" in body.lower()
