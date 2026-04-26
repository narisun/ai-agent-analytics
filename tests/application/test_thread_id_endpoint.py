"""Endpoint-level proof that LangGraph thread_id is tenant-scoped.

Bypasses the FastAPI lifespan by injecting a fake graph and a fake
conversation_store directly into `app.state` (same pattern as
`test_e2e_streaming.py::app_with_mocks`), overrides auth and rate-limit
dependencies, and asserts that two requests with the same session_id
but different X-User-Email values arrive at the graph with distinct
`configurable.thread_id` values.
"""
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

# IMPORTANT: set INTERNAL_API_KEY before importing app.py — module-level
# `make_api_key_verifier()` reads the env at request time but the
# dependency_overrides bypass the verifier entirely. We still set it so
# importing app.py doesn't surface unrelated config errors.
os.environ.setdefault("INTERNAL_API_KEY", "test-internal-api-key")

from src.app import app, verify_api_key, check_rate_limit  # noqa: E402


@pytest.fixture
def captured_configs():
    """List that the fake graph appends every received `config` dict to."""
    return []


@pytest.fixture
def client(captured_configs):
    """TestClient with a stub graph + store and overridden auth/rate-limit deps.

    Uses `app.state.X =` injection (no lifespan), exactly as
    `test_e2e_streaming.py::app_with_mocks` does. The /chat endpoint
    reads `app.state.conversation_store` mid-stream to persist messages,
    so we stub it with AsyncMocks. The /stream endpoint does not
    persist, but the same fixture serves both. After the test, all
    overrides are removed so subsequent tests see a clean app.
    """

    async def fake_astream_events(state, config, version):
        captured_configs.append(config)
        # Async generator that yields nothing — the endpoint code path
        # finishes the stream cleanly with no events.
        if False:
            yield  # pragma: no cover

    fake_graph = MagicMock()
    fake_graph.astream_events = fake_astream_events

    fake_store = MagicMock()
    fake_store.get_conversation = AsyncMock(return_value=None)
    fake_store.create_conversation = AsyncMock(return_value=None)
    fake_store.add_message = AsyncMock(return_value=None)

    app.state.graph = fake_graph
    app.state.conversation_store = fake_store

    app.dependency_overrides[verify_api_key] = lambda: "test-token"
    app.dependency_overrides[check_rate_limit] = lambda: None

    yield TestClient(app)

    app.dependency_overrides.clear()


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

    def test_two_users_with_same_session_id_get_distinct_thread_ids(
        self, client, captured_configs
    ):
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
        assert thread_ids[0] != thread_ids[1], (
            "Two distinct users with the same session_id MUST get "
            "distinct thread_ids — this is the tenant-isolation guarantee."
        )
        assert "alice@example.com" in thread_ids[0]
        assert "bob@example.com" in thread_ids[1]


class TestStreamEndpointThreadIdIsolation:
    """Cross-user isolation contract for POST /api/v1/analytics/stream.

    The legacy SSE endpoint must mirror /chat's tenant-scoping. Today
    /stream does not even read X-User-Email — these tests assert that
    after Task 3 it does, and that thread_id is namespaced the same way.
    """

    def test_thread_id_includes_user_email(self, client, captured_configs):
        payload = {
            "session_id": "shared-session-uuid",
            "message": "hi",
        }
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

    def test_two_users_with_same_session_id_get_distinct_thread_ids(
        self, client, captured_configs
    ):
        payload = {
            "session_id": "shared-session-uuid",
            "message": "hi",
        }
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
        assert thread_ids[0] != thread_ids[1], (
            "Same session_id with different X-User-Email MUST get "
            "different thread_ids on /stream too — same contract as /chat."
        )
        assert "alice@example.com" in thread_ids[0]
        assert "bob@example.com" in thread_ids[1]
