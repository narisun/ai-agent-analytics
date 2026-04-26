"""Auth header plumbing test.

Verifies authentication is enforced at the boundary by _user_ctx_from.
Tenant scoping at the persistence layer is intentionally NOT tested
here — the legacy concrete stores do not yet honour UserContext
(see routes/conversations.py docstring). Phase 9 will add tenant
filters to the stores; this test will then be tightened.
"""
import os
os.environ.setdefault("INTERNAL_API_KEY", "test-internal-api-key")

from unittest.mock import AsyncMock  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from src.app import create_app  # noqa: E402
from src.app_dependencies import AppDependencies  # noqa: E402
from tests.fakes.fake_conversation_store import FakeConversationStore  # noqa: E402


def _build_app(store):
    deps = AppDependencies(
        config=None, graph=None, conversation_store=store,
        mcp_tools_provider=None, llm_factory=None, telemetry=None,
        compaction=None, encoder_factory=None, chat_service_factory=None,
    )
    return create_app(deps)


def test_unauthenticated_request_returns_401():
    store = FakeConversationStore()
    store.list_conversations = AsyncMock(return_value=[])
    client = TestClient(_build_app(store))
    resp = client.get("/api/v1/conversations")
    assert resp.status_code == 401


def test_authenticated_request_returns_200():
    store = FakeConversationStore()
    store.list_conversations = AsyncMock(return_value=[{"id": "c1"}])
    client = TestClient(_build_app(store))
    resp = client.get(
        "/api/v1/conversations",
        headers={"X-User-Email": "alice@example.com"},
    )
    assert resp.status_code == 200


def test_invalid_authorization_format_still_authenticates_via_email_header():
    """Authorization Bearer is OPTIONAL — only X-User-Email is required.
    A malformed Authorization header does NOT block authentication."""
    store = FakeConversationStore()
    store.list_conversations = AsyncMock(return_value=[])
    client = TestClient(_build_app(store))
    resp = client.get(
        "/api/v1/conversations",
        headers={
            "X-User-Email": "alice@example.com",
            "Authorization": "Basic abc",  # not Bearer
        },
    )
    assert resp.status_code == 200
