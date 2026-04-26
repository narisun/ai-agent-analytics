"""Application-tier tests for /api/v1/conversations CRUD."""
import os
os.environ.setdefault("INTERNAL_API_KEY", "test-internal-api-key")

from unittest.mock import AsyncMock  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.app import create_app  # noqa: E402
from src.app_dependencies import AppDependencies  # noqa: E402
from tests.fakes.fake_conversation_store import FakeConversationStore  # noqa: E402


def _build_app(legacy_store):
    deps = AppDependencies(
        config=None,
        graph=None,
        conversation_store=legacy_store,
        mcp_tools_provider=None,
        llm_factory=None,
        telemetry=None,
        compaction=None,
        encoder_factory=None,
        chat_service_factory=None,
    )
    return create_app(deps)


def _make_legacy_store(list_value=None, get_value=None, create_value=None):
    store = FakeConversationStore()
    store.list_conversations = AsyncMock(return_value=list_value or [])
    store.get_conversation = AsyncMock(return_value=get_value)
    store.create_conversation = AsyncMock(return_value=create_value or {"id": "c1", "title": "x"})
    store.update_title = AsyncMock(return_value=None)
    store.delete_conversation = AsyncMock(return_value=None)
    return store


def test_list_conversations_200_with_auth():
    store = _make_legacy_store(list_value=[{"id": "c1", "title": "Hello"}])
    client = TestClient(_build_app(store))
    resp = client.get(
        "/api/v1/conversations",
        headers={"X-User-Email": "alice@example.com", "Authorization": "Bearer tok"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "c1" in str(body)


def test_list_conversations_401_without_auth():
    store = _make_legacy_store()
    client = TestClient(_build_app(store))
    resp = client.get("/api/v1/conversations")
    assert resp.status_code == 401


def test_get_conversation_404_when_missing():
    store = _make_legacy_store(get_value=None)
    client = TestClient(_build_app(store))
    resp = client.get(
        "/api/v1/conversations/nope",
        headers={"X-User-Email": "alice@example.com"},
    )
    assert resp.status_code == 404


def test_get_conversation_200_when_present():
    store = _make_legacy_store(get_value={"id": "c1", "title": "Hello", "messages": []})
    client = TestClient(_build_app(store))
    resp = client.get(
        "/api/v1/conversations/c1",
        headers={"X-User-Email": "alice@example.com"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "c1"


def test_create_conversation_201():
    store = _make_legacy_store(create_value={"id": "c2", "title": "New"})
    client = TestClient(_build_app(store))
    resp = client.post(
        "/api/v1/conversations",
        headers={"X-User-Email": "alice@example.com"},
        json={"id": "c2", "title": "New"},
    )
    assert resp.status_code == 201


def test_delete_conversation_204():
    store = _make_legacy_store()
    client = TestClient(_build_app(store))
    resp = client.delete(
        "/api/v1/conversations/c1",
        headers={"X-User-Email": "alice@example.com"},
    )
    assert resp.status_code == 204
