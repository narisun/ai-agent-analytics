"""Unit tests for FakeConversationStore."""
import pytest

from src.domain.types import Conversation, UserContext
from src.ports import ConversationStore
from tests.fakes.fake_conversation_store import FakeConversationStore


@pytest.fixture
def ctx():
    return UserContext(user_id="u1", tenant_id="t1", auth_token="tok")


async def test_satisfies_protocol():
    assert isinstance(FakeConversationStore(), ConversationStore)


async def test_save_then_load(ctx):
    store = FakeConversationStore()
    convo = Conversation(conversation_id="c1", title="hi", updated_at="2026-04-16T00:00:00Z")
    await store.save(convo, ctx)
    loaded = await store.load("c1", ctx)
    assert loaded is not None
    assert loaded.conversation_id == "c1"


async def test_load_missing_returns_none(ctx):
    store = FakeConversationStore()
    assert await store.load("nope", ctx) is None


async def test_list_tenant_scoped(ctx):
    store = FakeConversationStore()
    c = Conversation(conversation_id="c1", title="x", updated_at="2026-04-16T00:00:00Z")
    await store.save(c, ctx)
    other_ctx = UserContext(user_id="u2", tenant_id="OTHER", auth_token="tok")
    listed = await store.list(other_ctx)
    assert listed == []


async def test_delete(ctx):
    store = FakeConversationStore()
    c = Conversation(conversation_id="c1", title="x", updated_at="2026-04-16T00:00:00Z")
    await store.save(c, ctx)
    await store.delete("c1", ctx)
    assert await store.load("c1", ctx) is None
