"""Component tests for ConversationService — wraps ConversationStore behind a
service that takes UserContext per call (not in constructor) since one service
instance handles requests from many tenants."""
import pytest

from src.domain.errors import ConversationNotFound
from src.domain.types import Conversation, UserContext
from src.services.conversation_service import ConversationService
from tests.fakes.fake_conversation_store import FakeConversationStore


@pytest.fixture
def store():
    return FakeConversationStore()


@pytest.fixture
def alice():
    return UserContext(user_id="alice@example.com", tenant_id="default", auth_token="tok")


@pytest.fixture
def bob():
    return UserContext(user_id="bob@example.com", tenant_id="default", auth_token="tok")


async def test_save_then_get(store, alice):
    svc = ConversationService(store=store)
    convo = Conversation(conversation_id="c1", title="Hello", updated_at="2026-04-16T00:00:00Z")
    await svc.save(convo, alice)

    got = await svc.get("c1", alice)
    assert got.conversation_id == "c1"
    assert got.title == "Hello"


async def test_get_missing_raises_not_found(store, alice):
    svc = ConversationService(store=store)
    with pytest.raises(ConversationNotFound) as exc:
        await svc.get("nope", alice)
    assert exc.value.conversation_id == "nope"


async def test_list_returns_only_callers_tenant(store, alice, bob):
    svc = ConversationService(store=store)
    await svc.save(Conversation(conversation_id="A1", title="Alice's", updated_at="2026-04-16T00:00:00Z"), alice)
    await svc.save(Conversation(conversation_id="B1", title="Bob's", updated_at="2026-04-16T00:00:00Z"), bob)

    # FakeConversationStore scopes by tenant_id. alice and bob share tenant_id="default"
    # in this fixture, so each sees both. Use distinct tenants to verify scoping.
    a_only = UserContext(user_id="alice@example.com", tenant_id="TENANT_A", auth_token="tok")
    b_only = UserContext(user_id="bob@example.com", tenant_id="TENANT_B", auth_token="tok")
    await svc.save(Conversation(conversation_id="X1", title="Tenant A only", updated_at="2026-04-16T00:00:00Z"), a_only)
    await svc.save(Conversation(conversation_id="Y1", title="Tenant B only", updated_at="2026-04-16T00:00:00Z"), b_only)

    a_list = await svc.list(a_only)
    b_list = await svc.list(b_only)

    a_ids = {s.conversation_id for s in a_list}
    b_ids = {s.conversation_id for s in b_list}
    assert "X1" in a_ids and "Y1" not in a_ids
    assert "Y1" in b_ids and "X1" not in b_ids


async def test_delete_removes_conversation(store, alice):
    svc = ConversationService(store=store)
    convo = Conversation(conversation_id="c1", title="x", updated_at="2026-04-16T00:00:00Z")
    await svc.save(convo, alice)

    await svc.delete("c1", alice)
    with pytest.raises(ConversationNotFound):
        await svc.get("c1", alice)


async def test_delete_missing_is_idempotent(store, alice):
    """Deleting a non-existent conversation must not raise — idempotent semantics."""
    svc = ConversationService(store=store)
    # Should not raise.
    await svc.delete("never-existed", alice)
