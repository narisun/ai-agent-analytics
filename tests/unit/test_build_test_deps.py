"""Unit tests for build_test_dependencies factory."""
from tests.fakes.build_test_deps import build_test_dependencies
from tests.fakes.fake_conversation_store import FakeConversationStore
from tests.fakes.fake_llm_factory import FakeLLMFactory


def test_returns_populated_deps_by_default():
    deps = build_test_dependencies()
    assert deps.conversation_store is not None
    assert deps.mcp_tools_provider is not None
    assert deps.llm_factory is not None
    assert deps.telemetry is not None
    assert deps.compaction is not None


def test_override_individual_fake():
    custom_store = FakeConversationStore()
    deps = build_test_dependencies(conversation_store=custom_store)
    assert deps.conversation_store is custom_store


def test_override_llm_factory():
    custom_factory = FakeLLMFactory()
    deps = build_test_dependencies(llm_factory=custom_factory)
    assert deps.llm_factory is custom_factory
