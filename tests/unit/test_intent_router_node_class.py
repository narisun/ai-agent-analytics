"""IntentRouterNode is a callable class with constructor injection."""
import inspect

from src.nodes.intent_router import IntentRouterNode
from tests.fakes.fake_compaction import FakeCompactionModifier
from tests.fakes.fake_llm import FakeLLM
from tests.fakes.fake_mcp_tools_provider import FakeMCPToolsProvider


def test_constructor_takes_named_deps():
    sig = inspect.signature(IntentRouterNode.__init__)
    names = {p.name for p in sig.parameters.values() if p.name != "self"}
    assert {"llm", "tools_provider", "prompts", "compaction"} <= names


def test_is_async_callable():
    node = IntentRouterNode(
        llm=FakeLLM(),
        tools_provider=FakeMCPToolsProvider(),
        prompts=None,
        compaction=FakeCompactionModifier(),
    )
    assert callable(node)
    assert inspect.iscoroutinefunction(node.__call__)
