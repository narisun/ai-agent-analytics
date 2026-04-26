"""MCPToolCallerNode is a callable class with constructor injection."""
import inspect

from src.nodes.mcp_tool_caller import MCPToolCallerNode
from tests.fakes.fake_mcp_tools_provider import FakeMCPToolsProvider


def test_constructor_takes_tools_provider():
    sig = inspect.signature(MCPToolCallerNode.__init__)
    names = {p.name for p in sig.parameters.values() if p.name != "self"}
    assert {"tools_provider"} <= names


def test_is_async_callable():
    node = MCPToolCallerNode(tools_provider=FakeMCPToolsProvider())
    assert callable(node)
    assert inspect.iscoroutinefunction(node.__call__)
