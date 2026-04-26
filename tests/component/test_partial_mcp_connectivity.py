"""Regression test: partial MCP connectivity at startup does NOT crash (P0).

Two layers of defence are tested:

1. ``_MCPBridgeToolsProvider`` (in src/graph.py) skips bridges whose
   ``is_connected`` is False — so a degraded set of bridges still
   yields a usable (possibly empty) tool list.

2. ``app.py``'s lifespan uses ``asyncio.gather(..., return_exceptions=True)``
   so a single bridge's connect() failure does not propagate.

Layer 1 is testable as a small component test; layer 2 requires a
running event-loop assertion which is exercised indirectly via the
existing application-tier tests.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

# _MCPBridgeToolsProvider is private to graph.py; access via module path.
from src.graph import _MCPBridgeToolsProvider


def _bridge(*, connected: bool, tools: list | None = None) -> MagicMock:
    bridge = MagicMock()
    bridge.is_connected = connected
    bridge.get_langchain_tools = AsyncMock(return_value=tools or [])
    return bridge


async def test_returns_empty_list_when_all_bridges_disconnected():
    bridges = {
        "data-mcp": _bridge(connected=False),
        "salesforce-mcp": _bridge(connected=False),
    }
    provider = _MCPBridgeToolsProvider(bridges)

    tools = await provider.get_langchain_tools(user_ctx=None)
    assert tools == []
    # Disconnected bridges' get_langchain_tools must NOT be called.
    bridges["data-mcp"].get_langchain_tools.assert_not_called()
    bridges["salesforce-mcp"].get_langchain_tools.assert_not_called()


async def test_returns_only_reachable_bridges_tools():
    """One healthy + one unreachable: lifespan succeeds with reduced tool set."""
    healthy_tool = MagicMock(name="healthy_tool")
    bridges = {
        "healthy": _bridge(connected=True, tools=[healthy_tool]),
        "down": _bridge(connected=False),
    }
    provider = _MCPBridgeToolsProvider(bridges)

    tools = await provider.get_langchain_tools(user_ctx=None)
    assert tools == [healthy_tool]
    bridges["healthy"].get_langchain_tools.assert_awaited_once()
    bridges["down"].get_langchain_tools.assert_not_called()


async def test_aggregates_tools_across_multiple_connected_bridges():
    t1, t2, t3 = MagicMock(), MagicMock(), MagicMock()
    bridges = {
        "data-mcp": _bridge(connected=True, tools=[t1]),
        "salesforce-mcp": _bridge(connected=True, tools=[t2, t3]),
    }
    provider = _MCPBridgeToolsProvider(bridges)

    tools = await provider.get_langchain_tools(user_ctx=None)
    # Order of aggregation isn't part of the contract, so use a set-based check.
    assert set(map(id, tools)) == set(map(id, [t1, t2, t3]))


async def test_handles_empty_bridges_dict():
    provider = _MCPBridgeToolsProvider({})
    tools = await provider.get_langchain_tools(user_ctx=None)
    assert tools == []


async def test_user_ctx_propagated_to_connected_bridges():
    """When user_ctx is non-None, it's forwarded to each connected bridge."""
    bridges = {"healthy": _bridge(connected=True)}
    provider = _MCPBridgeToolsProvider(bridges)

    fake_ctx = MagicMock(user_id="u1", auth_token="tok")
    await provider.get_langchain_tools(user_ctx=fake_ctx)
    bridges["healthy"].get_langchain_tools.assert_awaited_once_with(fake_ctx)
