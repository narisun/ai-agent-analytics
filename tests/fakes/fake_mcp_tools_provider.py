"""FakeMCPToolsProvider — returns a preconfigured list of langchain tools."""
from __future__ import annotations

from langchain_core.tools import BaseTool

from src.domain.types import UserContext


class FakeMCPToolsProvider:
    def __init__(self, tools: list[BaseTool] | None = None):
        self._tools = list(tools or [])
        self.calls: list[UserContext] = []

    async def get_langchain_tools(self, user_ctx: UserContext) -> list[BaseTool]:
        self.calls.append(user_ctx)
        return list(self._tools)
