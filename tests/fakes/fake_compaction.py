"""FakeCompactionModifier — passthrough by default."""
from __future__ import annotations

from langchain_core.messages import BaseMessage


class FakeCompactionModifier:
    def __init__(self, passthrough: bool = True) -> None:
        self._passthrough = passthrough

    def apply(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        return list(messages) if self._passthrough else []
