"""In-memory BaseChatModel-compatible fake for tests.

Does not subclass BaseChatModel because langchain's Runnable machinery
is overkill for tests. Instead provides the two methods ChatService
and nodes actually call: ainvoke() and with_structured_output().bind.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage


class _StructuredBound:
    def __init__(self, response: Any):
        self._response = response
        self.calls: list[list[BaseMessage]] = []

    async def ainvoke(self, messages: list[BaseMessage], config: Any = None) -> Any:
        self.calls.append(messages)
        return self._response

    def invoke(self, messages: list[BaseMessage], config: Any = None) -> Any:
        self.calls.append(messages)
        return self._response


class FakeLLM:
    """Configurable fake chat model.

    - response: content for plain ainvoke() calls.
    - structured_response: return value for with_structured_output(...).ainvoke().
    """

    def __init__(
        self,
        response: str = "",
        structured_response: Any = None,
    ):
        self._response = response
        self._structured_response = structured_response
        self.calls: list[list[BaseMessage]] = []

    async def ainvoke(self, messages: list[BaseMessage], config: Any = None) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self._response)

    def invoke(self, messages: list[BaseMessage], config: Any = None) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self._response)

    def with_structured_output(self, schema: Any) -> _StructuredBound:
        return _StructuredBound(self._structured_response)
