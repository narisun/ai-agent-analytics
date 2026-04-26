"""Unit tests for FakeLLM, FakeLLMFactory, FakeMCPToolsProvider."""
from langchain_core.messages import AIMessage, HumanMessage

from src.domain.types import UserContext
from src.ports import LLMFactory, MCPToolsProvider
from tests.fakes.fake_llm import FakeLLM
from tests.fakes.fake_llm_factory import FakeLLMFactory
from tests.fakes.fake_mcp_tools_provider import FakeMCPToolsProvider


def test_fake_llm_factory_satisfies_protocol():
    assert isinstance(FakeLLMFactory(), LLMFactory)


def test_fake_mcp_tools_provider_satisfies_protocol():
    assert isinstance(FakeMCPToolsProvider(), MCPToolsProvider)


async def test_fake_llm_returns_canned_response():
    llm = FakeLLM(response="hello there")
    result = await llm.ainvoke([HumanMessage(content="hi")])
    assert isinstance(result, AIMessage)
    assert result.content == "hello there"


async def test_fake_llm_with_structured_output():
    from pydantic import BaseModel

    class Schema(BaseModel):
        answer: str

    llm = FakeLLM(structured_response=Schema(answer="42"))
    bound = llm.with_structured_output(Schema)
    result = await bound.ainvoke([HumanMessage(content="q?")])
    assert result.answer == "42"


async def test_fake_llm_records_calls():
    llm = FakeLLM(response="ok")
    await llm.ainvoke([HumanMessage(content="first")])
    await llm.ainvoke([HumanMessage(content="second")])
    assert len(llm.calls) == 2


async def test_fake_mcp_tools_provider_returns_injected_tools():
    from langchain_core.tools import tool

    @tool
    def dummy(x: int) -> int:
        """Dummy tool."""
        return x * 2

    provider = FakeMCPToolsProvider(tools=[dummy])
    ctx = UserContext(user_id="u", tenant_id="t", auth_token="tok")
    tools = await provider.get_langchain_tools(ctx)
    assert len(tools) == 1
    assert tools[0].name == "dummy"
