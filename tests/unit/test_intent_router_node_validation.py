"""P0 regression: unknown intents from the LLM route to error_handler, not a crash."""
from langchain_core.messages import HumanMessage

from src.domain.types import UserContext
from src.nodes.intent_router import IntentRouterNode, route_after_intent
from tests.fakes.fake_compaction import FakeCompactionModifier
from tests.fakes.fake_llm import FakeLLM
from tests.fakes.fake_mcp_tools_provider import FakeMCPToolsProvider


class _BogusIntentResult:
    """Mimics the IntentResult schema shape but with an invalid intent string."""
    intent = "bogus_intent"
    query_plan: list = []
    intent_reasoning = "LLM hallucinated a non-canonical intent."


async def test_unknown_intent_is_rewritten_to_clarification():
    # FakeLLM.with_structured_output(...).ainvoke() returns the configured object.
    llm = FakeLLM(structured_response=_BogusIntentResult())

    node = IntentRouterNode(
        llm=llm,
        tools_provider=FakeMCPToolsProvider(),
        prompts=None,
        compaction=FakeCompactionModifier(),
    )

    ctx = UserContext(user_id="u", tenant_id="t", auth_token="tok")
    state = {"messages": [HumanMessage(content="help")], "session_id": "s1", "turn_count": 0}
    config = {"configurable": {"user_ctx": ctx}}

    result = await node(state, config)

    assert result.get("intent") == "clarification", (
        "Unknown intents must be rewritten to 'clarification' (P0 fix)"
    )
    assert "bogus_intent" in result.get("intent_reasoning", ""), (
        "Reasoning must reference the original bogus value for debuggability"
    )
    assert result.get("query_plan", None) == [], (
        "An invalid intent must NOT carry through any query plan"
    )


async def test_route_after_intent_routes_clarification_to_error_handler():
    """Confirms the existing routing function — should already pass."""
    state = {"intent": "clarification"}
    assert route_after_intent(state) == "error_handler"


async def test_unknown_intent_full_chain_routes_to_error_handler():
    """End-to-end of the P0 fix: bogus LLM intent → clarification → error_handler."""
    llm = FakeLLM(structured_response=_BogusIntentResult())
    node = IntentRouterNode(
        llm=llm,
        tools_provider=FakeMCPToolsProvider(),
        prompts=None,
        compaction=FakeCompactionModifier(),
    )

    ctx = UserContext(user_id="u", tenant_id="t", auth_token="tok")
    state = {"messages": [HumanMessage(content="help")], "session_id": "s1", "turn_count": 0}
    config = {"configurable": {"user_ctx": ctx}}

    result = await node(state, config)
    final_state = {**state, **result}
    assert route_after_intent(final_state) == "error_handler"
