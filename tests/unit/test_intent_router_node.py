"""Unit tests for the intent router node."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage

from src.nodes.intent_router import make_intent_router_node
from src.schemas.intent import IntentResult, QueryPlanStep


@pytest.fixture
def mock_router_llm():
    """Create a mock LLM that returns structured IntentResult."""
    llm = MagicMock()
    structured = AsyncMock()
    llm.with_structured_output.return_value = structured
    return llm, structured


@pytest.mark.asyncio
class TestIntentRouterNode:
    async def test_classifies_data_query(self, mock_router_llm):
        llm, structured = mock_router_llm
        structured.ainvoke.return_value = IntentResult(
            intent="data_query",
            query_plan=[
                QueryPlanStep(
                    tool_name="get_salesforce_summary",
                    mcp_server="salesforce-mcp",
                    parameters={"account_name": "Acme"},
                    description="Fetch CRM data",
                ),
            ],
            reasoning="User asked for revenue data.",
        )

        node = make_intent_router_node(llm, bridges={})
        state = {
            "messages": [HumanMessage(content="Show me Q3 revenue for Acme Corp")],
            "turn_count": 0,
            "raw_data_context": {},
        }
        result = await node(state)

        assert result["intent"] == "data_query"
        assert len(result["query_plan"]) == 1
        assert result["turn_count"] == 1

    async def test_classifies_follow_up(self, mock_router_llm):
        llm, structured = mock_router_llm
        structured.ainvoke.return_value = IntentResult(
            intent="follow_up",
            reasoning="User is asking about existing data.",
        )

        node = make_intent_router_node(llm, bridges={})
        state = {
            "messages": [HumanMessage(content="Break that down by quarter")],
            "turn_count": 1,
            "raw_data_context": {"salesforce-mcp:get_salesforce_summary:0": {"data": "..."}},
        }
        result = await node(state)

        assert result["intent"] == "follow_up"
        assert result["query_plan"] == []

    async def test_handles_llm_error_with_data(self, mock_router_llm):
        llm, structured = mock_router_llm
        structured.ainvoke.side_effect = Exception("LLM timeout")

        node = make_intent_router_node(llm, bridges={})
        state = {
            "messages": [HumanMessage(content="What about EMEA?")],
            "turn_count": 1,
            "raw_data_context": {"some_key": "some_data"},
        }
        result = await node(state)

        # With existing data, should fallback to follow_up
        assert result["intent"] == "follow_up"
        assert len(result["errors"]) > 0

    async def test_handles_llm_error_without_data(self, mock_router_llm):
        llm, structured = mock_router_llm
        structured.ainvoke.side_effect = Exception("LLM timeout")

        node = make_intent_router_node(llm, bridges={})
        state = {
            "messages": [HumanMessage(content="Help")],
            "turn_count": 0,
            "raw_data_context": {},
        }
        result = await node(state)

        # Without data, should fallback to clarification
        assert result["intent"] == "clarification"
