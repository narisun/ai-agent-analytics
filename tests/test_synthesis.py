"""Unit tests for the synthesis node."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage

from src.nodes.synthesis import make_synthesis_node
from src.schemas.ui_components import AnalyticsResponse, UIComponent, ChartMetadata, ChartDataPoint


@pytest.fixture
def mock_synthesis_llm():
    """Create a mock LLM that returns structured AnalyticsResponse."""
    llm = MagicMock()
    structured = AsyncMock()
    llm.with_structured_output.return_value = structured
    return llm, structured


@pytest.mark.asyncio
class TestSynthesisNode:
    async def test_generates_narrative_and_components(self, mock_synthesis_llm):
        llm, structured = mock_synthesis_llm
        structured.ainvoke.return_value = AnalyticsResponse(
            narrative="Revenue grew 15% in Q3, driven by North America.",
            components=[
                UIComponent(
                    component_type="BarChart",
                    metadata=ChartMetadata(
                        title="Q3 Revenue by Region",
                        source="salesforce-mcp",
                        confidence_score=0.95,
                        format_hint="currency",
                    ),
                    data=[
                        ChartDataPoint(category="North America", value=1200000),
                        ChartDataPoint(category="EMEA", value=800000),
                    ],
                ),
            ],
        )

        node = make_synthesis_node(llm)
        state = {
            "messages": [HumanMessage(content="Show me Q3 revenue by region")],
            "raw_data_context": {"salesforce-mcp:get_salesforce_summary:0": {"revenue": {"NA": 1200000}}},
            "errors": [],
        }
        result = await node(state)

        assert "15%" in result["narrative"]
        assert len(result["ui_components"]) == 1
        assert result["ui_components"][0]["component_type"] == "BarChart"

    async def test_handles_synthesis_error(self, mock_synthesis_llm):
        llm, structured = mock_synthesis_llm
        structured.ainvoke.side_effect = Exception("Model overloaded")

        node = make_synthesis_node(llm)
        state = {
            "messages": [HumanMessage(content="Show revenue")],
            "raw_data_context": {"key": "data"},
            "errors": [],
        }
        result = await node(state)

        assert "error" in result["narrative"].lower()
        assert result["ui_components"] == []
        assert len(result["errors"]) > 0

    async def test_handles_empty_data(self, mock_synthesis_llm):
        llm, structured = mock_synthesis_llm
        structured.ainvoke.return_value = AnalyticsResponse(
            narrative="No data was retrieved for the requested query.",
        )

        node = make_synthesis_node(llm)
        state = {
            "messages": [HumanMessage(content="Show me revenue")],
            "raw_data_context": {},
            "errors": ["mcp_tool_caller: empty query plan"],
        }
        result = await node(state)

        assert result["ui_components"] == []
        assert len(result["narrative"]) > 0
