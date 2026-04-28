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

    async def test_barchart_data_sorted_desc_by_value(self, mock_synthesis_llm):
        """When the LLM emits a BarChart with values NOT in descending order
        (e.g. because the SQL ordered by a different column), the synthesis
        node must reorder the bars DESC by value so the visual matches what
        a "top N by amount" title implies. Without this, the leftmost bar
        looks like the largest when it might not be."""
        llm, structured = mock_synthesis_llm
        # Reproduces the BMO bug: data ordered by transaction_count DESC, so
        # values (amounts) are NOT in descending order.
        structured.ainvoke.return_value = AnalyticsResponse(
            narrative="Top parties by transaction amount with BMO.",
            components=[
                UIComponent(
                    component_type="BarChart",
                    metadata=ChartMetadata(
                        title="Top Parties Using BMO by Transaction Amount",
                        source="data-mcp",
                        confidence_score=0.9,
                        format_hint="currency",
                    ),
                    data=[
                        ChartDataPoint(category="Costco Wholesale", value=24_880_000),
                        ChartDataPoint(category="Precision Mfg.", value=26_230_000),
                        ChartDataPoint(category="Alphabet Inc.", value=28_900_000),
                        ChartDataPoint(category="The Home Depot", value=13_400_000),
                    ],
                ),
            ],
        )

        node = make_synthesis_node(llm)
        state = {
            "messages": [HumanMessage(content="which party uses BMO most?")],
            "raw_data_context": {"data-mcp:execute_read_query:0": "{}"},
            "errors": [],
        }
        result = await node(state)

        component = result["ui_components"][0]
        assert component["component_type"] == "BarChart"
        values = [d["value"] for d in component["data"]]
        assert values == sorted(values, reverse=True), (
            f"BarChart bars must be in descending order by value, got {values}"
        )
        # The first bar must be the actual maximum, not whatever order the LLM gave.
        assert component["data"][0]["category"] == "Alphabet Inc."

    async def test_barchart_already_sorted_stays_sorted(self, mock_synthesis_llm):
        """Idempotency: data already sorted DESC by value passes through unchanged."""
        llm, structured = mock_synthesis_llm
        structured.ainvoke.return_value = AnalyticsResponse(
            narrative="Revenue by region.",
            components=[
                UIComponent(
                    component_type="BarChart",
                    metadata=ChartMetadata(
                        title="Revenue by Region",
                        source="salesforce-mcp",
                        confidence_score=0.95,
                        format_hint="currency",
                    ),
                    data=[
                        ChartDataPoint(category="NA", value=1_200_000),
                        ChartDataPoint(category="EMEA", value=800_000),
                        ChartDataPoint(category="APAC", value=400_000),
                    ],
                ),
            ],
        )

        node = make_synthesis_node(llm)
        state = {
            "messages": [HumanMessage(content="Show revenue by region")],
            "raw_data_context": {"salesforce-mcp:get:0": "{}"},
            "errors": [],
        }
        result = await node(state)

        categories = [d["category"] for d in result["ui_components"][0]["data"]]
        assert categories == ["NA", "EMEA", "APAC"]

    async def test_linechart_data_order_preserved(self, mock_synthesis_llm):
        """LineCharts are time-series — order by category (date), not value.
        The DESC-by-value sort must NOT apply to LineCharts."""
        llm, structured = mock_synthesis_llm
        structured.ainvoke.return_value = AnalyticsResponse(
            narrative="Monthly revenue trend.",
            components=[
                UIComponent(
                    component_type="LineChart",
                    metadata=ChartMetadata(
                        title="Monthly Revenue",
                        source="data-mcp",
                        confidence_score=0.9,
                        format_hint="currency",
                    ),
                    data=[
                        ChartDataPoint(category="2026-01-01", value=100_000),
                        ChartDataPoint(category="2026-02-01", value=150_000),
                        ChartDataPoint(category="2026-03-01", value=120_000),
                    ],
                ),
            ],
        )

        node = make_synthesis_node(llm)
        state = {
            "messages": [HumanMessage(content="Show monthly revenue")],
            "raw_data_context": {"data-mcp:execute_read_query:0": "{}"},
            "errors": [],
        }
        result = await node(state)

        # Categories must remain in chronological order — NOT sorted by value
        categories = [d["category"] for d in result["ui_components"][0]["data"]]
        assert categories == ["2026-01-01", "2026-02-01", "2026-03-01"]
