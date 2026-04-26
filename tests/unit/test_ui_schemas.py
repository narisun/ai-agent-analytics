"""Unit tests for UI component Pydantic schemas."""
import pytest
from src.schemas.ui_components import (
    AnalyticsResponse,
    ChartDataPoint,
    ChartMetadata,
    KPIDataPoint,
    UIComponent,
)
from src.schemas.intent import IntentResult, QueryPlanStep


# ── ChartMetadata ──────────────────────────────────────────────────────────────

class TestChartMetadata:
    def test_valid_metadata(self):
        meta = ChartMetadata(
            title="Q3 Revenue by Region",
            source="salesforce-mcp",
            confidence_score=0.95,
            x_label="Region",
            y_label="Revenue",
            format_hint="currency",
        )
        assert meta.title == "Q3 Revenue by Region"
        assert meta.confidence_score == 0.95

    def test_minimal_metadata(self):
        meta = ChartMetadata(title="Test", source="data-mcp", confidence_score=0.5)
        assert meta.x_label is None
        assert meta.format_hint is None

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            ChartMetadata(title="Test", source="x", confidence_score=1.5)
        with pytest.raises(Exception):
            ChartMetadata(title="Test", source="x", confidence_score=-0.1)

    def test_empty_title_rejected(self):
        with pytest.raises(Exception):
            ChartMetadata(title="", source="x", confidence_score=0.5)


# ── ChartDataPoint ─────────────────────────────────────────────────────────────

class TestChartDataPoint:
    def test_valid_data_point(self):
        dp = ChartDataPoint(category="North America", value=1200000)
        assert dp.category == "North America"
        assert dp.value == 1200000
        assert dp.series is None

    def test_multi_series_data_point(self):
        dp = ChartDataPoint(category="Q1", value=500000, series="Product A")
        assert dp.series == "Product A"


# ── UIComponent ────────────────────────────────────────────────────────────────

class TestUIComponent:
    def test_bar_chart(self):
        comp = UIComponent(
            component_type="BarChart",
            metadata=ChartMetadata(title="Test", source="data-mcp", confidence_score=0.9),
            data=[
                ChartDataPoint(category="A", value=100),
                ChartDataPoint(category="B", value=200),
            ],
        )
        assert comp.component_type == "BarChart"
        assert len(comp.data) == 2

    def test_kpi_card(self):
        comp = UIComponent(
            component_type="KPICard",
            metadata=ChartMetadata(
                title="Total Revenue",
                source="salesforce-mcp",
                confidence_score=0.98,
                format_hint="currency",
            ),
            data=[
                {"label": "Current Quarter", "value": 4500000, "change": 12.5, "trend": "up"},
            ],
        )
        assert comp.component_type == "KPICard"

    def test_data_table(self):
        comp = UIComponent(
            component_type="DataTable",
            metadata=ChartMetadata(title="Top Accounts", source="salesforce-mcp", confidence_score=0.85),
            data=[
                {"name": "Acme Corp", "revenue": 500000, "status": "Active"},
                {"name": "Globex Inc", "revenue": 350000, "status": "Active"},
            ],
        )
        assert len(comp.data) == 2

    def test_invalid_component_type(self):
        with pytest.raises(Exception):
            UIComponent(
                component_type="Histogram",
                metadata=ChartMetadata(title="Test", source="x", confidence_score=0.5),
                data=[],
            )

    def test_serialization_round_trip(self):
        comp = UIComponent(
            component_type="LineChart",
            metadata=ChartMetadata(title="Trend", source="data-mcp", confidence_score=0.9),
            data=[ChartDataPoint(category="Jan", value=100)],
        )
        dumped = comp.model_dump()
        restored = UIComponent(**dumped)
        assert restored.component_type == "LineChart"


# ── AnalyticsResponse ──────────────────────────────────────────────────────────

class TestAnalyticsResponse:
    def test_valid_response(self):
        resp = AnalyticsResponse(
            narrative="Revenue grew 15% in Q3, driven by North America.",
            components=[
                UIComponent(
                    component_type="BarChart",
                    metadata=ChartMetadata(title="Q3 Revenue", source="salesforce-mcp", confidence_score=0.95),
                    data=[ChartDataPoint(category="NA", value=1200000)],
                ),
            ],
        )
        assert len(resp.components) == 1
        assert "15%" in resp.narrative

    def test_response_without_components(self):
        resp = AnalyticsResponse(narrative="No data available for the requested query.")
        assert resp.components == []

    def test_empty_narrative_rejected(self):
        with pytest.raises(Exception):
            AnalyticsResponse(narrative="")


# ── IntentResult ───────────────────────────────────────────────────────────────

class TestIntentResult:
    def test_data_query_intent(self):
        result = IntentResult(
            intent="data_query",
            query_plan=[
                QueryPlanStep(
                    tool_name="get_salesforce_summary",
                    mcp_server="salesforce-mcp",
                    parameters={"account_name": "Acme Corp"},
                    description="Fetch CRM summary for Acme Corp",
                ),
            ],
            reasoning="User asked for revenue data which requires CRM lookup.",
        )
        assert result.intent == "data_query"
        assert len(result.query_plan) == 1

    def test_follow_up_intent(self):
        result = IntentResult(
            intent="follow_up",
            reasoning="User is asking about previously retrieved data.",
        )
        assert result.query_plan == []

    def test_invalid_intent(self):
        with pytest.raises(Exception):
            IntentResult(intent="invalid_type", reasoning="test")
