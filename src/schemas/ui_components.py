"""
Analytics Agent — Pydantic models for UI component schemas.

These models define the contract between the synthesis node and the
frontend ChartRenderer. The agent produces AnalyticsResponse with
structured components; the frontend renders them via Recharts.

Supported component types:
  - BarChart:  Categorical comparison (revenue by region, counts by category)
  - LineChart: Time-series trends (monthly revenue, daily active users)
  - AreaChart: Cumulative time-series (pipeline growth, stacked contributions)
  - PieChart:  Part-of-whole distribution (market share, portfolio allocation)
  - KPICard:   Single metric with trend indicator (total revenue, deal count)
  - DataTable: Tabular data display (top accounts, transaction details)
"""
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field


class ChartMetadata(BaseModel):
    """Metadata for any chart component — rendered as the card header."""
    title: str = Field(
        min_length=1,
        max_length=120,
        description="Chart title displayed in the card header",
    )
    source: str = Field(
        description="MCP server that produced the data (e.g. 'salesforce-mcp', 'data-mcp')",
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Agent's confidence in the data accuracy (0.0-1.0)",
    )
    x_label: Optional[str] = Field(
        default=None,
        description="X-axis label (for chart types with axes)",
    )
    y_label: Optional[str] = Field(
        default=None,
        description="Y-axis label (for chart types with axes)",
    )
    format_hint: Optional[str] = Field(
        default=None,
        description="Value formatting: 'currency' | 'percent' | 'number' | 'compact' | 'date' | 'datetime'",
    )


class ChartDataPoint(BaseModel):
    """Single data point for bar, line, area, and pie charts."""
    category: str = Field(description="X-axis category or label")
    value: float = Field(description="Primary numeric value")
    series: Optional[str] = Field(
        default=None,
        description="Series name for multi-series charts",
    )


class KPIDataPoint(BaseModel):
    """Single KPI metric with trend indicator."""
    label: str = Field(description="KPI name (e.g. 'Total Revenue')")
    value: float = Field(description="Current value")
    change: Optional[float] = Field(
        default=None,
        description="Percentage change from previous period",
    )
    trend: Optional[str] = Field(
        default=None,
        description="'up' | 'down' | 'flat'",
    )


class TableRow(BaseModel):
    """Single row in a DataTable component. Keys are column names."""
    class Config:
        extra = "allow"


class UIComponent(BaseModel):
    """A single renderable UI component sent to the frontend via SSE.

    The frontend ChartRenderer dispatches on componentType to the correct
    Recharts wrapper component.
    """
    component_type: Literal[
        "BarChart", "LineChart", "AreaChart", "PieChart", "KPICard", "DataTable"
    ] = Field(description="React component to render")
    metadata: ChartMetadata
    data: Union[list[ChartDataPoint], list[KPIDataPoint], list[dict]] = Field(
        description="Array of data points — shape depends on componentType",
    )


class AnalyticsResponse(BaseModel):
    """Complete response from the synthesis node.

    Contains a narrative summary (streamed as text events), zero or more
    UI components (each emitted as a ui_component event), and up to 3
    follow-up suggestions to guide the user's next query.
    """
    narrative: str = Field(
        min_length=1,
        description="2-3 sentence analysis summary explaining the data insights",
    )
    components: list[UIComponent] = Field(
        default_factory=list,
        description="UI components to render in the frontend canvas area",
    )
    follow_up_suggestions: list[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Up to 3 complete, submittable follow-up queries the user might ask next, "
            "grounded in the data that was just shown. Each must be a full sentence "
            "the user can send as-is (no placeholder tokens). Maximum 3 items."
        ),
    )
