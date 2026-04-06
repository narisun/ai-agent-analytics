"""
Analytics Agent — Synthesis & Formatting Node.

Takes raw data from MCP tool calls and produces:
1. A narrative summary (2-3 sentences) explaining the key insights
2. UI component schemas (charts, KPIs, tables) for the frontend canvas

Uses complex-routing (GPT-4o) for high-quality narrative generation
and structured output to guarantee valid UI component schemas.
"""
import json
from datetime import datetime, timezone
from typing import Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage

from platform_sdk import get_logger
from ..schemas.ui_components import AnalyticsResponse, UIComponent
from ..state import AnalyticsState

log = get_logger(__name__)


_SYNTHESIS_SYSTEM_PROMPT = """You are a data analyst for an enterprise analytics platform.

Your job is to analyze raw data from enterprise systems and produce:
1. A clear, concise narrative (2-3 sentences) summarizing the key insights
2. Appropriate UI visualizations that best represent the data

## UI Component Types

Choose the most appropriate visualization for each data insight:

- **BarChart**: Categorical comparisons (revenue by region, counts by department)
- **LineChart**: Time-series trends (monthly revenue, daily active users)
- **AreaChart**: Cumulative time-series (pipeline growth over time)
- **PieChart**: Part-of-whole distributions (market share, allocation breakdown)
- **KPICard**: Single headline metrics with trend indicators (total revenue, deal count)
- **DataTable**: Detailed tabular data (top accounts, transaction records)

## Rules

1. ALWAYS provide at least one UI component when data is available.
2. Use KPICard for headline numbers with trends (e.g., total revenue up 12%).
3. Use BarChart for comparing categories. Use LineChart for time-series data.
4. For the `source` field, use the MCP server name that produced the data.
5. Set confidence_score based on data completeness (1.0 = all data present, lower if partial).
6. Use format_hint to tell the frontend how to format values:
   - "currency" for monetary values
   - "percent" for percentages
   - "number" for plain numbers
   - "compact" for abbreviated large numbers (1.2M, 500K)
7. Ensure data values are RAW NUMBERS (not formatted strings like "$1.2M").
8. The narrative should reference specific numbers and trends from the data.
9. If data contains errors, acknowledge them in the narrative and reduce confidence_score.
10. If the raw data contains a "_client_suggestions" key, the requested client was not found.
    In this case, your narrative MUST:
    - Tell the user clearly that the client name was not found
    - If suggestions are provided, list them and ask "Did you mean one of these?"
    - Do NOT generate any charts or KPIs — return an empty components list
    - Keep the tone helpful and encourage the user to try an exact name
"""


def make_synthesis_node(synthesis_llm, prompts=None, compaction_modifier: Optional[Callable] = None):
    """Build the synthesis node.

    Args:
        synthesis_llm: LangChain ChatModel configured with complex-routing.
        prompts: Optional PromptLoader for template overrides.
        compaction_modifier: Optional callable to trim messages before LLM call.
    """
    structured_llm = synthesis_llm.with_structured_output(AnalyticsResponse)

    async def synthesis_node(state: AnalyticsState) -> dict:
        raw_data = state.get("raw_data_context", {})

        # Apply compaction to conversation history if modifier is available
        messages = list(state["messages"])
        if compaction_modifier is not None:
            messages = compaction_modifier({"messages": messages})

        user_messages = [m for m in messages if hasattr(m, "type") and m.type == "human"]
        last_query = user_messages[-1].content if user_messages else "Analyze the data"

        # Build the synthesis prompt with raw data context
        data_summary = json.dumps(raw_data, indent=2, default=str)
        if len(data_summary) > 15000:
            # Truncate very large data payloads to fit context window
            data_summary = data_summary[:15000] + "\n... (truncated)"

        errors = state.get("errors", [])
        error_context = ""
        if errors:
            error_context = (
                f"\n\n## Data Retrieval Errors\n"
                f"The following errors occurred: {errors}\n"
                f"Acknowledge missing data in the narrative and lower confidence scores."
            )

        prompt_content = (
            f"{_SYNTHESIS_SYSTEM_PROMPT}\n\n"
            f"## User Query\n{last_query}\n\n"
            f"## Raw Data from MCP Servers\n```json\n{data_summary}\n```"
            f"{error_context}"
        )

        try:
            result = await structured_llm.ainvoke([HumanMessage(content=prompt_content)])
            log.info(
                "synthesis_complete",
                components=len(result.components),
                narrative_length=len(result.narrative),
            )
            return {
                "narrative": result.narrative,
                "ui_components": [c.model_dump() for c in result.components],
                "messages": [AIMessage(content=result.narrative)],
            }
        except Exception as exc:
            log.error("synthesis_error", error=str(exc))
            fallback_narrative = (
                f"I retrieved data from {len(raw_data)} sources but encountered an error "
                f"generating the analysis: {exc}. The raw data is available for inspection."
            )
            return {
                "narrative": fallback_narrative,
                "ui_components": [],
                "messages": [AIMessage(content=fallback_narrative)],
                "errors": [f"synthesis: {exc}"],
            }

    return synthesis_node
