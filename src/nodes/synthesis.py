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

# Default maximum data points per chart component.
# Overridden at runtime via AgentConfig.chart_max_data_points / CHART_MAX_DATA_POINTS env var.
# Large arrays (40+ items) often cause malformed structured output from the LLM.
MAX_CHART_DATA_POINTS = 20

# Number of retries when structured output fails (JSON parse errors)
SYNTHESIS_MAX_RETRIES = 2

# Template — {max_data_points} is resolved inside make_synthesis_node() using
# the runtime config value so operators can tune it without code changes.
_SYNTHESIS_SYSTEM_PROMPT_TEMPLATE = """You are a data analyst for an enterprise analytics platform.

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
   - "date" for date columns — the frontend will auto-detect and format date values
   - "datetime" for datetime columns
7. Ensure data values are RAW NUMBERS (not formatted strings like "$1.2M").
   For DATE columns, keep the original value (integer like 20250803 or string like "2025-08-03") —
   the frontend will detect and format it correctly.
8. The narrative should reference specific numbers and trends from the data.
9. If data contains errors, acknowledge them in the narrative and reduce confidence_score.
10. If the raw data contains a "_client_suggestions" key, the requested client was not found.
    In this case, your narrative MUST:
    - Tell the user clearly that the client name was not found
    - If suggestions are provided, list them and ask "Did you mean one of these?"
    - Do NOT generate any charts or KPIs — return an empty components list
    - Keep the tone helpful and encourage the user to try an exact name
11. IMPORTANT: Limit chart data arrays to at most {max_data_points} items.
    If there are more items, include only the top {max_data_points} by value
    and mention in the narrative that you are showing the top {max_data_points}.

## Follow-Up Suggestions

After generating the narrative and components, produce exactly 3 items in follow_up_suggestions.

Rules:
- Each suggestion must be a complete, self-contained sentence the user can submit as-is.
- Do NOT use placeholder tokens like [company name] — use actual entity names from the data
  you retrieved (e.g. "Microsoft Corp.", "Goldman Sachs", "Acme Corp").
- Each suggestion should approach the topic from a DIFFERENT angle:
    • One drill-down: more granular detail on a specific entity or time window
    • One comparison: benchmark the entity/metric against another entity or time period
    • One cross-domain: pivot to a different data source (CRM if you queried payments,
      payments if you queried CRM, news if you queried either)
- Keep each suggestion under 18 words.
- If no useful data was retrieved (error case), return 3 generic but helpful suggestions
  that guide the user toward a successful query.
"""


class SynthesisNode:
    """Callable class for the synthesis node.

    Constructor-injected with ``llm``, ``prompts``, ``compaction``
    (CompactionModifier Protocol), and ``chart_max_data_points``.

    See ``make_synthesis_node`` for the backward-compat shim.
    """

    def __init__(
        self,
        *,
        llm,
        prompts,
        compaction,
        chart_max_data_points: int,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._compaction = compaction
        self._chart_max_data_points = chart_max_data_points
        # Resolve system prompt with the runtime limit so the LLM knows the cap
        self._system_prompt = _SYNTHESIS_SYSTEM_PROMPT_TEMPLATE.format(
            max_data_points=chart_max_data_points
        )
        self._structured_llm = llm.with_structured_output(AnalyticsResponse)
        log.info("synthesis_node_built", chart_max_data_points=chart_max_data_points)

    async def __call__(self, state: AnalyticsState, config=None) -> dict:
        raw_data = state.get("raw_data_context", {})
        chart_max_data_points = self._chart_max_data_points

        # Apply compaction to conversation history if modifier is available
        messages = list(state["messages"])
        if self._compaction is not None:
            messages = self._compaction.apply(messages)

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
            f"{self._system_prompt}\n\n"
            f"## User Query\n{last_query}\n\n"
            f"## Raw Data from MCP Servers\n```json\n{data_summary}\n```"
            f"{error_context}"
        )

        last_error: Optional[Exception] = None
        for attempt in range(1 + SYNTHESIS_MAX_RETRIES):
            try:
                result = await self._structured_llm.ainvoke([HumanMessage(content=prompt_content)])

                # Post-process: enforce chart_max_data_points on each component.
                # This is a safety net — the LLM should respect the limit in the prompt,
                # but structured output parsing can return more items than instructed.
                for comp in result.components:
                    if (
                        comp.component_type in ("BarChart", "LineChart", "AreaChart")
                        and len(comp.data) > chart_max_data_points
                    ):
                        log.warning(
                            "chart_data_truncated",
                            component_type=comp.component_type,
                            original_count=len(comp.data),
                            truncated_to=chart_max_data_points,
                        )
                        # Sort descending by "value" and keep top N
                        try:
                            sorted_data = sorted(
                                comp.data,
                                key=lambda d: d.get("value", 0) if isinstance(d, dict) else 0,
                                reverse=True,
                            )
                            comp.data = sorted_data[:chart_max_data_points]
                        except (TypeError, AttributeError):
                            comp.data = comp.data[:chart_max_data_points]

                log.info(
                    "synthesis_complete",
                    components=len(result.components),
                    follow_ups=len(result.follow_up_suggestions),
                    narrative_length=len(result.narrative),
                    attempt=attempt + 1,
                )
                return {
                    "narrative": result.narrative,
                    "ui_components": [c.model_dump() for c in result.components],
                    "follow_up_suggestions": result.follow_up_suggestions,
                    "messages": [AIMessage(content=result.narrative)],
                }
            except Exception as exc:
                last_error = exc
                if attempt < SYNTHESIS_MAX_RETRIES:
                    log.warning(
                        "synthesis_retry",
                        error=str(exc),
                        attempt=attempt + 1,
                        max_retries=SYNTHESIS_MAX_RETRIES,
                    )
                    continue

        # All retries exhausted
        log.error("synthesis_error", error=str(last_error))
        fallback_narrative = (
            f"I retrieved data from {len(raw_data)} sources but encountered an error "
            f"generating the analysis. Please try rephrasing your question or asking "
            f"for a more specific subset of data."
        )
        return {
            "narrative": fallback_narrative,
            "ui_components": [],
            "follow_up_suggestions": [
                "Show total completed payments by transaction type",
                "List the top 10 CRM accounts ranked by annual revenue",
                "Which banks processed the highest payment volume last quarter?",
            ],
            "messages": [AIMessage(content=fallback_narrative)],
            "errors": [f"synthesis: {last_error}"],
        }


def make_synthesis_node(
    synthesis_llm,
    prompts=None,
    compaction_modifier: Optional[Callable] = None,
    chart_max_data_points: int = MAX_CHART_DATA_POINTS,
):
    """Build the synthesis node. Backward-compat shim.

    Args:
        synthesis_llm:         LangChain ChatModel configured with complex-routing.
        prompts:               Optional PromptLoader for template overrides.
        compaction_modifier:   Optional callable to trim messages before LLM call.
        chart_max_data_points: Maximum data points per chart component.  Passed via
                               AgentConfig.chart_max_data_points so operators can tune
                               it without a code change.  Defaults to MAX_CHART_DATA_POINTS (20).
    """

    class _LegacyCompactionAdapter:
        """Adapts old-style compaction_modifier callable to CompactionModifier Protocol.

        The old factory passed a callable that accepted {"messages": messages} and
        returned a list. The new SynthesisNode expects .apply(messages) -> list.
        """

        def __init__(self, modifier):
            self._modifier = modifier

        def apply(self, messages):
            return self._modifier({"messages": messages})

    compaction = (
        _LegacyCompactionAdapter(compaction_modifier)
        if compaction_modifier is not None
        else None
    )

    return SynthesisNode(
        llm=synthesis_llm,
        prompts=prompts,
        compaction=compaction,
        chart_max_data_points=chart_max_data_points,
    )
