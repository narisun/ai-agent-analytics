"""
Analytics Agent — LangGraph state schema.

AnalyticsState flows through every node in the analytics StateGraph.
Each node reads from it and writes only its own output field(s).

Graph flow:
  intent_router → (data_query) → mcp_tool_caller → synthesis → END
  intent_router → (follow_up)  → synthesis → END  (reuses existing data)
  intent_router → (clarification) → END

Multi-turn support:
  - intent field determines graph branch
  - raw_data_context persists across turns for follow-up queries
  - ui_components accumulates charts for the canvas area
"""
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


ANALYTICS_STATE_VERSION = 1


def _merge_data_context(old: dict, new: dict) -> dict:
    """Merge raw data from multiple MCP calls — last write wins per key."""
    merged = dict(old or {})
    merged.update(new or {})
    return merged


def _merge_errors(left: list, right: list) -> list:
    """Append errors from parallel nodes."""
    return list(left or []) + list(right or [])


class AnalyticsState(TypedDict):
    # ---- Conversation history (append-only; supports multi-turn) ----
    messages: Annotated[list, add_messages]

    # ---- Intent classification (set by intent_router) ----
    intent: str                       # "data_query" | "follow_up" | "clarification"
    query_plan: list[dict]            # [{tool_name, mcp_server, parameters, description}]
    intent_reasoning: Optional[str]   # Chain-of-thought (displayed in trace panel)

    # ---- Raw data from MCP tool calls (merged across parallel calls) ----
    raw_data_context: Annotated[dict, _merge_data_context]

    # ---- Active tool tracking (for glass-box trace panel) ----
    active_tools: list[dict]          # [{tool, server, status, started_at}]

    # ---- Final output ----
    ui_components: list[dict]         # Validated UIComponent schemas for frontend
    narrative: str                    # 2-3 sentence analysis summary

    # ---- Error tracking ----
    errors: Annotated[list, _merge_errors]

    # ---- Session ----
    session_id: str
    turn_count: int

    # ---- Schema versioning ----
    schema_version: int
