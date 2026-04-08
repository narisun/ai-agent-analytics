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
import logging
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

_log = logging.getLogger(__name__)

ANALYTICS_STATE_VERSION = 1


def _merge_data_context(old: dict, new: dict) -> dict:
    """Merge raw data from multiple MCP calls — last write wins per key.

    Logs a warning when a key collision occurs so operators can detect tool
    responses that overwrite each other's data.  Collisions are most likely
    caused by two MCP calls returning a top-level key with the same name
    (e.g. both returning "data" or "rows").  Fix by namespacing tool results
    in mcp_tool_caller before storing them in raw_data_context.
    """
    merged = dict(old or {})
    for k, v in (new or {}).items():
        if k in merged:
            _log.warning(
                "data_context_key_collision key=%s — existing value overwritten by new MCP result. "
                "Consider namespacing tool results by tool name to avoid silent data loss.",
                k,
            )
        merged[k] = v
    return merged


def _merge_errors(left: list, right: list) -> list:
    """Append errors from parallel nodes."""
    return list(left or []) + list(right or [])


def migrate_state(state: dict) -> dict:
    """Return a dict of state updates needed to bring an old checkpointed session
    up to the current schema version.

    Call this at the top of the intent_router node (the graph entry point) so
    that sessions checkpointed by an older deployment are migrated transparently
    on the next request without manual intervention.

    Returns an empty dict when no migration is needed (the common path).

    Usage in a node::

        from .state import migrate_state
        migration = migrate_state(state)
        if migration:
            state = {**state, **migration}
        # ... rest of node logic
        return {**migration, "intent": result.intent, ...}

    Design note: migrations are additive only (new fields with defaults).
    Renaming or removing fields requires a full session replay or a manual
    database migration on the checkpointer store.
    """
    version = state.get("schema_version", 0)
    if version >= ANALYTICS_STATE_VERSION:
        return {}  # no migration needed

    _log.warning(
        "state_schema_migrating from_version=%d to_version=%d — "
        "applying forward migrations for this session.",
        version,
        ANALYTICS_STATE_VERSION,
    )
    updates: dict = {}

    # v0 → v1: optional fields added in v1 that older sessions won't have
    if version < 1:
        if "follow_up_suggestions" not in state:
            updates["follow_up_suggestions"] = []
        if "active_tools" not in state:
            updates["active_tools"] = []
        if "intent_reasoning" not in state:
            updates["intent_reasoning"] = None

    updates["schema_version"] = ANALYTICS_STATE_VERSION
    return updates


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
    follow_up_suggestions: list[str]  # Up to 3 contextual follow-up queries

    # ---- Error tracking ----
    errors: Annotated[list, _merge_errors]

    # ---- Session ----
    session_id: str
    turn_count: int

    # ---- Schema versioning ----
    schema_version: int
