"""
Analytics Agent — LangGraph StateGraph Orchestrator.

Graph flow:
  intent_router → (data_query)    → mcp_tool_caller → synthesis → END
  intent_router → (follow_up)     → synthesis → END
  intent_router → (clarification) → error_handler → END

Model tiering:
  intent_router:    fast-routing  (GPT-4o-mini — low latency classification)
  mcp_tool_caller:  no LLM       (pure Python — MCP tool execution)
  synthesis:        complex-routing (GPT-4o — high-quality narrative + charts)
  error_handler:    no LLM       (pure Python — error/clarification messages)
"""
from typing import Optional

from langgraph.graph import END, StateGraph

from platform_sdk import AgentConfig, get_logger, make_chat_llm, make_checkpointer, setup_checkpointer
from platform_sdk.compaction import make_compaction_modifier
from platform_sdk.prompts import PromptLoader
from .state import AnalyticsState
from .nodes.intent_router import make_intent_router_node, route_after_intent
from .nodes.mcp_tool_caller import make_mcp_tool_caller_node
from .nodes.synthesis import make_synthesis_node
from .nodes.error_handler import make_error_handler_node

log = get_logger(__name__)


def build_analytics_graph(
    bridges: dict,
    config: AgentConfig,
    prompts: Optional[PromptLoader] = None,
    checkpointer=None,
):
    """Build the Analytics Agent StateGraph.

    Args:
        bridges:      Dict mapping MCP server names to MCPToolBridge instances.
        config:       AgentConfig with model routing configuration.
        prompts:      Optional PromptLoader for template overrides.
        checkpointer: Pre-initialised LangGraph checkpointer.  Pass the result of
                      ``await setup_checkpointer(config)`` from your async lifespan
                      so checkpoint tables are guaranteed to exist before the first
                      request.  If None, falls back to make_checkpointer() (sync,
                      no table setup — safe only when tables already exist).

    Returns:
        Compiled LangGraph StateGraph.

    Example (FastAPI lifespan)::

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            cp = await setup_checkpointer(config)
            app.state.graph = build_analytics_graph(bridges, config, checkpointer=cp)
            yield
    """
    # Create LLMs with appropriate model tiers
    router_llm = make_chat_llm(config.router_model_route, config=config)
    synthesis_llm = make_chat_llm(config.synthesis_model_route, config=config)

    # Create compaction modifier for trimming long conversation histories
    compaction_modifier = make_compaction_modifier(config)

    # Build the state graph
    builder = StateGraph(AnalyticsState)

    # Register nodes (pass compaction_modifier to LLM-calling nodes)
    builder.add_node("intent_router", make_intent_router_node(router_llm, bridges, prompts, compaction_modifier))
    builder.add_node("mcp_tool_caller", make_mcp_tool_caller_node(bridges))
    builder.add_node("synthesis", make_synthesis_node(
        synthesis_llm,
        prompts,
        compaction_modifier,
        chart_max_data_points=config.chart_max_data_points,
    ))
    builder.add_node("error_handler", make_error_handler_node())

    # Set entry point
    builder.set_entry_point("intent_router")

    # Conditional routing after intent classification
    builder.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {
            "mcp_tool_caller": "mcp_tool_caller",  # data_query → fetch data
            "synthesis": "synthesis",                # follow_up → reuse existing data
            "error_handler": "error_handler",        # clarification → ask user
        },
    )

    # After MCP tool calls, always go to synthesis (even with errors)
    builder.add_edge("mcp_tool_caller", "synthesis")

    # Terminal edges
    builder.add_edge("synthesis", END)
    builder.add_edge("error_handler", END)

    # Use the pre-initialised checkpointer from the caller's async lifespan when
    # available; fall back to the sync factory only when tables already exist.
    cp = checkpointer if checkpointer is not None else make_checkpointer(config)
    compiled = builder.compile(checkpointer=cp)

    log.info(
        "analytics_graph_built",
        nodes=["intent_router", "mcp_tool_caller", "synthesis", "error_handler"],
        router_model=config.router_model_route,
        synthesis_model=config.synthesis_model_route,
        chart_max_data_points=config.chart_max_data_points,
        checkpointer_type=type(cp).__name__,
    )

    return compiled
