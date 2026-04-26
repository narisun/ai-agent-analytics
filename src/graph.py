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

from .nodes.error_handler import ErrorHandlerNode
from .nodes.intent_router import IntentRouterNode
from .nodes.mcp_tool_caller import MCPToolCallerNode
from .nodes.synthesis import SynthesisNode
from .graph_dependencies import GraphDependencies

log = get_logger(__name__)


class _MCPBridgeToolsProvider:
    """Wraps a dict of MCPToolBridge instances behind the MCPToolsProvider Protocol.

    Defined here in graph.py so the legacy path of build_analytics_graph can
    construct a GraphDependencies on the fly. Phase 4.3's lifespan.py will own
    its own equivalent adapter once it lands.
    """

    def __init__(self, bridges: dict):
        self._bridges = bridges or {}

    async def get_langchain_tools(self, user_ctx=None):
        tools: list = []
        for bridge in self._bridges.values():
            if getattr(bridge, "is_connected", False):
                bridge_tools = (
                    await bridge.get_langchain_tools(user_ctx)
                    if user_ctx is not None
                    else await bridge.get_langchain_tools()
                )
                tools.extend(bridge_tools)
        return tools


class _LLMFactoryAdapter:
    """Wraps platform_sdk.make_chat_llm into the LLMFactory Protocol shape.

    Defined here for the same reason as _MCPBridgeToolsProvider.
    """

    def __init__(self, config: "AgentConfig"):
        self._config = config

    def make_router_llm(self):
        return make_chat_llm(self._config.router_model_route, config=self._config)

    def make_synthesis_llm(self):
        return make_chat_llm(self._config.synthesis_model_route, config=self._config)


class _CompactionAdapter:
    """Adapts the make_compaction_modifier callable to the CompactionModifier Protocol.

    The compaction modifier returned by make_compaction_modifier accepts a state
    dict or list and returns a list. The CompactionModifier Protocol (ports.py)
    requires an .apply(messages) -> list method.
    """

    def __init__(self, modifier):
        self._modifier = modifier

    def apply(self, messages):
        return self._modifier(messages)


def build_analytics_graph(
    deps: "GraphDependencies | None" = None,
    *,
    bridges: dict | None = None,
    config: "AgentConfig | None" = None,
    prompts: Optional[PromptLoader] = None,
    checkpointer=None,
    schema_context: str = "",
):
    """Build the Analytics Agent StateGraph.

    Two call shapes are supported:

    1. **Preferred:** ``build_analytics_graph(deps, checkpointer=cp)`` where
       ``deps`` is a fully-wired ``GraphDependencies`` (Phase 4 composition root).

    2. **Legacy:** ``build_analytics_graph(bridges=..., config=..., prompts=..., checkpointer=...)``
       — kept for the existing ``app.py`` call site. The implementation builds a
       ``GraphDependencies`` on the fly from these kwargs and delegates to the
       new path. This shape is retained transiently and will be removed once
       lifespan.py owns the wiring (Phase 4.3 / Phase 9).

    Args:
        deps:         Fully-wired GraphDependencies. Mutually exclusive with
                      the legacy kwargs.
        bridges:      Dict mapping MCP server names to MCPToolBridge instances.
                      LEGACY — use ``deps`` instead.
        config:       AgentConfig. LEGACY — use ``deps.config`` instead.
        prompts:      Optional PromptLoader. LEGACY — use ``deps.prompts``.
        checkpointer: Pre-initialised LangGraph checkpointer.

    Returns:
        Compiled LangGraph StateGraph.
    """
    if deps is None:
        if config is None:
            raise ValueError(
                "build_analytics_graph requires either `deps` or the legacy "
                "(`bridges`, `config`) kwargs."
            )
        compaction_modifier = make_compaction_modifier(config)
        compaction = _CompactionAdapter(compaction_modifier)
        deps = GraphDependencies(
            llm_factory=_LLMFactoryAdapter(config),
            tools_provider=_MCPBridgeToolsProvider(bridges or {}),
            compaction=compaction,
            config=config,
            prompts=prompts,
        )

    # Build the state graph
    builder = StateGraph(AnalyticsState)

    # Instantiate node classes from deps. Each class was added in Phase 6.2/6.4.
    intent_node = IntentRouterNode(
        llm=deps.llm_factory.make_router_llm(),
        tools_provider=deps.tools_provider,
        prompts=deps.prompts,
        compaction=deps.compaction,
        schema_context=schema_context,
    )
    tool_node = MCPToolCallerNode(tools_provider=deps.tools_provider)
    synth_node = SynthesisNode(
        llm=deps.llm_factory.make_synthesis_llm(),
        prompts=deps.prompts,
        compaction=deps.compaction,
        chart_max_data_points=deps.config.chart_max_data_points,
    )
    err_node = ErrorHandlerNode()

    builder.add_node("intent_router", intent_node)
    builder.add_node("mcp_tool_caller", tool_node)
    builder.add_node("synthesis", synth_node)
    builder.add_node("error_handler", err_node)

    builder.set_entry_point("intent_router")
    builder.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {
            "mcp_tool_caller": "mcp_tool_caller",
            "synthesis": "synthesis",
            "error_handler": "error_handler",
        },
    )
    builder.add_edge("mcp_tool_caller", "synthesis")
    builder.add_edge("synthesis", END)
    builder.add_edge("error_handler", END)

    # Use the pre-initialised checkpointer from the caller when available;
    # fall back to the sync factory only when tables already exist.
    cp = checkpointer if checkpointer is not None else make_checkpointer(deps.config)
    compiled = builder.compile(checkpointer=cp)

    log.info(
        "analytics_graph_built",
        nodes=["intent_router", "mcp_tool_caller", "synthesis", "error_handler"],
        router_model=deps.config.router_model_route,
        synthesis_model=deps.config.synthesis_model_route,
        chart_max_data_points=deps.config.chart_max_data_points,
        checkpointer_type=type(cp).__name__,
    )

    return compiled
