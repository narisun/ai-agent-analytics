"""Analytics Agent — FastAPI app, built on platform_sdk.BaseAgentApp.

The agent subclasses :class:`platform_sdk.BaseAgentApp`, which owns the
FastAPI lifespan: telemetry init, MCP bridge connection, checkpointer
setup, conversation-store init, CORS, exception handler registration,
and router inclusion. Analytics-specific wiring lives in three hooks:

  * :meth:`build_dependencies` — sync; assembles the ``AppDependencies``
    dataclass for the route handlers (no I/O).
  * :meth:`on_started` — async; performs the one-shot data-mcp schema
    fetch and rebuilds the LangGraph with the schema injected. Runs
    after ``build_dependencies`` returns and after ``app.state.deps``
    is set, so routes never see a partially-initialised graph.
  * :meth:`register_exception_handlers` — domain-error → HTTP mapping.

Tests build their own ``AppDependencies`` via
``tests.fakes.build_test_deps.build_test_dependencies()`` and call
``create_app(deps)`` directly — no lifespan, no env, no Docker.

The module-level ``app`` symbol is created with placeholder empty deps
so ``uvicorn src.app:app`` can import it without env vars; lifespan
populates the real deps on startup.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from platform_sdk import (
    AgentConfig,
    AgentContext,
    BaseAgentApp,
    get_logger,
)

from .app_dependencies import AppDependencies
from .domain.errors import AnalyticsError, AuthError, ConversationNotFound
from .domain.types import UserContext
from .graph import build_analytics_graph
from .persistence import MemoryConversationStore, PostgresConversationStore
from .routes.chat import chat_router
from .routes.conversations import conversations_router
from .routes.health import health_router
from .routes.stream import stream_router
from .services.chat_service import ChatService
from .streaming.data_stream_encoder import DataStreamEncoder

log = get_logger(__name__)


async def _fetch_schema_context(data_bridge) -> str:
    """Call data-mcp's get_schema_context tool once at startup.

    Returns the Markdown payload, or an empty string when the bridge isn't
    connected, the tool isn't registered, or the call fails. The router
    handles an empty value gracefully (logs a warning, runs with no schema
    knowledge — useful for unit tests and degraded environments).
    """
    if data_bridge is None or not getattr(data_bridge, "is_connected", False):
        log.warning("schema_context_skipped", reason="data-mcp bridge not connected")
        return ""
    try:
        tools = await data_bridge.get_langchain_tools()
    except Exception as exc:
        log.warning("schema_context_tools_failed", error=str(exc))
        return ""
    schema_tool = next((t for t in tools if t.name == "get_schema_context"), None)
    if schema_tool is None:
        log.warning("schema_context_tool_missing")
        return ""
    try:
        result = await schema_tool.ainvoke({})
    except Exception as exc:
        log.warning("schema_context_call_failed", error=str(exc))
        return ""
    if not isinstance(result, str) or not result.strip():
        log.warning("schema_context_empty_result")
        return ""
    log.info("schema_context_loaded", chars=len(result))
    return result


class AnalyticsAgentApp(BaseAgentApp):
    """Analytics agent FastAPI surface, wired on top of BaseAgentApp."""

    service_name = "analytics-agent"
    service_title = "Analytics Agent"
    service_description = "Enterprise Agentic Analytics Platform — LangGraph orchestrator"
    config_model = AgentConfig
    mcp_dependencies = [
        "data-mcp",
        "salesforce-mcp",
        "payments-mcp",
        "news-search-mcp",
    ]
    enable_telemetry = True
    requires_checkpointer = True
    requires_conversation_store = True

    def service_agent_context(self) -> AgentContext:
        return AgentContext(
            rm_id="analytics-agent",
            rm_name="Analytics Agent",
            role="manager",
            team_id="analytics",
            assigned_account_ids=(),
            compliance_clearance=("standard", "aml_view"),
        )

    def build_conversation_store(self) -> Any:
        """Create the appropriate conversation store based on config."""
        db_url = self.config.database_url
        # 'local' was the legacy alias for dev with in-memory store; in 0.6.0
        # the strict Environment Literal removes 'local'. Use Memory in dev.
        use_postgres = (
            bool(db_url)
            and self.config.environment != "dev"
            and PostgresConversationStore is not None
        )
        if use_postgres:
            return PostgresConversationStore(db_url)
        if db_url and PostgresConversationStore is None:
            log.warning("asyncpg_not_available", fallback="MemoryConversationStore")
        return MemoryConversationStore()

    def routes(self) -> Iterable[Any]:
        return [health_router, chat_router, conversations_router, stream_router]

    def register_exception_handlers(self, app: FastAPI) -> None:
        @app.exception_handler(AuthError)
        async def _on_auth_error(request: Request, exc: AuthError):
            return JSONResponse(
                {"error_id": uuid.uuid4().hex, "type": "auth", "message": str(exc)},
                status_code=401,
            )

        @app.exception_handler(ConversationNotFound)
        async def _on_not_found(request: Request, exc: ConversationNotFound):
            return JSONResponse(
                {"error_id": uuid.uuid4().hex, "type": "not_found", "message": str(exc)},
                status_code=404,
            )

        @app.exception_handler(AnalyticsError)
        async def _on_analytics_error(request: Request, exc: AnalyticsError):
            return JSONResponse(
                {"error_id": uuid.uuid4().hex, "type": "internal"},
                status_code=500,
            )

    def build_dependencies(self, *, bridges, checkpointer, store) -> AppDependencies:
        """Assemble AppDependencies (sync). The schema-aware graph is
        installed later in :meth:`on_started` — see the holder note below.
        """
        # NB (Approach B): build_dependencies is sync, but the schema fetch
        # is async. We construct the graph here with no schema (so routes can
        # rely on a non-None deps.graph immediately if anything peeked) and
        # let on_started rebuild it. To keep chat_service_factory bound to
        # the *final* graph without restructuring AppDependencies, we capture
        # a tiny mutable holder dict and update it (plus deps.graph) in
        # on_started. stream.py already reads deps.graph at request time, so
        # it picks up the schema-aware graph automatically.
        config = self.config
        checkpointer_local = checkpointer
        conversation_store = store

        encoder_factory: Callable[[], DataStreamEncoder] = lambda: DataStreamEncoder()

        # Initial graph with empty schema context. on_started replaces it.
        initial_graph = build_analytics_graph(
            bridges=bridges,
            config=config,
            checkpointer=checkpointer_local,
            schema_context="",
        )
        graph_holder: dict[str, Any] = {"graph": initial_graph}

        def chat_service_factory(user_ctx: UserContext) -> ChatService:
            # FakeTelemetryScope-shaped no-op for the production path until a
            # real TelemetryScope adapter lands (Phase 9). Inline class avoids
            # an extra module just to hold a noop scope.
            class _NoopTelemetry:
                @contextmanager
                def start_span(self, name):
                    yield None

                def record_event(self, name, **attrs):
                    pass

            return ChatService(
                graph=graph_holder["graph"],
                conversation_store=conversation_store,
                config=config,
                user_ctx=user_ctx,
                encoder_factory=encoder_factory,
                telemetry=_NoopTelemetry(),
            )

        deps = AppDependencies(
            config=config,
            graph=initial_graph,
            conversation_store=conversation_store,
            mcp_tools_provider=None,
            llm_factory=None,
            telemetry=None,
            compaction=None,
            encoder_factory=encoder_factory,
            chat_service_factory=chat_service_factory,
        )
        # Stash the holder on deps so on_started can update it without a
        # second closure or a global. Underscored — not part of the public
        # AppDependencies contract; tests never see it.
        deps._graph_holder = graph_holder  # type: ignore[attr-defined]
        return deps

    async def on_started(self, deps, *, bridges, config, checkpointer, store) -> None:
        """Fetch the live data-mcp schema and rebuild the graph with it.

        Approach B: keeps build_dependencies sync, late-binds the schema
        into both ``deps.graph`` and the holder dict captured by
        ``chat_service_factory``. Routes that read ``deps.graph`` (stream.py)
        and ChatService instances built via the factory (chat.py) all end
        up using the same schema-aware graph.
        """
        schema_context = await _fetch_schema_context(bridges.get("data-mcp"))
        new_graph = build_analytics_graph(
            bridges=bridges,
            config=config,
            checkpointer=checkpointer,
            schema_context=schema_context,
        )
        deps.graph = new_graph
        holder = getattr(deps, "_graph_holder", None)
        if holder is not None:
            holder["graph"] = new_graph


# --------------------------------------------------------------------
# Module-level entry point — uvicorn loads this `app` symbol
# --------------------------------------------------------------------


_agent = AnalyticsAgentApp()


def create_app(deps: AppDependencies | None = None) -> FastAPI:
    """Public factory used by tests and uvicorn.

    Tests pass a fully-built ``AppDependencies`` and skip the lifespan;
    production callers pass ``None`` (or omit) and rely on the lifespan
    populating ``app.state.deps``.
    """
    return _agent.create_app(deps=deps)


_empty_deps = AppDependencies(
    config=None,
    graph=None,
    conversation_store=None,
    mcp_tools_provider=None,
    llm_factory=None,
    telemetry=None,
    compaction=None,
    encoder_factory=None,
    chat_service_factory=None,
)
app = _agent.create_app(deps=_empty_deps)
