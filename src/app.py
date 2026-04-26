"""Analytics Agent — FastAPI application factory.

create_app(deps: AppDependencies) is a pure function — no env reads, no
I/O, no module-level state. It registers routes, middleware, and
exception handlers on a fresh FastAPI instance using the provided
dependencies, and returns it.

Production startup goes through `lifespan(app)`, which reads the
environment, connects MCP bridges, builds the LangGraph graph,
constructs the conversation store, and assembles the real
AppDependencies. The lifespan is the ONLY place that performs I/O at
startup.

The module-level `app` exists so uvicorn can find it
(`uvicorn src.app:app`); it starts with empty placeholder deps and is
populated by lifespan() on startup.

Tests construct their own AppDependencies via
`tests.fakes.build_test_deps.build_test_dependencies()` and call
`create_app(deps)` directly — no lifespan, no env, no Docker.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from platform_sdk import (
    AgentConfig,
    AgentContext,
    configure_logging,
    flush_langfuse,
    get_logger,
    setup_checkpointer,
    setup_telemetry,
)
from platform_sdk.mcp_bridge import MCPToolBridge

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

configure_logging()
log = get_logger(__name__)


# --------------------------------------------------------------------
# Conversation store factory (env-aware)
# --------------------------------------------------------------------


def _make_conversation_store():
    """Create the appropriate conversation store based on environment."""
    db_url = os.getenv("DATABASE_URL")
    if (
        db_url
        and os.getenv("ENVIRONMENT") not in ("local",)
        and PostgresConversationStore is not None
    ):
        return PostgresConversationStore(db_url)
    if db_url and PostgresConversationStore is None:
        log.warning("asyncpg_not_available", fallback="MemoryConversationStore")
    return MemoryConversationStore()


# --------------------------------------------------------------------
# Lifespan — the ONLY place that performs startup I/O
# --------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build real AppDependencies from environment; assign to app.state.deps."""
    setup_telemetry("analytics-agent")
    config = AgentConfig.from_env()

    # Service-level AgentContext for outbound MCP SSE connections.
    agent_context = AgentContext(
        rm_id="analytics-agent",
        rm_name="Analytics Agent",
        role="manager",
        team_id="analytics",
        assigned_account_ids=(),
        compliance_clearance=("standard", "aml_view"),
    )

    mcp_urls = {
        "data-mcp":         os.getenv("DATA_MCP_URL",       "http://data-mcp:8000/sse"),
        "salesforce-mcp":   os.getenv("SALESFORCE_MCP_URL", "http://salesforce-mcp:8000/sse"),
        "payments-mcp":     os.getenv("PAYMENTS_MCP_URL",   "http://payments-mcp:8000/sse"),
        "news-search-mcp":  os.getenv("NEWS_MCP_URL",       "http://news-search-mcp:8000/sse"),
    }

    bridges: dict[str, MCPToolBridge] = {
        name: MCPToolBridge(url, agent_context=agent_context)
        for name, url in mcp_urls.items()
    }

    log.info(
        "mcp_connecting_all",
        servers=list(mcp_urls.keys()),
        timeout=config.mcp_startup_timeout,
    )
    await asyncio.gather(
        *[bridge.connect(startup_timeout=config.mcp_startup_timeout) for bridge in bridges.values()],
        return_exceptions=True,
    )
    for name, bridge in bridges.items():
        log.info("mcp_startup_status", server=name, connected=bridge.is_connected)

    checkpointer = await setup_checkpointer(config)
    graph = build_analytics_graph(bridges=bridges, config=config, checkpointer=checkpointer)

    conversation_store = _make_conversation_store()
    if hasattr(conversation_store, "connect"):
        await conversation_store.connect()

    encoder_factory: Callable[[], DataStreamEncoder] = lambda: DataStreamEncoder()

    def chat_service_factory(user_ctx: UserContext) -> ChatService:
        # FakeTelemetryScope-shaped no-op for the production path until a
        # real TelemetryScope adapter lands (Phase 9). Inline class avoids
        # an extra module just to hold a noop scope.
        from contextlib import contextmanager

        class _NoopTelemetry:
            @contextmanager
            def start_span(self, name):
                yield None

            def record_event(self, name, **attrs):
                pass

        return ChatService(
            graph=graph,
            conversation_store=conversation_store,
            config=config,
            user_ctx=user_ctx,
            encoder_factory=encoder_factory,
            telemetry=_NoopTelemetry(),
        )

    deps = AppDependencies(
        config=config,
        graph=graph,
        conversation_store=conversation_store,
        mcp_tools_provider=None,
        llm_factory=None,
        telemetry=None,
        compaction=None,
        encoder_factory=encoder_factory,
        chat_service_factory=chat_service_factory,
    )
    app.state.deps = deps
    # Backward-compat: the legacy /stream route reads app.state.bridges
    # via deps; readiness probe in routes/health.py also looks here.
    app.state.bridges = bridges
    log.info("analytics_agent_ready")

    yield

    flush_langfuse()
    if hasattr(conversation_store, "disconnect"):
        await conversation_store.disconnect()
    for name, bridge in bridges.items():
        await bridge.disconnect()
        log.info("mcp_disconnected", server=name)


# --------------------------------------------------------------------
# create_app(deps) — pure factory, no I/O
# --------------------------------------------------------------------


def create_app(deps: AppDependencies) -> FastAPI:
    """Create and configure a FastAPI app with the provided dependencies.

    Pure function — no env reads, no I/O. The `deps` object is attached
    to `app.state.deps` so route handlers can access services via
    `request.app.state.deps`.
    """
    application = FastAPI(
        title="Analytics Agent",
        description="Enterprise Agentic Analytics Platform — LangGraph orchestrator",
        version="1.0.0",
        lifespan=lifespan,
    )

    application.state.deps = deps

    # CORS — tightened to avoid wildcard + credentials (invalid per CORS spec).
    raw_origins = os.getenv("ALLOWED_ORIGINS")
    if raw_origins and raw_origins != "*":
        allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
        allow_credentials = True
    else:
        # Wildcard + credentials is invalid per CORS spec — browsers reject it.
        # Default to wildcard origins with credentials disabled. Production must
        # set ALLOWED_ORIGINS to an explicit comma-separated list.
        allowed_origins = ["*"]
        allow_credentials = False
        log.warning(
            "cors_wildcard_no_credentials",
            hint="set ALLOWED_ORIGINS=https://your.dashboard.example to enable credentials",
        )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Domain → HTTP error handlers.
    @application.exception_handler(AuthError)
    async def _on_auth_error(request: Request, exc: AuthError):
        return JSONResponse(
            {"error_id": uuid.uuid4().hex, "type": "auth", "message": str(exc)},
            status_code=401,
        )

    @application.exception_handler(ConversationNotFound)
    async def _on_not_found(request: Request, exc: ConversationNotFound):
        return JSONResponse(
            {"error_id": uuid.uuid4().hex, "type": "not_found", "message": str(exc)},
            status_code=404,
        )

    @application.exception_handler(AnalyticsError)
    async def _on_analytics_error(request: Request, exc: AnalyticsError):
        return JSONResponse(
            {"error_id": uuid.uuid4().hex, "type": "internal"},
            status_code=500,
        )

    application.include_router(health_router)
    application.include_router(chat_router)
    application.include_router(conversations_router)
    application.include_router(stream_router)

    return application


# --------------------------------------------------------------------
# Module-level entry point — uvicorn loads this `app` symbol
# --------------------------------------------------------------------


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
app = create_app(_empty_deps)
