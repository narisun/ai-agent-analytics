"""
Analytics Agent — FastAPI application with SSE streaming.

Exposes:
  POST /api/v1/analytics/stream  — SSE endpoint (legacy custom SSE protocol)
  POST /api/v1/analytics/chat    — Vercel AI SDK Data Stream Protocol endpoint
  GET  /health                   — Basic liveness probe
  GET  /health/ready             — Readiness probe (checks MCP bridge connectivity)

Data Stream Protocol (for /chat endpoint):
  0:"text"                — Narrative tokens
  g:"reasoning"           — Reasoning/thinking tokens (intent, tool calls)
  b:{toolCallId,toolName} — Tool call begin
  a:{toolCallId,result}   — Tool call result
  2:[{...component}]      — Custom data (UI components)
  3:"error"               — Error message
  d:{finishReason}        — Stream complete
"""
import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from platform_sdk import AgentConfig, AgentContext, configure_logging, get_logger, setup_checkpointer, setup_telemetry, flush_langfuse
from platform_sdk.mcp_bridge import MCPToolBridge, set_user_auth_token, reset_user_auth_token
from platform_sdk.security import make_api_key_verifier
from .graph import build_analytics_graph
from .middleware.rate_limiter import make_rate_limiter
from .persistence import PostgresConversationStore, MemoryConversationStore
from .thread_id import make_thread_id

configure_logging()
log = get_logger(__name__)


# Conversation store factory

def _make_conversation_store():
    """Create the appropriate conversation store based on environment.

    Returns PostgresConversationStore if DATABASE_URL is set, ENVIRONMENT is not 'local',
    and asyncpg is available. Falls back to MemoryConversationStore otherwise.
    """
    db_url = os.getenv("DATABASE_URL")
    if db_url and os.getenv("ENVIRONMENT") not in ("local",) and PostgresConversationStore is not None:
        return PostgresConversationStore(db_url)
    if db_url and PostgresConversationStore is None:
        log.warning("asyncpg_not_available", fallback="MemoryConversationStore")
    return MemoryConversationStore()


# Request / Response models

class ChatRequest(BaseModel):
    """Incoming chat request from the analytics UI."""
    session_id: str = Field(description="Conversation thread ID (UUID)")
    message: str = Field(min_length=1, description="User's analytics query")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    bridges: Optional[dict] = None


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    id: str = Field(description="UUID for the new conversation")
    title: str = Field(min_length=1, description="Initial conversation title")


class UpdateConversationRequest(BaseModel):
    """Request to update conversation metadata."""
    title: str = Field(min_length=1, description="New conversation title")


# Application lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize MCP bridges and build the analytics graph on startup."""
    setup_telemetry("analytics-agent")
    config = AgentConfig.from_env()

    # MCP server URLs from environment
    mcp_urls = {
        "data-mcp": os.getenv("DATA_MCP_URL", "http://data-mcp:8000/sse"),
        "salesforce-mcp": os.getenv("SALESFORCE_MCP_URL", "http://salesforce-mcp:8000/sse"),
        "payments-mcp": os.getenv("PAYMENTS_MCP_URL", "http://payments-mcp:8000/sse"),
        "news-search-mcp": os.getenv("NEWS_MCP_URL", "http://news-search-mcp:8000/sse"),
    }

    # Build service-level agent context for MCP SSE connections.
    # This identifies the AGENT SERVICE (not the user).  User identity is
    # carried per-request via the HMAC-signed auth_context token.
    agent_context = AgentContext(
        rm_id="analytics-agent",
        rm_name="Analytics Agent",
        role="manager",
        team_id="analytics",
        assigned_account_ids=(),
        compliance_clearance=("standard", "aml_view"),
    )

    # Connect to all MCP servers in parallel
    bridges = {}
    startup_timeout = config.mcp_startup_timeout
    for name, url in mcp_urls.items():
        bridges[name] = MCPToolBridge(url, agent_context=agent_context)

    log.info("mcp_connecting_all", servers=list(mcp_urls.keys()), timeout=startup_timeout)
    await asyncio.gather(
        *[bridge.connect(startup_timeout=startup_timeout) for bridge in bridges.values()],
        return_exceptions=True,
    )

    for name, bridge in bridges.items():
        log.info("mcp_startup_status", server=name, connected=bridge.is_connected)

    # Initialise the checkpointer (creates Postgres tables if needed) before
    # compiling the graph.  Using setup_checkpointer() ensures checkpoint tables
    # exist before the first multi-turn request.
    checkpointer = await setup_checkpointer(config)

    # Build the compiled graph
    graph = build_analytics_graph(bridges=bridges, config=config, checkpointer=checkpointer)

    # Initialize conversation store (PostgreSQL or in-memory)
    store = _make_conversation_store()
    if hasattr(store, "connect"):
        await store.connect()
    app.state.conversation_store = store

    app.state.graph = graph
    app.state.bridges = bridges
    app.state.config = config
    log.info("analytics_agent_ready")

    yield

    # Shutdown: flush LangFuse traces before disconnecting
    flush_langfuse()

    # Shutdown: disconnect conversation store if needed
    if hasattr(store, "disconnect"):
        await store.disconnect()

    # Shutdown: disconnect all bridges
    for name, bridge in bridges.items():
        await bridge.disconnect()
        log.info("mcp_disconnected", server=name)


# FastAPI app

app = FastAPI(
    title="Analytics Agent",
    description="Enterprise Agentic Analytics Platform — LangGraph orchestrator",
    version="1.0.0",
    lifespan=lifespan,
)

# TODO(production): Replace allow_origins=["*"] with specific origins.
# Wildcard + credentials is invalid per CORS spec — browsers will reject it.
# Use env var ALLOWED_ORIGINS to configure per environment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication and rate limiting dependencies
verify_api_key = make_api_key_verifier()
check_rate_limit = make_rate_limiter()


# Health endpoints

@app.get("/health")
async def health():
    """Basic liveness probe."""
    return HealthResponse(status="ok")


@app.get("/health/ready")
async def health_ready():
    """Readiness probe — checks MCP bridge connectivity."""
    bridges = getattr(app.state, "bridges", {})
    bridge_status = {name: bridge.is_connected for name, bridge in bridges.items()}
    all_connected = all(bridge_status.values()) if bridge_status else False
    return HealthResponse(
        status="ready" if all_connected else "degraded",
        bridges=bridge_status,
    )


# SSE endpoint (legacy)

@app.post("/api/v1/analytics/stream")
async def stream_analytics(
    request: ChatRequest,
    _token: str = Depends(verify_api_key),
    _rate: None = Depends(check_rate_limit),
):
    """Stream analytics results via Server-Sent Events.

    Requires JWT authentication and respects per-session rate limits (20 req/min by default).

    Consumes the LangGraph astream_events(v2) protocol and translates
    internal graph events into the frontend SSE contract:
      - text: narrative tokens
      - tool_call_start: node execution begins
      - tool_call_end: node execution ends
      - ui_component: chart/KPI/table schema
      - error: error details
      - end: stream complete
    """
    graph = app.state.graph

    async def event_generator():
        try:
            async for event in graph.astream_events(
                {
                    "messages": [{"role": "user", "content": request.message}],
                    "session_id": request.session_id,
                },
                config={
                    "configurable": {"thread_id": request.session_id},
                },
                version="v2",
            ):
                event_type = event.get("event", "")
                event_name = event.get("name", "")

                # Emit node execution start for trace panel
                if event_type == "on_chain_start" and event_name in (
                    "intent_router", "mcp_tool_caller", "synthesis", "error_handler"
                ):
                    yield _sse_event("tool_call_start", {
                        "node": event_name,
                        "run_id": str(event.get("run_id", "")),
                    })

                # Emit node execution end with results
                elif event_type == "on_chain_end" and event_name in (
                    "intent_router", "mcp_tool_caller", "synthesis", "error_handler"
                ):
                    output = event.get("data", {}).get("output", {})

                    # Emit intent reasoning for the trace panel
                    if event_name == "intent_router" and "intent_reasoning" in output:
                        yield _sse_event("tool_call_end", {
                            "node": event_name,
                            "intent": output.get("intent"),
                            "reasoning": output.get("intent_reasoning"),
                            "plan_steps": len(output.get("query_plan", [])),
                        })

                    # Emit active tools for trace panel
                    elif event_name == "mcp_tool_caller" and "active_tools" in output:
                        yield _sse_event("tool_call_end", {
                            "node": event_name,
                            "tools": output.get("active_tools", []),
                        })

                    # Emit UI components for the canvas
                    elif event_name == "synthesis" and "ui_components" in output:
                        yield _sse_event("tool_call_end", {"node": event_name})
                        for component in output.get("ui_components", []):
                            yield _sse_event("ui_component", component)

                    else:
                        yield _sse_event("tool_call_end", {"node": event_name})

                # Stream narrative text from LLM tokens
                elif event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield _sse_event("text", {"token": chunk.content})

            # Signal stream completion
            yield _sse_event("end", {})

        except Exception as exc:
            log.error("stream_error", error=str(exc), session=request.session_id)
            yield _sse_event("error", {"message": str(exc)})
            yield _sse_event("end", {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_event(event_type: str, data: dict) -> str:
    """Format a single SSE event string."""
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


# Data Stream Protocol endpoint

class VercelChatRequest(BaseModel):
    """Vercel AI SDK useChat request format."""
    messages: list[dict] = Field(description="Chat messages array from useChat")
    id: Optional[str] = Field(default=None, description="Chat ID from useChat")


def _ds_text(token: str) -> str:
    """Data Stream Protocol: text token (prefix 0:)."""
    return f'0:{json.dumps(token)}\n'


def _ds_reasoning(text: str) -> str:
    """Data Stream Protocol: reasoning token (prefix g:)."""
    return f'g:{json.dumps(text)}\n'


def _ds_tool_call_begin(tool_call_id: str, tool_name: str) -> str:
    """Data Stream Protocol: tool call start (prefix b:)."""
    return f'b:{json.dumps({"toolCallId": tool_call_id, "toolName": tool_name})}\n'


def _ds_tool_result(tool_call_id: str, result: str) -> str:
    """Data Stream Protocol: tool result (prefix a:)."""
    return f'a:{json.dumps({"toolCallId": tool_call_id, "result": result})}\n'


def _ds_data(payload: list) -> str:
    """Data Stream Protocol: custom data message (prefix 2:)."""
    return f'2:{json.dumps(payload, default=str)}\n'


def _ds_error(message: str) -> str:
    """Data Stream Protocol: error (prefix 3:)."""
    return f'3:{json.dumps(message)}\n'


def _ds_finish(reason: str = "stop") -> str:
    """Data Stream Protocol: finish (prefix d:)."""
    return f'd:{json.dumps({"finishReason": reason, "usage": {"promptTokens": 0, "completionTokens": 0}})}\n'


@app.post("/api/v1/analytics/chat")
async def chat_analytics(
    raw_request: Request,
    request: VercelChatRequest,
    _token: str = Depends(verify_api_key),
    _rate: None = Depends(check_rate_limit),
):
    """Vercel AI SDK Data Stream Protocol endpoint.

    Requires JWT authentication and respects per-session rate limits (20 req/min by default).

    Compatible with useChat({ streamProtocol: 'data' }) from @ai-sdk/react.
    Consumes the same LangGraph astream_events(v2) and maps them to the
    line-based Data Stream Protocol format.

    Auth flow:
      Dashboard extracts Auth0 session → X-User-Email + X-User-Role headers.
      These are forwarded into the graph state and injected into MCP tool
      calls so OPA can evaluate per-user authorization.

    Conversation history: The Vercel AI SDK sends the full messages array on
    every request. We pass recent history to the graph so the intent router
    can distinguish follow-ups from new queries. The checkpointer handles
    long-term state persistence.
    """
    graph = app.state.graph

    # Extract authenticated user identity from headers (set by dashboard API route).
    # The dashboard authenticates via Auth0; these headers are trusted because this
    # endpoint requires INTERNAL_API_KEY (service-to-service auth).
    user_email = raw_request.headers.get("x-user-email", "anonymous")
    user_role = raw_request.headers.get("x-user-role", "")
    log.info("chat_user_context", user_email=user_email, user_role=user_role)

    # Build per-request AgentContext carrying user identity and HMAC-sign it.
    # This signed token flows to every MCP tool call via the bridge ContextVar,
    # ensuring user_role cannot be forged by the LLM or any intermediate caller.
    per_request_ctx = AgentContext(
        rm_id=user_email,
        rm_name=user_email.split("@")[0] if "@" in user_email else user_email,
        role="manager",
        team_id="analytics",
        assigned_account_ids=(),
        compliance_clearance=("standard", "aml_view"),
        user_email=user_email,
        user_role=user_role,
    )
    auth_ctx_token = set_user_auth_token(per_request_ctx.to_header_value())

    # Extract latest user message
    user_message = ""
    for msg in reversed(request.messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    if not user_message:
        return StreamingResponse(
            iter([_ds_error("No user message found"), _ds_finish("error")]),
            media_type="text/plain; charset=utf-8",
        )

    session_id = request.id or str(uuid.uuid4())
    active_tool_calls: dict[str, str] = {}

    # Build conversation history from the Vercel AI SDK messages array.
    # Include recent history (last N turns) so the intent router sees context
    # for follow-up classification. The checkpointer has the full history.
    MAX_HISTORY_MESSAGES = 10  # Last 5 turns (user + assistant pairs)
    history_messages = []
    for msg in request.messages[-(MAX_HISTORY_MESSAGES + 1):-1]:  # Exclude last (current) msg
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            history_messages.append({"role": role, "content": content})

    # Graph input: recent history + current user message
    graph_messages = history_messages + [{"role": "user", "content": user_message}]

    # Node tracking for routing LLM tokens as reasoning vs narrative
    GRAPH_NODES = {"intent_router", "mcp_tool_caller", "synthesis", "error_handler"}
    REASONING_NODES = {"intent_router", "mcp_tool_caller", "error_handler"}

    # Accumulator for captured output to persist after stream
    captured_narrative = ""
    captured_components: list[dict] = []
    stream_error: Optional[str] = None

    async def data_stream_generator():
        nonlocal captured_narrative, captured_components, stream_error
        current_node: str | None = None
        synthesis_streamed = False

        try:
            async for event in graph.astream_events(
                {
                    "messages": graph_messages,
                    "session_id": session_id,
                },
                config={
                    "configurable": {
                        "thread_id": make_thread_id(user_email, session_id),
                    },
                },
                version="v2",
            ):
                event_type = event.get("event", "")
                event_name = event.get("name", "")

                # Track active node and emit reasoning on start
                if event_type == "on_chain_start" and event_name in GRAPH_NODES:
                    current_node = event_name
                    if event_name == "intent_router":
                        yield _ds_reasoning("Classifying intent...\n")
                    elif event_name == "mcp_tool_caller":
                        tc_id = f"tc_{uuid.uuid4().hex[:8]}"
                        active_tool_calls[event_name] = tc_id
                        yield _ds_reasoning("Executing data retrieval...\n")
                    elif event_name == "synthesis":
                        yield _ds_reasoning("Generating analysis...\n")

                # Emit results and clear active node on end
                elif event_type == "on_chain_end" and event_name in GRAPH_NODES:
                    output = event.get("data", {}).get("output", {})

                    if event_name == "intent_router" and "intent_reasoning" in output:
                        intent = output.get("intent", "unknown")
                        reasoning = output.get("intent_reasoning", "")
                        plan_steps = len(output.get("query_plan", []))
                        yield _ds_reasoning(
                            f"Intent: {intent} | {reasoning} | Plan: {plan_steps} steps\n"
                        )
                        for step in output.get("query_plan", []):
                            tc_id = f"tc_{uuid.uuid4().hex[:8]}"
                            tool_name = step.get("tool_name", "unknown")
                            server = step.get("mcp_server", "unknown")
                            active_tool_calls[tool_name] = tc_id
                            yield _ds_tool_call_begin(tc_id, f"{server}/{tool_name}")

                            # Emit SQL queries so the frontend can display them
                            if tool_name == "execute_read_query":
                                sql = step.get("parameters", {}).get("query", "")
                                if sql:
                                    yield _ds_reasoning(f"__SQL_QUERY__:{sql}:__END_SQL__\n")

                    elif event_name == "mcp_tool_caller":
                        tools = output.get("active_tools", [])
                        for tool_info in tools:
                            tool_name = tool_info.get("tool", "") if isinstance(tool_info, dict) else str(tool_info)
                            tc_id = active_tool_calls.get(tool_name, f"tc_{uuid.uuid4().hex[:8]}")
                            yield _ds_tool_result(tc_id, json.dumps({"status": "complete", "tool": tool_name}, default=str))

                        # Extract _sql_queries from MCP tool results for UI display
                        raw_data = output.get("raw_data_context", {})
                        for _rk, rv in raw_data.items():
                            try:
                                parsed = json.loads(rv) if isinstance(rv, str) else rv
                                if isinstance(parsed, dict) and "_sql_queries" in parsed:
                                    for sq in parsed["_sql_queries"]:
                                        yield _ds_reasoning(f"__SQL_QUERY__:{sq}:__END_SQL__\n")
                            except (json.JSONDecodeError, TypeError):
                                pass

                        # Stream any MCP errors as visible text
                        mcp_errors = output.get("errors", [])
                        if mcp_errors:
                            for err in mcp_errors:
                                yield _ds_reasoning(f"Tool error: {err}\n")

                    elif event_name == "synthesis":
                        # Emit UI components as data messages
                        components = output.get("ui_components", [])
                        if components:
                            yield _ds_data(components)
                            captured_components.extend(components)
                        # Emit follow-up suggestions as a typed data event so the
                        # frontend can render them below the assistant message.
                        follow_ups = output.get("follow_up_suggestions", [])
                        if follow_ups:
                            yield _ds_data([{
                                "type": "follow_up_suggestions",
                                "suggestions": follow_ups,
                            }])
                        # Emit narrative as text if not already streamed via on_chat_model_stream
                        narrative = output.get("narrative", "")
                        if narrative and not synthesis_streamed:
                            yield _ds_text(narrative)
                            captured_narrative = narrative

                    elif event_name == "error_handler":
                        # Stream errors as visible text to the user
                        narrative = output.get("narrative", "")
                        errors = output.get("errors", [])
                        if narrative:
                            yield _ds_text(narrative)
                            captured_narrative = narrative
                        elif errors:
                            error_msg = f"I encountered an issue: {'; '.join(str(e) for e in errors)}"
                            yield _ds_text(error_msg)
                            captured_narrative = error_msg
                        # Emit follow-up suggestions so the user has guided recovery options
                        follow_ups = output.get("follow_up_suggestions", [])
                        if follow_ups:
                            yield _ds_data([{
                                "type": "follow_up_suggestions",
                                "suggestions": follow_ups,
                            }])

                    current_node = None

                # Route LLM tokens as reasoning or narrative based on active node
                elif event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        token = chunk.content
                        if current_node in REASONING_NODES:
                            yield _ds_reasoning(token)
                        else:
                            yield _ds_text(token)
                            captured_narrative += token
                            synthesis_streamed = True

            yield _ds_finish("stop")

        except Exception as exc:
            log.error("data_stream_error", error=str(exc), session=session_id)
            stream_error = str(exc)
            yield _ds_error(str(exc))
            yield _ds_finish("error")

    async def stream_with_persistence():
        """Wrapper that streams output and persists conversation after completion."""
        store = app.state.conversation_store

        # Ensure conversation exists
        conv = await store.get_conversation(session_id)
        if not conv:
            await store.create_conversation(session_id, "New Conversation")

        # Stream the response to the client
        async for chunk in data_stream_generator():
            yield chunk

        # After stream completes, persist the user message and assistant response
        try:
            if not stream_error:  # Only persist on success
                # Persist user message
                await store.add_message(
                    session_id,
                    "user",
                    user_message,
                    components=None,
                )

                # Persist assistant response with captured narrative and components
                if captured_narrative:
                    await store.add_message(
                        session_id,
                        "assistant",
                        captured_narrative,
                        components=captured_components if captured_components else None,
                    )
                    log.info("conversation_persisted", conversation_id=session_id)
        except Exception as exc:
            log.error(
                "conversation_persistence_failed",
                error=str(exc),
                conversation_id=session_id,
            )

    async def stream_with_auth_cleanup():
        """Wrap the stream to ensure the auth ContextVar is reset after completion."""
        try:
            async for chunk in stream_with_persistence():
                yield chunk
        finally:
            reset_user_auth_token(auth_ctx_token)

    return StreamingResponse(
        stream_with_auth_cleanup(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Vercel-AI-Data-Stream": "v1",
        },
    )


# Conversation persistence endpoints

@app.get("/api/v1/conversations")
async def list_conversations(_: str = Depends(verify_api_key)):
    """List all conversations, sorted by most recently updated.

    Requires API key authentication.
    """
    store = app.state.conversation_store
    try:
        conversations = await store.list_conversations(limit=100)
        return {"conversations": conversations}
    except Exception as exc:
        log.error("list_conversations_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, _: str = Depends(verify_api_key)):
    """Get a single conversation with full message history.

    Requires API key authentication.
    """
    store = app.state.conversation_store
    try:
        conv = await store.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv
    except HTTPException:
        raise
    except Exception as exc:
        log.error("get_conversation_error", error=str(exc), conversation_id=conversation_id)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/conversations")
async def create_conversation(
    body: CreateConversationRequest,
    _: str = Depends(verify_api_key),
):
    """Create a new conversation.

    Requires API key authentication.
    """
    store = app.state.conversation_store
    try:
        conv = await store.create_conversation(body.id, body.title)
        return JSONResponse(content=conv, status_code=201)
    except Exception as exc:
        log.error(
            "create_conversation_error",
            error=str(exc),
            conversation_id=body.id,
        )
        raise HTTPException(status_code=500, detail=str(exc))


@app.patch("/api/v1/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    body: UpdateConversationRequest,
    _: str = Depends(verify_api_key),
):
    """Update a conversation's title.

    Requires API key authentication.
    """
    store = app.state.conversation_store
    try:
        await store.update_title(conversation_id, body.title)
        conv = await store.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv
    except HTTPException:
        raise
    except Exception as exc:
        log.error(
            "update_conversation_error",
            error=str(exc),
            conversation_id=conversation_id,
        )
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/v1/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, _: str = Depends(verify_api_key)):
    """Delete a conversation and all its messages.

    Requires API key authentication.
    """
    store = app.state.conversation_store
    try:
        await store.delete_conversation(conversation_id)
        return JSONResponse(content={"status": "deleted"}, status_code=204)
    except Exception as exc:
        log.error(
            "delete_conversation_error",
            error=str(exc),
            conversation_id=conversation_id,
        )
        raise HTTPException(status_code=500, detail=str(exc))
