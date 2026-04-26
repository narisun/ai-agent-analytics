"""Legacy SSE streaming endpoint — kept for backwards compatibility.

This endpoint pre-dates the Vercel Data Stream Protocol (`/chat`) and
uses a custom SSE format (`event: <type>\\ndata: <json>\\n\\n`). The
dashboard no longer uses it, but it remains for any third-party
clients still pointed at the legacy URL.

Authentication is permissive (anonymous fallback) to preserve the
pre-Phase-7 behavior. New endpoints use `_user_ctx_from(request)`
which raises on missing X-User-Email.
"""
from __future__ import annotations

import json
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from platform_sdk import get_logger
from ..thread_id import make_thread_id

log = get_logger(__name__)

stream_router = APIRouter(prefix="/api/v1/analytics", tags=["stream"])


class StreamRequest(BaseModel):
    """Legacy SSE request body."""
    session_id: str = Field(description="Conversation thread ID (UUID)")
    message: str = Field(min_length=1, description="User's analytics query")


def _sse_event(event_type: str, data: dict) -> str:
    """Format a single SSE event string."""
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


@stream_router.post("/stream")
async def stream_analytics(body: StreamRequest, request: Request):
    """Legacy SSE endpoint — streams graph events as Server-Sent Events.

    Emits one SSE event per LangGraph event:
      - tool_call_start: node execution begins
      - tool_call_end:   node execution ends (with results)
      - ui_component:    chart/KPI/table schema (from synthesis)
      - error:           error details
      - end:             stream complete
    """
    deps = getattr(request.app.state, "deps", None)
    graph = getattr(deps, "graph", None) if deps else None
    if graph is None:
        raise HTTPException(503, "Graph not initialised")

    # Permissive auth — preserves pre-Phase-7 anonymous fallback behavior.
    user_email = (request.headers.get("x-user-email", "") or "anonymous").strip().lower()

    async def event_generator():
        active_tool_calls: dict[str, str] = {}
        try:
            async for event in graph.astream_events(
                {
                    "messages": [{"role": "user", "content": body.message}],
                    "session_id": body.session_id,
                },
                config={
                    "configurable": {
                        "thread_id": make_thread_id(user_email, body.session_id),
                    },
                },
                version="v2",
            ):
                event_type = event.get("event", "")
                event_name = event.get("name", "")

                if event_type == "on_chain_start" and event_name in (
                    "intent_router", "mcp_tool_caller", "synthesis", "error_handler",
                ):
                    yield _sse_event("tool_call_start", {
                        "node": event_name,
                        "run_id": str(event.get("run_id", "")),
                    })
                elif event_type == "on_chain_end" and event_name in (
                    "intent_router", "mcp_tool_caller", "synthesis", "error_handler",
                ):
                    output = event.get("data", {}).get("output", {}) or {}

                    if event_name == "intent_router" and "intent_reasoning" in output:
                        yield _sse_event("tool_call_end", {
                            "node": event_name,
                            "intent": output.get("intent"),
                            "reasoning": output.get("intent_reasoning"),
                            "plan_steps": len(output.get("query_plan", [])),
                        })
                    elif event_name == "mcp_tool_caller" and "active_tools" in output:
                        yield _sse_event("tool_call_end", {
                            "node": event_name,
                            "tools": output.get("active_tools", []),
                        })
                    elif event_name == "synthesis" and "ui_components" in output:
                        yield _sse_event("tool_call_end", {"node": event_name})
                        for component in output.get("ui_components", []):
                            yield _sse_event("ui_component", component)

            yield _sse_event("end", {})

        except Exception as exc:
            log.error("legacy_stream_error", error=str(exc), session=body.session_id)
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
