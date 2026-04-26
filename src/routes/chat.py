"""POST /api/v1/analytics/chat — Vercel AI SDK Data Stream Protocol endpoint.

Thin handler: extracts UserContext from headers, pulls the
chat_service_factory from app.state.deps, builds a ChatService for
this request, and returns a StreamingResponse over its execute()
generator. The actual orchestration (graph events, encoder, persistence)
lives entirely in ChatService — Phase 5.

Body shape mirrors the Vercel AI SDK ``useChat`` payload:
    {"id": "<conversation-id>", "messages": [{"role": ..., "content": ...}, ...]}

The route extracts the latest user message and the recent history (last
N turns) and packages them into a ``ChatRequest`` for ChatService.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..domain.types import ChatRequest
from ._auth import _user_ctx_from

chat_router = APIRouter(prefix="/api/v1/analytics", tags=["chat"])


class VercelChatRequest(BaseModel):
    """Vercel AI SDK ``useChat`` body."""
    messages: list[dict] = Field(description="Chat messages array")
    id: Optional[str] = Field(default=None, description="Conversation/Chat ID")


# Last N raw messages forwarded to the graph as history (excluding the
# current message). Five turns ≈ ten user/assistant pairs on average.
_MAX_HISTORY_MESSAGES = 10


def _last_user_message(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "") or ""
    return ""


def _recent_history(messages: list[dict]) -> list[dict]:
    history: list[dict] = []
    # Drop the current (last) message and keep the prior _MAX_HISTORY_MESSAGES.
    for msg in messages[-(_MAX_HISTORY_MESSAGES + 1):-1]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            history.append({"role": role, "content": content})
    return history


@chat_router.post("/chat")
async def chat(body: VercelChatRequest, request: Request) -> StreamingResponse:
    user_ctx = _user_ctx_from(request)  # AuthError → 401

    deps = getattr(request.app.state, "deps", None)
    if deps is None or getattr(deps, "chat_service_factory", None) is None:
        raise HTTPException(503, "Chat service not initialised")

    user_message = _last_user_message(body.messages)
    if not user_message:
        raise HTTPException(422, "No user message found in messages array")

    req = ChatRequest(
        message=user_message,
        conversation_id=body.id,
        history=_recent_history(body.messages),
    )

    service = deps.chat_service_factory(user_ctx)
    return StreamingResponse(
        service.execute(req),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Vercel-AI-Data-Stream": "v1",
        },
    )
