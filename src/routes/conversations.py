"""Conversation CRUD endpoints — thin layer over the conversation store.

The store accessed here is the LEGACY concrete store
(MemoryConversationStore or PostgresConversationStore) on
``request.app.state.deps.conversation_store``. Those stores have
imperative methods (``list_conversations``, ``get_conversation``,
``create_conversation``, ``update_title``, ``delete_conversation``)
that pre-date the Phase 1 ``ConversationStore`` Protocol.

The new ``ConversationService`` (Phase 7 Sub-B) satisfies the Protocol
shape but the concrete stores do not yet — Phase 9 will migrate them.
Until then, this module talks to the stores directly so the existing
dashboard's CRUD continues to work without behavioral change.

Authentication: every endpoint calls ``_user_ctx_from(request)`` so an
unauthenticated request gets ``AuthError`` → 401 (registered as an
exception handler by ``create_app`` in Phase 7 Sub-E). The current
legacy stores are NOT tenant-scoped; Phase 9 will add tenant filters
once the stores satisfy the Protocol.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from platform_sdk import get_logger
from ._auth import _user_ctx_from

log = get_logger(__name__)


class CreateConversationRequest(BaseModel):
    id: str = Field(description="UUID for the new conversation")
    title: str = Field(min_length=1, description="Initial title")


class UpdateConversationRequest(BaseModel):
    title: str = Field(min_length=1, description="New title")


conversations_router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


def _store(request: Request):
    """Pull the conversation store off app.state.deps; raise 503 if missing."""
    deps = getattr(request.app.state, "deps", None)
    if deps is None or getattr(deps, "conversation_store", None) is None:
        raise HTTPException(503, "Conversation store not initialised")
    return deps.conversation_store


@conversations_router.get("")
async def list_conversations(request: Request) -> dict:
    _ = _user_ctx_from(request)  # AuthError → 401 via registered handler
    store = _store(request)
    try:
        return {"conversations": await store.list_conversations(limit=100)}
    except Exception as exc:
        log.error("list_conversations_error", error=str(exc))
        raise HTTPException(500, str(exc))


@conversations_router.get("/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request):
    _ = _user_ctx_from(request)
    store = _store(request)
    try:
        conv = await store.get_conversation(conversation_id)
    except HTTPException:
        raise
    except Exception as exc:
        log.error(
            "get_conversation_error",
            error=str(exc), conversation_id=conversation_id,
        )
        raise HTTPException(500, str(exc))
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return conv


@conversations_router.post("")
async def create_conversation(body: CreateConversationRequest, request: Request):
    _ = _user_ctx_from(request)
    store = _store(request)
    try:
        conv = await store.create_conversation(body.id, body.title)
        return JSONResponse(content=conv, status_code=201)
    except Exception as exc:
        log.error(
            "create_conversation_error",
            error=str(exc), conversation_id=body.id,
        )
        raise HTTPException(500, str(exc))


@conversations_router.patch("/{conversation_id}")
async def update_conversation(
    conversation_id: str, body: UpdateConversationRequest, request: Request,
):
    _ = _user_ctx_from(request)
    store = _store(request)
    try:
        await store.update_title(conversation_id, body.title)
        conv = await store.get_conversation(conversation_id)
    except HTTPException:
        raise
    except Exception as exc:
        log.error(
            "update_conversation_error",
            error=str(exc), conversation_id=conversation_id,
        )
        raise HTTPException(500, str(exc))
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return conv


@conversations_router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, request: Request):
    _ = _user_ctx_from(request)
    store = _store(request)
    try:
        await store.delete_conversation(conversation_id)
    except Exception as exc:
        log.error(
            "delete_conversation_error",
            error=str(exc), conversation_id=conversation_id,
        )
        raise HTTPException(500, str(exc))
    return None
