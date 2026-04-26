"""ChatService — orchestrates graph invocation, streaming, and persistence.

Constructor-injected. Per-request UserContext is held on the instance.
No module-level state. Emits bytes via the injected StreamEncoder.

This is the new, decoupled implementation introduced in Phase 5 of the
SoC/DI refactor. It is purely additive — the existing inline chat
orchestration in src/app.py continues to run for every request today.
Phase 7 will replace those inline routes with thin wrappers around
ChatService.execute, and Phase 4.3's lifespan will populate
AppDependencies.chat_service_factory with `lambda user_ctx: ChatService(...)`.
"""
from __future__ import annotations

import uuid
from typing import Any, AsyncIterator, Callable, Optional

from platform_sdk import AgentConfig, get_langfuse_callback_handler, get_logger

from ..domain.errors import AnalyticsError
from ..domain.types import ChatRequest, UserContext
from ..ports import ConversationStore, StreamEncoder, TelemetryScope
from ..thread_id import make_thread_id

log = get_logger(__name__)


class ChatService:
    """One per request. Owns graph lifecycle, streaming, persistence.

    Constructor-injected; the per-request UserContext is held on the
    instance so route handlers can build a service via:

        service = deps.chat_service_factory(user_ctx)

    and then iterate ``service.execute(req)`` to get the wire bytes.
    """

    def __init__(
        self,
        *,
        graph: Any,
        conversation_store: ConversationStore,
        config: AgentConfig,
        user_ctx: UserContext,
        encoder_factory: Callable[[], StreamEncoder],
        telemetry: TelemetryScope,
    ) -> None:
        self._graph = graph
        self._store = conversation_store
        self._config = config
        self._user_ctx = user_ctx
        self._encoder_factory = encoder_factory
        self._telemetry = telemetry

    async def execute(self, req: ChatRequest) -> AsyncIterator[bytes]:
        """Execute one chat request; yield encoded wire bytes.

        On graph success: streams events as they arrive, captures the
        final narrative + components emitted by the synthesis node, and
        persists user/assistant messages to the conversation store
        after the stream completes. On failure: encodes a single error
        event, finalises the encoder, and exits cleanly — never re-raises
        across the streaming boundary.
        """
        encoder = self._encoder_factory()
        session_id = req.conversation_id or str(uuid.uuid4())
        thread_id = make_thread_id(self._user_ctx.user_id, session_id)

        captured_narrative = ""
        captured_components: list[dict] = []
        error_id: Optional[str] = None

        with self._telemetry.start_span("chat.execute"):
            try:
                # Ensure conversation exists (if the store supports it).
                if hasattr(self._store, "get_conversation"):
                    conv = await self._store.get_conversation(session_id)
                    if not conv and hasattr(self._store, "create_conversation"):
                        await self._store.create_conversation(
                            session_id, "New Conversation"
                        )

                graph_messages = list(req.history) + [
                    {"role": "user", "content": req.message}
                ]
                run_config: dict[str, Any] = {
                    "configurable": {
                        "user_ctx": self._user_ctx,
                        "thread_id": thread_id,
                        "session_id": session_id,
                    }
                }
                lf_handler = get_langfuse_callback_handler(
                    session_id=session_id,
                    user_id=self._user_ctx.user_id,
                    trace_name="analytics-agent.chat",
                )
                if lf_handler is not None:
                    run_config["callbacks"] = [lf_handler]

                async for event in self._graph.astream_events(
                    {"messages": graph_messages, "session_id": session_id},
                    config=run_config,
                    version="v2",
                ):
                    yield encoder.encode_event(event)

                    if (
                        event.get("event") == "on_chain_end"
                        and event.get("name") == "synthesis"
                    ):
                        out = event.get("data", {}).get("output", {}) or {}
                        captured_narrative = out.get(
                            "narrative", captured_narrative
                        )
                        captured_components = out.get(
                            "ui_components", captured_components
                        )

                yield encoder.finalize()

                if captured_narrative:
                    try:
                        if hasattr(self._store, "add_message"):
                            await self._store.add_message(
                                session_id, "user", req.message, components=None
                            )
                            await self._store.add_message(
                                session_id,
                                "assistant",
                                captured_narrative,
                                components=captured_components or None,
                            )
                    except Exception as exc:
                        log.error(
                            "conversation_persistence_failed",
                            error=str(exc),
                            conversation_id=session_id,
                        )

            except AnalyticsError as exc:
                error_id = uuid.uuid4().hex
                log.error(
                    "chat_service_error",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    error_id=error_id,
                    user_id=self._user_ctx.user_id,
                )
                yield encoder.encode_error(exc, error_id=error_id)
                yield encoder.finalize()
            except Exception as exc:
                error_id = uuid.uuid4().hex
                log.error(
                    "chat_service_unexpected_error",
                    error=str(exc),
                    error_id=error_id,
                )
                yield encoder.encode_error(exc, error_id=error_id)
                yield encoder.finalize()
