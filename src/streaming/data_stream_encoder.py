"""DataStreamEncoder — Vercel AI SDK Data Stream Protocol wire-format encoder.

Translates LangGraph event dicts (from astream_events(version='v2')) into
the line-based protocol bytes that the analytics dashboard consumes:

    0:"text"        — narrative tokens
    g:"reasoning"   — reasoning/thinking tokens
    b:{...}         — tool call begin
    a:{...}         — tool result
    2:[...]         — custom data (UI components, follow-up suggestions)
    3:"error"       — error
    d:{...}         — finish

The encoder is STATEFUL — it tracks the active graph node so chat-model
tokens can be routed to reasoning or narrative based on the node's role,
and tracks whether synthesis already streamed so the on_chain_end
narrative is not emitted twice.

This is a faithful port of the inline data_stream_generator that lived
in src/app.py before Phase 7. The only behavior change is the
factoring — the wire bytes are byte-for-byte identical.
"""
from __future__ import annotations

import json
import uuid
from typing import Any


# ---------------------------------------------------------------------------
# Frame helpers — exact port of _ds_* from src/app.py:313-353
# ---------------------------------------------------------------------------


def _ds_text(token: str) -> bytes:
    return f'0:{json.dumps(token)}\n'.encode()


def _ds_reasoning(text: str) -> bytes:
    return f'g:{json.dumps(text)}\n'.encode()


def _ds_tool_call_begin(tool_call_id: str, tool_name: str) -> bytes:
    return f'b:{json.dumps({"toolCallId": tool_call_id, "toolName": tool_name})}\n'.encode()


def _ds_tool_result(tool_call_id: str, result: str) -> bytes:
    return f'a:{json.dumps({"toolCallId": tool_call_id, "result": result})}\n'.encode()


def _ds_data(payload: list) -> bytes:
    return f'2:{json.dumps(payload, default=str)}\n'.encode()


def _ds_error(message: str) -> bytes:
    return f'3:{json.dumps(message)}\n'.encode()


def _ds_finish(reason: str = "stop") -> bytes:
    return f'd:{json.dumps({"finishReason": reason, "usage": {"promptTokens": 0, "completionTokens": 0}})}\n'.encode()


# ---------------------------------------------------------------------------
# DataStreamEncoder
# ---------------------------------------------------------------------------


_GRAPH_NODES = {"intent_router", "mcp_tool_caller", "synthesis", "error_handler"}
_REASONING_NODES = {"intent_router", "mcp_tool_caller", "error_handler"}


class DataStreamEncoder:
    """Stateful wire-format encoder. Construct one per request."""

    def __init__(self) -> None:
        self._current_node: str | None = None
        self._synthesis_streamed: bool = False
        # tool_name → tool_call_id, so on_chain_end of mcp_tool_caller can
        # match active_tools to the IDs emitted earlier in
        # on_chain_end of intent_router (which generates b: frames).
        self._active_tool_calls: dict[str, str] = {}

    def encode_event(self, event: dict[str, Any]) -> bytes:
        """Translate a LangGraph event into Data Stream Protocol bytes.

        Returns an empty bytes object when an event yields no wire output
        (e.g., events for nodes the protocol does not surface). Multiple
        wire frames produced by a single event are concatenated.
        """
        event_type = event.get("event", "")
        event_name = event.get("name", "")

        if event_type == "on_chain_start" and event_name in _GRAPH_NODES:
            return self._on_chain_start(event_name)

        if event_type == "on_chain_end" and event_name in _GRAPH_NODES:
            return self._on_chain_end(event_name, event)

        if event_type == "on_chat_model_stream":
            return self._on_chat_model_stream(event)

        return b""

    def encode_error(self, err: Exception, *, error_id: str) -> bytes:
        return _ds_error(str(err))

    def finalize(self) -> bytes:
        return _ds_finish("stop")

    # ------------------------------------------------------------------
    # Internal handlers — port of the if/elif chain in
    # src/app.py:data_stream_generator pre-Phase-7
    # ------------------------------------------------------------------

    def _on_chain_start(self, name: str) -> bytes:
        self._current_node = name
        if name == "intent_router":
            return _ds_reasoning("Classifying intent...\n")
        if name == "mcp_tool_caller":
            tc_id = f"tc_{uuid.uuid4().hex[:8]}"
            self._active_tool_calls[name] = tc_id
            return _ds_reasoning("Executing data retrieval...\n")
        if name == "synthesis":
            return _ds_reasoning("Generating analysis...\n")
        # error_handler has no on_start frame
        return b""

    def _on_chain_end(self, name: str, event: dict) -> bytes:
        output = event.get("data", {}).get("output", {}) or {}
        try:
            if name == "intent_router":
                return self._on_intent_router_end(output)
            if name == "mcp_tool_caller":
                return self._on_mcp_tool_caller_end(output)
            if name == "synthesis":
                return self._on_synthesis_end(output)
            if name == "error_handler":
                return self._on_error_handler_end(output)
            return b""
        finally:
            self._current_node = None

    def _on_intent_router_end(self, output: dict) -> bytes:
        if "intent_reasoning" not in output:
            return b""

        chunks: list[bytes] = []
        intent = output.get("intent", "unknown")
        reasoning = output.get("intent_reasoning", "")
        plan_steps = len(output.get("query_plan", []))
        chunks.append(_ds_reasoning(
            f"Intent: {intent} | {reasoning} | Plan: {plan_steps} steps\n"
        ))

        for step in output.get("query_plan", []):
            tc_id = f"tc_{uuid.uuid4().hex[:8]}"
            tool_name = step.get("tool_name", "unknown")
            server = step.get("mcp_server", "unknown")
            self._active_tool_calls[tool_name] = tc_id
            chunks.append(_ds_tool_call_begin(tc_id, f"{server}/{tool_name}"))

            # Surface SQL queries so the dashboard can render them.
            if tool_name == "execute_read_query":
                sql = step.get("parameters", {}).get("query", "")
                if sql:
                    chunks.append(_ds_reasoning(
                        f"__SQL_QUERY__:{sql}:__END_SQL__\n"
                    ))
        return b"".join(chunks)

    def _on_mcp_tool_caller_end(self, output: dict) -> bytes:
        chunks: list[bytes] = []

        for tool_info in output.get("active_tools", []):
            tool_name = tool_info.get("tool", "") if isinstance(tool_info, dict) else str(tool_info)
            tc_id = self._active_tool_calls.get(
                tool_name, f"tc_{uuid.uuid4().hex[:8]}"
            )
            chunks.append(_ds_tool_result(
                tc_id,
                json.dumps({"status": "complete", "tool": tool_name}, default=str),
            ))

        # Surface SQL queries from MCP tool results.
        raw_data = output.get("raw_data_context", {}) or {}
        for _rk, rv in raw_data.items():
            try:
                parsed = json.loads(rv) if isinstance(rv, str) else rv
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict):
                for sq in parsed.get("_sql_queries", []):
                    chunks.append(_ds_reasoning(
                        f"__SQL_QUERY__:{sq}:__END_SQL__\n"
                    ))

        for err in output.get("errors", []) or []:
            chunks.append(_ds_reasoning(f"Tool error: {err}\n"))

        return b"".join(chunks)

    def _on_synthesis_end(self, output: dict) -> bytes:
        chunks: list[bytes] = []

        components = output.get("ui_components", []) or []
        if components:
            chunks.append(_ds_data(components))

        follow_ups = output.get("follow_up_suggestions", []) or []
        if follow_ups:
            chunks.append(_ds_data([{
                "type": "follow_up_suggestions",
                "suggestions": follow_ups,
            }]))

        narrative = output.get("narrative", "") or ""
        if narrative and not self._synthesis_streamed:
            chunks.append(_ds_text(narrative))

        return b"".join(chunks)

    def _on_error_handler_end(self, output: dict) -> bytes:
        chunks: list[bytes] = []
        narrative = output.get("narrative", "") or ""
        errors = output.get("errors", []) or []
        if narrative:
            chunks.append(_ds_text(narrative))
        elif errors:
            chunks.append(_ds_text(
                f"I encountered an issue: {'; '.join(str(e) for e in errors)}"
            ))

        follow_ups = output.get("follow_up_suggestions", []) or []
        if follow_ups:
            chunks.append(_ds_data([{
                "type": "follow_up_suggestions",
                "suggestions": follow_ups,
            }]))
        return b"".join(chunks)

    def _on_chat_model_stream(self, event: dict) -> bytes:
        chunk = event.get("data", {}).get("chunk")
        if not chunk or not getattr(chunk, "content", None):
            return b""

        token = chunk.content
        if self._current_node in _REASONING_NODES:
            return _ds_reasoning(token)

        # Synthesis (or any non-reasoning node) → text frame.
        self._synthesis_streamed = True
        return _ds_text(token)
