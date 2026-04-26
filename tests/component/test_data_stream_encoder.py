"""Component tests for DataStreamEncoder.

Verifies the wire-format port preserves the byte-level contract of the
pre-Phase-7 inline data_stream_generator in src/app.py.
"""
import json

import pytest

from src.ports import StreamEncoder
from src.streaming.data_stream_encoder import DataStreamEncoder


def test_satisfies_stream_encoder_protocol():
    assert isinstance(DataStreamEncoder(), StreamEncoder)


def test_finalize_emits_finish_frame():
    enc = DataStreamEncoder()
    out = enc.finalize()
    assert out.startswith(b"d:")
    payload = json.loads(out[2:].decode().strip())
    assert payload["finishReason"] == "stop"


def test_encode_error_emits_error_frame():
    enc = DataStreamEncoder()
    out = enc.encode_error(ValueError("boom"), error_id="e1")
    assert out.startswith(b"3:")
    body = json.loads(out[2:].decode().strip())
    assert "boom" in body


def test_on_chain_start_intent_router_emits_reasoning():
    enc = DataStreamEncoder()
    out = enc.encode_event({"event": "on_chain_start", "name": "intent_router"})
    assert out.startswith(b"g:")
    text = json.loads(out[2:].decode().strip())
    assert "Classifying intent" in text


def test_on_chain_start_synthesis_emits_reasoning():
    enc = DataStreamEncoder()
    out = enc.encode_event({"event": "on_chain_start", "name": "synthesis"})
    assert out.startswith(b"g:")
    text = json.loads(out[2:].decode().strip())
    assert "Generating analysis" in text


def test_on_chain_end_intent_router_emits_intent_and_tool_calls():
    enc = DataStreamEncoder()
    event = {
        "event": "on_chain_end",
        "name": "intent_router",
        "data": {
            "output": {
                "intent": "data_query",
                "intent_reasoning": "user wants data",
                "query_plan": [
                    {
                        "tool_name": "execute_read_query",
                        "mcp_server": "data-mcp",
                        "parameters": {"query": "SELECT 1"},
                    }
                ],
            }
        },
    }
    out = enc.encode_event(event)
    text = out.decode()
    # The intent reasoning line must appear.
    assert "Intent: data_query" in text
    # And exactly one tool_call_begin frame.
    assert "b:" in text
    assert "data-mcp/execute_read_query" in text
    # SQL query is also surfaced for execute_read_query.
    assert "__SQL_QUERY__:" in text


def test_on_chain_end_synthesis_emits_ui_components_and_narrative():
    enc = DataStreamEncoder()
    event = {
        "event": "on_chain_end",
        "name": "synthesis",
        "data": {
            "output": {
                "narrative": "the answer",
                "ui_components": [{"component_type": "BarChart", "data": []}],
            }
        },
    }
    out = enc.encode_event(event)
    text = out.decode()
    # Components appear as a 2: data frame.
    assert "2:" in text
    assert "BarChart" in text
    # Narrative emitted as text since no synthesis_streamed yet.
    assert "0:" in text


def test_on_chat_model_stream_routes_to_reasoning_in_reasoning_node():
    enc = DataStreamEncoder()
    # Open intent_router so encoder tracks it as a REASONING_NODE.
    enc.encode_event({"event": "on_chain_start", "name": "intent_router"})

    # Stream a token while intent_router is current.
    fake_chunk = type("C", (), {"content": "thinking"})()
    out = enc.encode_event({
        "event": "on_chat_model_stream",
        "data": {"chunk": fake_chunk},
    })
    assert out.startswith(b"g:"), f"Expected reasoning frame, got {out!r}"


def test_on_chat_model_stream_routes_to_text_in_synthesis():
    enc = DataStreamEncoder()
    enc.encode_event({"event": "on_chain_start", "name": "synthesis"})

    fake_chunk = type("C", (), {"content": "the "})()
    out = enc.encode_event({
        "event": "on_chat_model_stream",
        "data": {"chunk": fake_chunk},
    })
    # Synthesis tokens go to text (0:) and mark synthesis_streamed=True.
    assert out.startswith(b"0:")


def test_on_chain_end_synthesis_skips_narrative_after_streaming():
    enc = DataStreamEncoder()
    # Synthesis starts; tokens stream via on_chat_model_stream.
    enc.encode_event({"event": "on_chain_start", "name": "synthesis"})
    fake_chunk = type("C", (), {"content": "the answer"})()
    enc.encode_event({"event": "on_chat_model_stream", "data": {"chunk": fake_chunk}})

    # Synthesis ends; narrative should NOT be re-emitted as a 0: frame
    # (already streamed token by token).
    end_out = enc.encode_event({
        "event": "on_chain_end",
        "name": "synthesis",
        "data": {"output": {"narrative": "the answer", "ui_components": []}},
    })
    text = end_out.decode()
    # No 0: frame for the duplicated narrative.
    assert "0:" not in text
    # No components either since the list is empty.
    assert text == ""


def test_unknown_event_returns_empty_bytes():
    enc = DataStreamEncoder()
    out = enc.encode_event({"event": "on_some_unknown_event", "name": "X"})
    assert out == b""


def test_on_chain_end_error_handler_emits_narrative_or_errors():
    enc = DataStreamEncoder()
    event = {
        "event": "on_chain_end",
        "name": "error_handler",
        "data": {
            "output": {
                "narrative": "Something went wrong",
                "errors": [],
                "follow_up_suggestions": ["try again", "rephrase"],
            }
        },
    }
    out = enc.encode_event(event)
    text = out.decode()
    assert "0:" in text
    assert "Something went wrong" in text
    # Follow-ups emitted as data frame
    assert "2:" in text
    assert "follow_up_suggestions" in text
