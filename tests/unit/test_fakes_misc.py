"""Unit tests for FakeStreamEncoder, FakeTelemetryScope, FakeCompactionModifier."""

from src.ports import (
    CompactionModifier,
    StreamEncoder,
    TelemetryScope,
)
from tests.fakes.fake_compaction import FakeCompactionModifier
from tests.fakes.fake_stream_encoder import FakeStreamEncoder
from tests.fakes.fake_telemetry import FakeTelemetryScope


def test_stream_encoder_satisfies_protocol():
    assert isinstance(FakeStreamEncoder(), StreamEncoder)


def test_telemetry_satisfies_protocol():
    assert isinstance(FakeTelemetryScope(), TelemetryScope)


def test_compaction_satisfies_protocol():
    assert isinstance(FakeCompactionModifier(), CompactionModifier)


def test_encoder_records_events():
    enc = FakeStreamEncoder()
    enc.encode_event({"type": "x"})
    enc.encode_event({"type": "y"})
    enc.encode_error(ValueError("bad"), error_id="e1")
    enc.finalize()
    assert len(enc.events) == 2
    assert len(enc.errors) == 1
    assert enc.finalized is True


def test_telemetry_records_spans_and_events():
    t = FakeTelemetryScope()
    with t.start_span("step1"):
        t.record_event("work", x=1)
    assert t.spans == ["step1"]
    assert t.events == [("work", {"x": 1})]


def test_compaction_passthrough():
    c = FakeCompactionModifier()
    msgs = ["a", "b", "c"]
    assert c.apply(msgs) == msgs
