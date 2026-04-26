"""FakeTelemetryScope — records spans and events."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


class _FakeSpan:
    def __init__(self) -> None:
        self.exceptions: list[Exception] = []
        self.status: str | None = None

    def record_exception(self, err: Exception) -> None:
        self.exceptions.append(err)

    def set_status(self, status: str) -> None:
        self.status = status


class FakeTelemetryScope:
    def __init__(self) -> None:
        self.spans: list[str] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.last_span = _FakeSpan()

    @contextmanager
    def start_span(self, name: str) -> Iterator[_FakeSpan]:
        self.spans.append(name)
        self.last_span = _FakeSpan()
        yield self.last_span

    def record_event(self, name: str, **attrs: Any) -> None:
        self.events.append((name, attrs))
