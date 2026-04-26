"""FakeStreamEncoder — records calls, returns deterministic bytes."""
from __future__ import annotations

import json
from typing import Any


class FakeStreamEncoder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.errors: list[tuple[str, str]] = []  # (error_id, str(err))
        self.finalized: bool = False

    def encode_event(self, event: dict[str, Any]) -> bytes:
        self.events.append(event)
        return json.dumps({"event": event}).encode() + b"\n"

    def encode_error(self, err: Exception, *, error_id: str) -> bytes:
        self.errors.append((error_id, str(err)))
        return json.dumps({"error": str(err), "error_id": error_id}).encode() + b"\n"

    def finalize(self) -> bytes:
        self.finalized = True
        return b""
