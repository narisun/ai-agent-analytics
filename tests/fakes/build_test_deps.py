"""build_test_dependencies — one-knob factory for test composition.

Builds an AppDependencies populated with fakes. Override any piece by
passing the keyword.

NOTE: AppDependencies is imported from src.app_dependencies once Phase 4
Task 4.1 lands. Until then this module defines a placeholder dataclass
with the same shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

try:
    from src.app_dependencies import AppDependencies  # type: ignore
except ImportError:  # pre-Phase-4
    @dataclass
    class AppDependencies:  # type: ignore[no-redef]
        config: Any = None
        graph: Any = None
        conversation_store: Any = None
        mcp_tools_provider: Any = None
        llm_factory: Any = None
        telemetry: Any = None
        compaction: Any = None
        encoder_factory: Callable[[], Any] | None = None
        chat_service_factory: Callable[..., Any] | None = None

from .fake_compaction import FakeCompactionModifier
from .fake_conversation_store import FakeConversationStore
from .fake_llm_factory import FakeLLMFactory
from .fake_mcp_tools_provider import FakeMCPToolsProvider
from .fake_stream_encoder import FakeStreamEncoder
from .fake_telemetry import FakeTelemetryScope


def build_test_dependencies(
    *,
    config: Any | None = None,
    graph: Any | None = None,
    conversation_store: Any | None = None,
    mcp_tools_provider: Any | None = None,
    llm_factory: Any | None = None,
    telemetry: Any | None = None,
    compaction: Any | None = None,
    encoder_factory: Callable[[], Any] | None = None,
    chat_service_factory: Callable[..., Any] | None = None,
) -> AppDependencies:
    """Build AppDependencies with fakes; override per keyword."""
    return AppDependencies(
        config=config,
        graph=graph,
        conversation_store=conversation_store or FakeConversationStore(),
        mcp_tools_provider=mcp_tools_provider or FakeMCPToolsProvider(),
        llm_factory=llm_factory or FakeLLMFactory(),
        telemetry=telemetry or FakeTelemetryScope(),
        compaction=compaction or FakeCompactionModifier(),
        encoder_factory=encoder_factory or (lambda: FakeStreamEncoder()),
        chat_service_factory=chat_service_factory,
    )
