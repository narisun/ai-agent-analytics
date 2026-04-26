"""AppDependencies — the single wiring dataclass for analytics-agent.

Built by lifespan() at startup from the environment; passed to create_app()
as its only argument. Tests build it via tests.fakes.build_test_dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from platform_sdk import AgentConfig

from .domain.types import UserContext
from .ports import (
    CompactionModifier,
    ConversationStore,
    LLMFactory,
    MCPToolsProvider,
    StreamEncoder,
    TelemetryScope,
)


@dataclass
class AppDependencies:
    """Fully-wired singletons + per-request factories."""

    config: AgentConfig | None
    graph: Any  # CompiledGraph from langgraph; left as Any to avoid heavy import
    conversation_store: ConversationStore | None
    mcp_tools_provider: MCPToolsProvider | None
    llm_factory: LLMFactory | None
    telemetry: TelemetryScope | None
    compaction: CompactionModifier | None
    encoder_factory: Callable[[], StreamEncoder] | None
    chat_service_factory: Callable[[UserContext], Any] | None
