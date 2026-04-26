"""GraphDependencies — dependencies needed to build the analytics graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from platform_sdk import AgentConfig

from .ports import CompactionModifier, LLMFactory, MCPToolsProvider


@dataclass
class GraphDependencies:
    llm_factory: LLMFactory
    tools_provider: MCPToolsProvider
    compaction: CompactionModifier
    config: AgentConfig
    prompts: Any = None  # PromptLoader | None
