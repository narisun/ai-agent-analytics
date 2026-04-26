"""GraphDependencies dataclass exists and has required fields."""
from dataclasses import fields

from src.graph_dependencies import GraphDependencies


def test_fields():
    names = {f.name for f in fields(GraphDependencies)}
    assert {"llm_factory", "tools_provider", "compaction", "config", "prompts"} <= names
