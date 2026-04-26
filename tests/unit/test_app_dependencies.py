"""AppDependencies dataclass holds wired singletons + factories."""
from dataclasses import fields

from src.app_dependencies import AppDependencies


def test_has_required_fields():
    names = {f.name for f in fields(AppDependencies)}
    expected = {
        "config",
        "graph",
        "conversation_store",
        "mcp_tools_provider",
        "llm_factory",
        "telemetry",
        "compaction",
        "encoder_factory",
        "chat_service_factory",
    }
    assert expected <= names, f"Missing: {expected - names}"


def test_instantiation_with_nones_is_allowed_for_tests():
    deps = AppDependencies(
        config=None,
        graph=None,
        conversation_store=None,
        mcp_tools_provider=None,
        llm_factory=None,
        telemetry=None,
        compaction=None,
        encoder_factory=None,
        chat_service_factory=None,
    )
    assert deps is not None
