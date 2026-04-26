"""build_analytics_graph(deps) takes a single GraphDependencies argument."""
import inspect

from src.graph import build_analytics_graph


def test_signature_first_param_is_deps():
    sig = inspect.signature(build_analytics_graph)
    params = list(sig.parameters)
    assert params[0] == "deps", f"Expected first param 'deps', got {params}"


def test_legacy_kwargs_still_accepted():
    # bridges, config, prompts, checkpointer must remain as keyword-only
    # parameters so the existing app.py call site keeps working.
    sig = inspect.signature(build_analytics_graph)
    names = set(sig.parameters)
    expected = {"deps", "bridges", "config", "prompts", "checkpointer"}
    assert expected <= names, f"Missing legacy kwargs: {expected - names}"
