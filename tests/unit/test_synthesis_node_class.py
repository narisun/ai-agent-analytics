"""SynthesisNode is a callable class with constructor injection."""
import inspect

from src.nodes.synthesis import SynthesisNode
from tests.fakes.fake_compaction import FakeCompactionModifier
from tests.fakes.fake_llm import FakeLLM


def test_constructor_takes_named_deps():
    sig = inspect.signature(SynthesisNode.__init__)
    names = {p.name for p in sig.parameters.values() if p.name != "self"}
    assert {"llm", "prompts", "compaction", "chart_max_data_points"} <= names


def test_is_async_callable():
    node = SynthesisNode(
        llm=FakeLLM(),
        prompts=None,
        compaction=FakeCompactionModifier(),
        chart_max_data_points=20,
    )
    assert callable(node)
    assert inspect.iscoroutinefunction(node.__call__)
