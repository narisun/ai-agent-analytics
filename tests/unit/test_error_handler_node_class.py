"""ErrorHandlerNode is a callable class — no deps."""
import inspect

from src.nodes.error_handler import ErrorHandlerNode


def test_no_args_constructor():
    sig = inspect.signature(ErrorHandlerNode.__init__)
    names = {p.name for p in sig.parameters.values() if p.name != "self"}
    assert names == set(), f"Expected empty deps, got {names}"


def test_is_async_callable():
    node = ErrorHandlerNode()
    assert callable(node)
    assert inspect.iscoroutinefunction(node.__call__)
