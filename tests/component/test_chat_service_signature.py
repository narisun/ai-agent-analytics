"""Lock the new ChatService constructor signature."""
import inspect

from src.services.chat_service import ChatService


def test_constructor_takes_named_deps():
    sig = inspect.signature(ChatService.__init__)
    names = {p.name for p in sig.parameters.values() if p.name != "self"}
    expected = {
        "graph",
        "conversation_store",
        "config",
        "user_ctx",
        "encoder_factory",
        "telemetry",
    }
    assert expected <= names, f"Missing ctor params: {expected - names}"


def test_execute_takes_chat_request():
    sig = inspect.signature(ChatService.execute)
    params = [p.name for p in sig.parameters.values() if p.name != "self"]
    assert params == ["req"], f"Expected execute(self, req), got {params}"
