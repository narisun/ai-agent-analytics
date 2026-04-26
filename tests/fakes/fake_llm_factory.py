"""FakeLLMFactory — returns FakeLLM instances for router and synthesis."""
from __future__ import annotations

from .fake_llm import FakeLLM


class FakeLLMFactory:
    def __init__(
        self,
        router_llm: FakeLLM | None = None,
        synthesis_llm: FakeLLM | None = None,
    ):
        self._router = router_llm or FakeLLM()
        self._synthesis = synthesis_llm or FakeLLM()

    def make_router_llm(self) -> FakeLLM:
        return self._router

    def make_synthesis_llm(self) -> FakeLLM:
        return self._synthesis
