from typing import Literal

from benchmarks.contracts import Subject, SubjectBlueprint
from magnitude_engine.composition import Blueprint, component
from magnitude_engine.engine.contracts import EngineInstance


@component
class ContextWorkload(SubjectBlueprint):
    engine: Blueprint[EngineInstance]
    artifact: str
    fixture: Literal["prose.moby-dick", "tools.bfcl"]
    context_tokens: int
    mode: Literal["prefill", "replay", "generate"]
    measured_tokens: int = 256
    offset: int = 0

    @staticmethod
    def implementation() -> type[Subject]:
        from .runtime import ContextTrace

        return ContextTrace
