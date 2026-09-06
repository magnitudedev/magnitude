from typing import Literal

from benchmarks.contracts import Subject, SubjectBlueprint
from magnitude_engine.composition import Blueprint, component
from magnitude_engine.engine.contracts import EngineInstance


@component
class ModelPrefill(SubjectBlueprint):
    engine: Blueprint[EngineInstance]
    prefix_tokens: int
    input_tokens: int
    rows: int = 1
    execution: Literal["shared", "independent"] = "shared"

    @staticmethod
    def implementation() -> type[Subject]:
        from .runtime import PrefillTrace

        return PrefillTrace
