from typing import Literal

from benchmarks.contracts import Subject, SubjectBlueprint
from magnitude_engine.composition import Blueprint, component
from magnitude_engine.engine.contracts import EngineInstance


@component
class GenerationDecode(SubjectBlueprint):
    engine: Blueprint[EngineInstance]
    prompt_tokens: int
    output_tokens: int
    rows: int
    token_allowance: int = 4
    execution: Literal['shared', 'independent'] = 'shared'

    @staticmethod
    def implementation() -> type[Subject]:
        from .runtime import GenerationTrace

        return GenerationTrace
