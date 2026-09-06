from benchmarks.contracts import Subject, SubjectBlueprint
from magnitude_engine.composition import Blueprint, component
from magnitude_engine.engine.contracts import EngineInstance


@component
class EngineWaves(SubjectBlueprint):
    engine: Blueprint[EngineInstance]
    prompt_text: str
    prompt_tokens: int
    output_tokens: int
    rows: int
    prefix_reuse: bool = True

    @staticmethod
    def implementation() -> type[Subject]:
        from .runtime import EngineTrace

        return EngineTrace
