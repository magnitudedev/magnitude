from benchmarks.contracts import Subject, SubjectBlueprint
from magnitude_engine.composition import Blueprint, component
from magnitude_engine.engine.contracts import EngineInstance


@component
class PlainDecode(SubjectBlueprint):
    engine: Blueprint[EngineInstance]
    prompt_tokens: int
    output_tokens: int
    token_allowance: int = 1

    @staticmethod
    def implementation() -> type[Subject]:
        from .runtime import DecodeTrace

        return DecodeTrace
