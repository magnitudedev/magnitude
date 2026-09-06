from typing import Literal

from benchmarks.contracts import Subject, SubjectBlueprint
from magnitude_engine.composition import Blueprint, component
from magnitude_engine.models.recurrence.contracts import DeltaRecurrence


@component
class Recurrence(SubjectBlueprint):
    update: Blueprint[DeltaRecurrence]
    tokens: int
    key_heads: int = 16
    value_heads: int = 32
    key_width: int = 128
    value_width: int = 128
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    @staticmethod
    def implementation() -> type[Subject]:
        from .runtime import DeltaTrace

        return DeltaTrace


@component
class RecurrenceShapes(SubjectBlueprint):
    update: Blueprint[DeltaRecurrence]
    first_tokens: int = 33
    shapes: int = 16

    def validate_measurement(self, warmup: int, repetitions: int) -> None:
        if warmup != 0 or repetitions != 1:
            raise ValueError("cold shape sweep requires one sample and no warmup")

    @staticmethod
    def implementation() -> type[Subject]:
        from .runtime import DeltaShapeTrace

        return DeltaShapeTrace
