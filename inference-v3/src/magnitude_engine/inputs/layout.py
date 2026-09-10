"""Logical input boundaries and language history, independent of numerical state."""

from enum import StrEnum
from typing import NewType

from pydantic import Field, model_validator

from magnitude_engine.data import Record

InputPosition = NewType("InputPosition", int)
ConditioningIdentity = NewType("ConditioningIdentity", str)


class BoundaryRule(StrEnum):
    CAUSAL = "causal"
    INDIVISIBLE = "indivisible"


class InputSpan(Record):
    start: InputPosition = Field(ge=0)
    end: InputPosition = Field(gt=0)
    identity: ConditioningIdentity = Field(min_length=1)
    boundaries: BoundaryRule = BoundaryRule.CAUSAL
    language_history: bool = False

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("input span must be nonempty")
        return self


class InputLayout(Record):
    """Unmarked positions are ordinary language; spans describe conditioned input.

    Logical positions may continue beyond the prompt. A soft chunk allowance is
    enlarged only when needed to complete the first indivisible semantic unit.
    """

    count: int = Field(ge=0, le=0x7FFFFFFF)
    spans: tuple[InputSpan, ...] = ()

    @model_validator(mode="after")
    def ordered(self):
        previous = 0
        for span in self.spans:
            if span.start < previous or span.end > self.count:
                raise ValueError("input spans must be ordered, disjoint and inside the prompt")
            previous = span.end
        return self

    def boundary(self, position: int) -> bool:
        return (
            type(position) is int
            and 0 <= position <= 0x7FFFFFFF
            and not any(
                span.boundaries == BoundaryRule.INDIVISIBLE and span.start < position < span.end
                for span in self.spans
            )
        )

    def chunk_end(self, start: int, available_end: int, allowance: int) -> InputPosition:
        if not self.boundary(start) or not self.boundary(available_end):
            raise ValueError("chunk range must start and finish at legal input boundaries")
        if type(allowance) is not int or allowance <= 0 or available_end <= start:
            raise ValueError("chunk requires available input and a positive allowance")
        end = min(start + allowance, available_end)
        for span in self.spans:
            if span.boundaries == BoundaryRule.INDIVISIBLE and span.start < end < span.end:
                end = span.start if span.start > start else span.end
                break
        return InputPosition(end)

    def language(self, position: int) -> bool:
        if type(position) is not int or not 0 <= position <= 0x7FFFFFFF:
            raise ValueError("language history requires a valid logical input position")
        return all(
            span.language_history for span in self.spans if span.start <= position < span.end
        )
