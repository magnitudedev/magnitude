"""Logical sampling requests and host-visible result meanings."""

from enum import IntEnum
from typing import NewType

from pydantic import Field

from engine.data import Record, TokenId

SamplingSeed = NewType("SamplingSeed", int)
SamplePosition = NewType("SamplePosition", int)


class SelectionKind(IntEnum):
    GREEDY = 0
    CATEGORICAL = 1


class DrawDomain(IntEnum):
    TARGET = 0
    DRAFT = 1
    ACCEPTANCE = 2
    RESIDUAL = 3


class Draw(Record):
    kind: SelectionKind
    seed: SamplingSeed = Field(ge=0, lt=2**64)
    position: SamplePosition = Field(ge=0, lt=2**64)
    domain: DrawDomain = DrawDomain.TARGET

    def words(self) -> tuple[int, int, int, int, int, int]:
        return (
            int(self.kind),
            self.seed & 0xFFFFFFFF,
            self.seed >> 32,
            self.position & 0xFFFFFFFF,
            self.position >> 32,
            int(self.domain),
        )


class SelectionFailure(IntEnum):
    EMPTY = 1
    NONFINITE = 2


class SampledToken(Record):
    token: TokenId = Field(ge=0, le=0x7FFFFFFF)


class UnselectableDistribution(Record):
    reason: SelectionFailure


type SampleResult = SampledToken | UnselectableDistribution
