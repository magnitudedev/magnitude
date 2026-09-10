"""Position-addressed selection, independent of model and generation state.

The caller supplies the prepared distribution logits. Temperature, penalties,
constraints and truncation belong to distribution construction; this component
selects from that distribution and reports a separate result for each row.
"""

import math
import struct
from enum import IntEnum
from typing import NewType

from pydantic import Field

from magnitude_engine.data import Record, TokenId
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import (
    DeviceContext,
    DType,
    Executable,
    Prepared,
    Tensor,
    TensorSpec,
    Ticket,
)

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

    def words(self) -> tuple[int, ...]:
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


class SampleSelector:
    def __init__(self, context: DeviceContext, tile: int = 1024):
        if type(tile) is not int or tile <= 0:
            raise ValueError("selection tile must be a positive integer")
        self.context, self.tile = context, tile
        self._plans: dict[tuple[int, int], tuple[Executable, Executable]] = {}

    def prepare(
        self, logits: Tensor, draws: tuple[Draw, ...], output: Tensor
    ) -> tuple[Prepared, ...]:
        if len(logits.spec.shape) != 2 or logits.spec.dtype != DType.F32:
            raise ValueError("selection requires FP32 distribution rows")
        rows, vocabulary = logits.spec.shape
        if len(draws) != rows or output.spec != TensorSpec((rows, 2), DType.I32):
            raise ValueError("draws and output must match distribution rows")
        key = rows, vocabulary
        tiles = math.ceil(vocabulary / self.tile)
        if key not in self._plans:
            from magnitude_engine.numerics.sampling import finish_selection, select_tiles

            cpu = self.context.backend == Backend.LLVM
            first = self.context.specialize(select_tiles, rows, vocabulary, cpu=cpu, tile=self.tile)
            second = self.context.specialize(finish_selection, rows, tiles, cpu=cpu)
            self._plans[key] = first, second
        with Preparation(self.context) as p:
            words = tuple(word for draw in draws for word in draw.words())
            addresses = p.upload(
                TensorSpec((rows, 6), DType.U32), struct.pack(f"={len(words)}I", *words)
            )
            values = p.allocate(TensorSpec((rows, tiles), DType.F32))
            indices = p.allocate(TensorSpec((rows, tiles), DType.I32))
            invalid = p.allocate(indices.spec)
            first, second = self._plans[key]
            p.add(Prepared(self.context, first, [logits, addresses, values, indices, invalid]))
            p.add(Prepared(self.context, second, [values, indices, invalid, output]))
            return p.finish()

    def read(self, output: Tensor, *, after: Ticket) -> tuple[SampleResult, ...]:
        if (
            len(output.spec.shape) != 2
            or output.spec.shape[1] != 2
            or output.spec.dtype != DType.I32
        ):
            raise ValueError("sample results require token/status rows")
        raw = self.context.read(output, after=after)
        return tuple(
            SampledToken(token=TokenId(token))
            if status == 0
            else UnselectableDistribution(reason=SelectionFailure(status))
            for token, status in struct.iter_unpack("=ii", raw)
        )

    def close(self) -> None:
        self._plans.clear()
