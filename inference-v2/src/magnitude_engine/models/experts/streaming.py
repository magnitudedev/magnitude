"""Demand-streamed expert operation with capped decode and one-layer prefill scratch."""

from typing import cast

import mlx.core as mx

from magnitude_engine.components import component
from magnitude_engine.resources.io.reader import PositionalReader

from ..execution import ExecutionScope
from .bank import ExpertBank, ExpertSource
from .computation import GatedExpertMath
from .residency import Residency, Transfer


@component("MODEL:EXPERTS:MAG:STREAMED")
class StreamedExperts:
    def __init__(
        self,
        source: ExpertSource,
        math: GatedExpertMath,
        *,
        bank: ExpertBank,
        scratch: ExpertBank,
        reader: PositionalReader,
    ):
        if scratch.slots != source.experts:
            raise ValueError("exact prefill requires one scratch slot per logical expert")
        self.source, self.math = source, math
        self.bank, self.scratch, self.reader = bank, scratch, reader
        self.residency = Residency(source.experts, bank.slots)

    def compute(self, hidden: mx.array, assignments: mx.array, scope: ExecutionScope) -> mx.array:
        # MLX's tolist annotation cannot express the known rank-one shape.
        logical = cast(list[int], assignments.reshape(-1).tolist())
        if any(
            type(expert) is not int or not 0 <= expert < self.source.experts for expert in logical
        ):
            raise ValueError("authoritative expert routes are out of bounds")
        unique = tuple(dict.fromkeys(logical))
        if assignments.size > self.bank.slots:
            # Full logical expert axis preserves the reference QMM dispatch.
            # The same scratch is reused by the next layer only after consumers complete.
            lease = scope.acquire(self.scratch.acquire)
            self.scratch.fill(
                self.source, tuple((expert, expert) for expert in unique), self.reader
            )
            output = self.math.apply(
                self.scratch.weights, hidden, assignments, expand_assignments=True
            )
            scope.depend(output)
            scope.retire(lease, output)
            return output
        scope.acquire(self.bank.acquire)
        existing = tuple(expert for expert in unique if self.residency.resolve(expert) is not None)
        protection = self.residency.pin(existing)
        transfers: list[Transfer] = []
        try:
            for expert in unique:
                transfer = self.residency.reserve(expert)
                if transfer is not None:
                    transfers.append(transfer)
            self.bank.fill(self.source, tuple((t.expert, t.slot) for t in transfers), self.reader)
        except BaseException:
            for transfer in transfers:
                self.residency.finish(transfer, publish=False)
            raise
        finally:
            protection.close()
        for transfer in transfers:
            self.residency.finish(transfer, publish=True)
        self.residency.touch(unique)
        physical = mx.array([self.residency.resolve(expert) for expert in logical], dtype=mx.int32)
        output = self.math.apply(self.bank.weights, hidden, physical.reshape(assignments.shape))
        scope.depend(output)
        return output
