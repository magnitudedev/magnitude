"""Final MLX bank allocations and direct expert-record transfers."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from magnitude_engine.artifacts.layouts import LogicalTensor
from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.io.reader import PositionalReader, Read

from .computation import ExpertWeights, QuantizedProjection


@dataclass(frozen=True)
class ProjectionSource:
    weight: LogicalTensor
    scales: LogicalTensor
    biases: LogicalTensor

    @property
    def components(self) -> tuple[LogicalTensor, ...]:
        return self.weight, self.scales, self.biases


@dataclass(frozen=True)
class ExpertSource:
    up: ProjectionSource
    gate: ProjectionSource
    down: ProjectionSource
    encoding: AffineEncoding

    def __post_init__(self) -> None:
        for projection in self.projections:
            weight, scales, biases = projection.components
            if (
                len(weight.shape) != 3
                or weight.shape[0] <= 0
                or weight.shape[0] != self.experts
                or scales.shape != biases.shape
                or scales.shape[:2] != weight.shape[:2]
                or len(scales.shape) != 3
                or weight.dtype != "U32"
                or scales.dtype not in ("BF16", "F16", "F32")
                or biases.dtype != scales.dtype
                or weight.shape[-1] * 32 // self.encoding.bits
                != scales.shape[-1] * self.encoding.group_size
            ):
                raise ValueError("expert projection geometry or affine encoding is inconsistent")
        if (
            self.up.weight.shape != self.gate.weight.shape
            or self.down.weight.shape[-1] * 32 // self.encoding.bits != self.up.weight.shape[1]
            or self.down.weight.shape[1] != self.up.weight.shape[-1] * 32 // self.encoding.bits
        ):
            raise ValueError("expert gate/up/down dimensions do not compose")

    @property
    def experts(self) -> int:
        return self.up.weight.shape[0]

    @property
    def projections(self) -> tuple[ProjectionSource, ...]:
        return self.up, self.gate, self.down

    @property
    def expert_bytes(self) -> int:
        return (
            sum(
                tensor.nbytes for projection in self.projections for tensor in projection.components
            )
            // self.experts
        )


class BankLease:
    def __init__(self, bank: ExpertBank):
        self.bank, self.closed = bank, False

    def close(self) -> None:
        if not self.closed:
            self.bank._leased = False
            self.closed = True


class ExpertBank:
    """One exclusive mutable bank; consumers retire through the execution scope."""

    def __init__(self, source: ExpertSource, slots: int, budget: MemoryBudget, *, owner: str):
        if not 0 < slots <= source.experts:
            raise ValueError("bank capacity exceeds logical experts")
        self.slots = slots
        self._leased = False
        self._closed = False
        self._encoding = source.encoding
        self._geometry = tuple(
            (t.dtype, t.shape[1:]) for p in source.projections for t in p.components
        )
        self._reservation = budget.reserve(owner, slots * source.expert_bytes)
        dtypes = {"U32": mx.uint32, "BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}
        self.arrays: tuple[mx.array, ...] = ()
        try:
            self.arrays = tuple(
                mx.zeros((slots, *shape), dtype=dtypes[dtype]) for dtype, shape in self._geometry
            )
            mx.eval(*self.arrays)
        except BaseException:
            self.arrays = ()
            self._reservation.close()
            raise
        projections = [
            QuantizedProjection(
                self.arrays[index], self.arrays[index + 1], self.arrays[index + 2], source.encoding
            )
            for index in range(0, 9, 3)
        ]
        self._weights: ExpertWeights | None = ExpertWeights(*projections)
        self.metrics = {"experts_read": 0, "bytes_read": 0, "read_calls": 0}

    @property
    def weights(self) -> ExpertWeights:
        if self._weights is None:
            raise RuntimeError("expert bank is closed")
        return self._weights

    def acquire(self) -> BankLease:
        if self._closed or self._leased:
            raise RuntimeError("expert bank is closed or still leased by an earlier execution")
        self._leased = True
        return BankLease(self)

    def fill(
        self,
        source: ExpertSource,
        destinations: tuple[tuple[int, int], ...],
        reader: PositionalReader,
    ) -> None:
        if self._closed or not self._leased:
            raise RuntimeError("expert transport requires an exclusive bank lease")
        geometry = tuple((t.dtype, t.shape[1:]) for p in source.projections for t in p.components)
        if geometry != self._geometry or source.encoding != self._encoding:
            raise ValueError("shared expert scratch requires identical component geometry")
        if len({slot for _, slot in destinations}) != len(destinations) or any(
            not 0 <= slot < self.slots for _, slot in destinations
        ):
            raise ValueError("expert transfers require distinct in-range destination slots")
        reads = []
        tensors = [t for projection in source.projections for t in projection.components]
        for tensor, array in zip(tensors, self.arrays, strict=True):
            size = tensor.nbytes // source.experts
            buffer = memoryview(array).cast("B")
            for expert, slot in destinations:
                offset = slot * size
                for piece in tensor.row(expert):
                    reads.append(Read(piece, buffer[offset : offset + piece.size]))
                    offset += piece.size
        batch = reader.submit(tuple(reads))
        try:
            metrics = batch.result()
        finally:
            batch.close()
        expected = len(destinations) * source.expert_bytes
        if metrics.bytes != expected:
            raise RuntimeError("expert read did not fill every component")
        self.metrics["experts_read"] += len(destinations)
        self.metrics["bytes_read"] += metrics.bytes
        self.metrics["read_calls"] += metrics.calls

    def close(self) -> None:
        if self._closed:
            return
        if self._leased:
            raise RuntimeError("retire expert consumers before closing their bank")
        self.arrays = ()
        self._weights = None
        self._reservation.close()
        self._closed = True
