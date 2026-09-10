"""Materialize run visibility once per invocation, shared by every model layer.

State owns the physical extents. These operands only expose their numerical read
and write windows, together with device metadata. No state ownership is cached.
"""

from dataclasses import dataclass

from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.platform.execution import Tensor
from magnitude_engine.state.kv import KVReadGroup, KVReadRun, KVWrite, grouped_reads


@dataclass(frozen=True)
class ReadGeometry:
    """Stable buffer extent and a bounded specialization range; metadata is visibility."""

    capacity: int
    segment_capacity: int
    segments: int


@dataclass(frozen=True)
class ReadBinding:
    geometry: ReadGeometry
    keys: Tensor
    values: Tensor
    metadata: Tensor

    @property
    def operands(self) -> tuple[Tensor, ...]:
        return self.keys, self.values, self.metadata


@dataclass(frozen=True)
class WriteGeometry:
    capacity: int
    source: int
    length: int


@dataclass(frozen=True)
class WriteBinding:
    geometry: WriteGeometry
    keys: Tensor
    values: Tensor
    offset: Tensor

    @property
    def operands(self) -> tuple[Tensor, ...]:
        return self.keys, self.values, self.offset


class KVBinding:
    def __init__(
        self, preparation: Preparation, history: tuple[KVReadRun, ...], writes: tuple[KVWrite, ...]
    ):
        self.preparation = preparation
        self.groups: tuple[KVReadGroup, ...] = grouped_reads(history)
        self.writes = writes
        self.leases = tuple(preparation.own(group.acquire()) for group in self.groups)
        self.metadata = tuple(
            preparation.indices(
                tuple(value for segment in group.segments for value in segment),
                (len(group.runs), 3),
            )
            for group in self.groups
        )
        self.offsets = tuple(preparation.indices((write.destination,)) for write in writes)

    def reads(self, layer: int) -> tuple[ReadBinding, ...]:
        results = []
        for group, lease, metadata in zip(self.groups, self.leases, self.metadata, strict=True):
            keys, values = lease.views(layer)
            self.preparation.own(keys)
            self.preparation.own(values)
            results.append(
                ReadBinding(
                    ReadGeometry(
                        group.capacity,
                        min(group.capacity, 1 << (group.segment_capacity - 1).bit_length()),
                        len(group.runs),
                    ),
                    keys,
                    values,
                    metadata,
                )
            )
        return tuple(results)

    def appends(self, layer: int) -> tuple[WriteBinding, ...]:
        results = []
        for write, offset in zip(self.writes, self.offsets, strict=True):
            keys, values = write.run.views(layer)
            self.preparation.own(keys)
            self.preparation.own(values)
            results.append(
                WriteBinding(
                    WriteGeometry(write.run.capacity, write.source, write.length),
                    keys,
                    values,
                    offset,
                )
            )
        return tuple(results)
