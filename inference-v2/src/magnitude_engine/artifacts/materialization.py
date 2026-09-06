"""Partition tensor ownership among peer components before any allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from magnitude_engine.resources.budget import MemoryBudget, Reservation
from magnitude_engine.resources.io.reader import PositionalReader, Read

from .layouts import LogicalTensor


class MaterializedResource(Protocol):
    def close(self) -> None: ...


class TensorMaterializer(Protocol):
    """A dense store, expert operation or row lookup receives only its own tensor partition."""

    def materialize(self, tensors: dict[str, LogicalTensor]) -> MaterializedResource: ...


@dataclass(frozen=True)
class TensorPartition:
    owner: str
    names: frozenset[str]
    materializer: TensorMaterializer


class ModelAllocation:
    def __init__(self, resources: dict[str, MaterializedResource]):
        self.resources = resources
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        failures = []
        for resource in reversed(tuple(self.resources.values())):
            try:
                resource.close()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise BaseExceptionGroup("model resource retirement failed", failures)
        self.resources.clear()
        self.closed = True


def materialize(
    tensors: dict[str, LogicalTensor], partitions: tuple[TensorPartition, ...]
) -> ModelAllocation:
    assigned: set[str] = set()
    owners: set[str] = set()
    for partition in partitions:
        if not partition.owner or partition.owner in owners or assigned & partition.names:
            raise ValueError("model tensors and allocation owners must have unique ownership")
        owners.add(partition.owner)
        assigned.update(partition.names)
    if assigned != set(tensors):
        raise ValueError("tensor partitions must exactly cover the artifact")
    allocation = ModelAllocation({})
    try:
        for partition in partitions:
            allocation.resources[partition.owner] = partition.materializer.materialize(
                {name: tensors[name] for name in sorted(partition.names)}
            )
    except BaseException as error:
        try:
            allocation.close()
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "model construction and cleanup failed", [error, cleanup]
            ) from error
        raise
    return allocation


class ResidentTensors:
    def __init__(self, arrays: dict[str, mx.array], reservation: Reservation):
        self.arrays, self.reservation = arrays, reservation

    def close(self) -> None:
        self.arrays.clear()
        self.reservation.close()


class ResidentMaterializer:
    """Direct reads into final MLX allocations, without an entire-model staging copy."""

    def __init__(self, budget: MemoryBudget, reader: PositionalReader, *, owner: str):
        self.budget, self.reader, self.owner = budget, reader, owner

    def materialize(self, tensors: dict[str, LogicalTensor]) -> ResidentTensors:
        dtypes = {
            "BOOL": mx.bool_,
            "U8": mx.uint8,
            "I8": mx.int8,
            "U16": mx.uint16,
            "I16": mx.int16,
            "BF16": mx.bfloat16,
            "F16": mx.float16,
            "U32": mx.uint32,
            "I32": mx.int32,
            "F32": mx.float32,
            "U64": mx.uint64,
            "I64": mx.int64,
        }
        if any(tensor.dtype not in dtypes for tensor in tensors.values()):
            raise ValueError("tensor dtype is not supported by MLX")
        reservation = self.budget.reserve(self.owner, sum(t.nbytes for t in tensors.values()))
        arrays: dict[str, mx.array] = {}
        try:
            arrays = {name: mx.zeros(t.shape, dtype=dtypes[t.dtype]) for name, t in tensors.items()}
            mx.eval(*arrays.values())
            reads = []
            for name, tensor in tensors.items():
                destination = memoryview(arrays[name]).cast("B")
                offset = 0
                for piece in tensor.pieces:
                    reads.append(Read(piece, destination[offset : offset + piece.size]))
                    offset += piece.size
            batch = self.reader.submit(tuple(reads))
            try:
                batch.result()
            finally:
                batch.close()
        except BaseException:
            arrays.clear()
            reservation.close()
            raise
        return ResidentTensors(arrays, reservation)
