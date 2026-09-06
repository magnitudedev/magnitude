"""Assign model tensors to injected operation owners before materialization."""

from dataclasses import dataclass

import mlx.nn as nn

from magnitude_engine.artifacts.layouts import LogicalTensor
from magnitude_engine.artifacts.materialization import ModelAllocation
from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.io.reader import PositionalReader

from ..embeddings.contracts import EmbeddingFactory, EmbeddingLookup
from ..experts.contracts import ExpertFactory, ExpertOperator
from .parameters import materialize_parameters
from .partitions import (
    EmbeddingPartition,
    ExpertPartition,
    OperationResources,
)
from .validation import configure_affine_modules, validate_parameters


@dataclass
class BoundOperations:
    allocation: ModelAllocation
    resources: OperationResources
    embeddings: dict[str, EmbeddingLookup]
    experts: dict[int, ExpertOperator]

    def close(self) -> None:
        self.embeddings.clear()
        self.experts.clear()
        try:
            self.resources.close()
        finally:
            self.allocation.close()


def prepare_layout(
    model: nn.Module, tensors: dict[str, LogicalTensor], encodings: dict[str, AffineEncoding]
) -> None:
    configure_affine_modules(model, encodings)
    validate_parameters(model, {name: tensor.shape for name, tensor in tensors.items()})


def bind_operations(
    model: nn.Module,
    tensors: dict[str, LogicalTensor],
    *,
    embeddings: dict[str, tuple[EmbeddingPartition, EmbeddingFactory]],
    experts: dict[int, tuple[ExpertPartition, ExpertFactory]],
    budget: MemoryBudget,
    reader: PositionalReader,
) -> BoundOperations:
    excluded: frozenset[str] = frozenset()
    assigned: set[str] = set()
    assignments = [
        (partition.tensors, factory.excluded(partition))
        for partition, factory in embeddings.values()
    ] + [
        (partition.tensors, factory.excluded(partition)) for partition, factory in experts.values()
    ]
    for partition_tensors, removed in assignments:
        names = set(partition_tensors)
        if assigned & names or not names <= tensors.keys():
            raise ValueError("operation tensor ownership overlaps or exceeds the artifact")
        assigned.update(names)
        if not removed <= names:
            raise ValueError("an operation cannot exclude a peer's tensors")
        excluded |= removed
    allocation = materialize_parameters(
        model,
        tensors,
        budget=budget,
        reader=reader,
        owner="model.weights",
        excluded=excluded,
    )
    resources = OperationResources(budget)
    try:
        lookups = {
            name: factory.bind(partition, resources)
            for name, (partition, factory) in embeddings.items()
        }
        operators = {
            index: factory.bind(partition, resources)
            for index, (partition, factory) in experts.items()
        }
        return BoundOperations(allocation, resources, lookups, operators)
    except BaseException:
        try:
            resources.close()
        finally:
            allocation.close()
        raise
