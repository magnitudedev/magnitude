"""Semantic operation contracts and the package operation registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .graph import Effects, Value
from .types import TensorSpec

type AbstractEvaluation = Callable[
    [tuple[TensorSpec, ...], Mapping[str, Any]], tuple[TensorSpec, ...]
]
type ReferenceEvaluation = Callable[[tuple[Any, ...], Mapping[str, Any]], tuple[Any, ...]]


@dataclass(frozen=True, slots=True)
class NumericalContract:
    accumulation_dtype: object | None = None
    output_rounding: str = "declared-output-dtype"
    exceptional_values: str = "propagate"


DEFAULT_NUMERICAL_CONTRACT = NumericalContract()


@dataclass(frozen=True, slots=True)
class Operation:
    name: str
    abstract: AbstractEvaluation
    reference: ReferenceEvaluation | None = None
    numerical: NumericalContract = DEFAULT_NUMERICAL_CONTRACT
    resource_reads: tuple[int, ...] = ()
    resource_writes: tuple[int, ...] = ()
    aliases: tuple[tuple[int, int], ...] = ()
    host_observation: bool = False
    tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.name or "." in self.name:
            raise ValueError("operation names must be non-empty unqualified identifiers")
        if len(set(self.resource_writes)) != len(self.resource_writes):
            raise ValueError("resource write operands must be unique")
        if len({output for output, _ in self.aliases}) != len(self.aliases):
            raise ValueError("an output may alias at most one input")

    def infer(
        self, inputs: tuple[TensorSpec, ...], attributes: Mapping[str, Any]
    ) -> tuple[TensorSpec, ...]:
        outputs = self.abstract(inputs, MappingProxyType(dict(attributes)))
        if not isinstance(outputs, tuple) or any(
            not isinstance(item, TensorSpec) for item in outputs
        ):
            raise TypeError(f"{self.name} abstract evaluation must return TensorSpec tuple")
        if not outputs:
            raise ValueError(f"{self.name} must produce at least one value")
        return outputs

    def effects(self, inputs: Sequence[Value], output_count: int) -> Effects:
        reads = tuple(_resource(inputs, index, self.name) for index in self.resource_reads)
        writes = []
        for index in self.resource_writes:
            value = inputs[index]
            resource = _resource(inputs, index, self.name)
            writes.append(
                (resource, value.resource_version or 0, (value.resource_version or 0) + 1)
            )
        for output, source in self.aliases:
            if not 0 <= output < output_count or not 0 <= source < len(inputs):
                raise ValueError(f"{self.name} has an invalid alias declaration")
        return Effects(reads, tuple(writes), self.aliases, self.host_observation)


def _resource(inputs: Sequence[Value], index: int, operation: str) -> int:
    if not 0 <= index < len(inputs) or inputs[index].resource_id is None:
        raise ValueError(f"{operation} operand {index} must be a mutable resource")
    resource_id = inputs[index].resource_id
    assert resource_id is not None
    return resource_id


class OperationRegistry:
    def __init__(self) -> None:
        self._operations: dict[str, Operation] = {}

    def register(self, operation: Operation) -> Operation:
        if operation.name in self._operations:
            raise ValueError(f"operation {operation.name!r} is already registered")
        self._operations[operation.name] = operation
        return operation

    def replace(self, operation: Operation) -> None:
        if operation.name not in self._operations:
            raise KeyError(operation.name)
        self._operations[operation.name] = operation

    def get(self, name: str) -> Operation:
        try:
            return self._operations[name]
        except KeyError as error:
            raise KeyError(f"unknown Magnitensor operation {name!r}") from error

    def __contains__(self, name: str) -> bool:
        return name in self._operations

    def __iter__(self):
        return iter(self._operations.values())


operations = OperationRegistry()


def operation(
    name: str,
    *,
    reference: ReferenceEvaluation | None = None,
    numerical: NumericalContract = DEFAULT_NUMERICAL_CONTRACT,
    resource_reads: tuple[int, ...] = (),
    resource_writes: tuple[int, ...] = (),
    aliases: tuple[tuple[int, int], ...] = (),
    host_observation: bool = False,
    tags: frozenset[str] = frozenset(),
):
    def decorate(abstract: AbstractEvaluation) -> AbstractEvaluation:
        operations.register(
            Operation(
                name,
                abstract,
                reference,
                numerical,
                resource_reads,
                resource_writes,
                aliases,
                host_observation,
                tags,
            )
        )
        return abstract

    return decorate


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    outputs: tuple[Any, ...]
    values: Mapping[int, Any] = field(default_factory=dict)


def evaluate_reference(graph, bindings: Mapping[int | str, Any]) -> ReferenceResult:
    values: dict[int, Any] = {}
    for value_id in (*graph.inputs, *graph.constants, *graph.resources):
        value = graph.value(value_id)
        if value_id in bindings:
            values[value_id] = bindings[value_id]
        elif value.name is not None and value.name in bindings:
            values[value_id] = bindings[value.name]
        else:
            raise KeyError(f"missing reference binding for {value.name or value_id}")
    for node in graph.nodes:
        definition = operations.get(node.operation)
        if definition.reference is None:
            raise NotImplementedError(f"{node.operation} has no reference evaluator")
        outputs = definition.reference(tuple(values[item] for item in node.inputs), node.attributes)
        if len(outputs) != len(node.outputs):
            raise ValueError(f"{node.operation} reference returned the wrong number of outputs")
        values.update(zip(node.outputs, outputs, strict=True))
    return ReferenceResult(tuple(values[item] for item in graph.outputs), MappingProxyType(values))
