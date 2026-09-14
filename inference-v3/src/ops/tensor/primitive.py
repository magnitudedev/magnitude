"""Semantic operation contracts and the package operation registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .graph import Effects, Value
from .types import DType, TensorSpec

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


def round_reference(value, dtype: DType):
    """Logical output rounding for the independent reference (never execution).

    BF16 values use exact FP32 carriers in NumPy; ties-to-even rounding preserves
    their BF16 value set without introducing another numerical runtime dependency.
    """
    import numpy as np

    if dtype != DType.BF16:
        return np.asarray(value).astype(dtype.value, copy=False)
    values = np.asarray(value, dtype=np.float32)
    bits = values.view(np.uint32)
    nan = ((bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)) & ((bits & np.uint32(0x007FFFFF)) != 0)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    # A NaN with payload only in its low bits must not round to infinity.
    rounded = np.where(nan, (bits >> np.uint32(16)) | np.uint32(0x40), rounded).astype(np.uint32)
    return (rounded << np.uint32(16)).view(np.float32)


@dataclass(frozen=True, slots=True)
class Primitive:
    name: str
    abstract: AbstractEvaluation
    reference: ReferenceEvaluation | None = None
    numerical: NumericalContract = DEFAULT_NUMERICAL_CONTRACT
    resource_reads: tuple[int, ...] = ()
    resource_writes: tuple[int, ...] = ()
    aliases: tuple[tuple[int, int], ...] = ()
    host_observation: bool = False
    tags: frozenset[str] = frozenset()
    work: Callable | None = None

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
        # Pure numerical primitives can consume mutable resources too. Derive
        # those read hazards from their actual ports so a later writer cannot
        # overtake a delayed consumer merely because the primitive also accepts
        # ordinary immutable tensors. Explicit reads still validate required state.
        reads = tuple(dict.fromkeys((
            *(_resource(inputs, index, self.name) for index in self.resource_reads),
            *(value.resource_id for value in inputs if value.resource_id is not None),
        )))
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

    def evaluate(self, inputs, attributes, outputs: tuple[TensorSpec, ...]) -> tuple[Any, ...]:
        import numpy as np

        if self.reference is None:
            raise NotImplementedError(f"{self.name} has no reference evaluator")
        readonly = []
        for value in inputs:
            view = np.asarray(value).view()
            view.flags.writeable = False
            readonly.append(view)
        values = self.reference(tuple(readonly), attributes)
        if not isinstance(values, tuple) or len(values) != len(outputs):
            raise ValueError(f"{self.name} reference returned the wrong number of outputs")
        result = tuple(round_reference(value, spec.dtype) for value, spec in zip(values, outputs, strict=True))
        if any(value.shape != spec.shape for value, spec in zip(result, outputs, strict=True)):
            raise ValueError(f"{self.name} reference returned the wrong output shape")
        return result


def _resource(inputs: Sequence[Value], index: int, operation: str) -> int:
    if not 0 <= index < len(inputs) or inputs[index].resource_id is None:
        raise ValueError(f"{operation} operand {index} must be a mutable resource")
    resource_id = inputs[index].resource_id
    assert resource_id is not None
    return resource_id


class PrimitiveRegistry:
    def __init__(self) -> None:
        self._operations: dict[str, Primitive] = {}

    def register(self, operation: Primitive) -> Primitive:
        if operation.name in self._operations:
            raise ValueError(f"operation {operation.name!r} is already registered")
        self._operations[operation.name] = operation
        return operation

    def replace(self, operation: Primitive) -> None:
        if operation.name not in self._operations:
            raise KeyError(operation.name)
        self._operations[operation.name] = operation

    def get(self, name: str) -> Primitive:
        try:
            return self._operations[name]
        except KeyError as error:
            raise KeyError(f"unknown Ops operation {name!r}") from error

    def __contains__(self, name: str) -> bool:
        return name in self._operations

    def __iter__(self):
        return iter(self._operations.values())


primitives = PrimitiveRegistry()


def primitive(
    name: str,
    *,
    reference: ReferenceEvaluation | None = None,
    numerical: NumericalContract = DEFAULT_NUMERICAL_CONTRACT,
    resource_reads: tuple[int, ...] = (),
    resource_writes: tuple[int, ...] = (),
    aliases: tuple[tuple[int, int], ...] = (),
    host_observation: bool = False,
    tags: frozenset[str] = frozenset(),
    work: Callable | None = None,
):
    def decorate(abstract: AbstractEvaluation) -> AbstractEvaluation:
        primitives.register(
            Primitive(
                name,
                abstract,
                reference,
                numerical,
                resource_reads,
                resource_writes,
                aliases,
                host_observation,
                tags,
                work,
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
        definition = primitives.get(node.operation)
        outputs = definition.evaluate(tuple(values[item] for item in node.inputs), node.attributes,
                                      tuple(graph.value(item).spec for item in node.outputs))
        values.update(zip(node.outputs, outputs, strict=True))
    return ReferenceResult(tuple(values[item] for item in graph.outputs), MappingProxyType(values))
