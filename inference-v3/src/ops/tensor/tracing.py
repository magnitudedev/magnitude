"""Tensor syntax and deterministic tracing."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .graph import Graph, Node, SourceLocation, Value, ValueKind
from .primitive import primitives
from .types import DType, TensorSpec

_ACTIVE_TRACE: ContextVar[Trace | None] = ContextVar("ops_trace", default=None)


@dataclass(frozen=True, slots=True)
class Argument:
    spec: TensorSpec
    name: str | None = None
    kind: ValueKind = ValueKind.INPUT

    def __post_init__(self) -> None:
        if self.kind not in (ValueKind.INPUT, ValueKind.CONSTANT, ValueKind.RESOURCE):
            raise ValueError("trace arguments must be inputs, constants or resources")


@dataclass(frozen=True, slots=True)
class Signature:
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any] = MappingProxyType({})
    static_kwargs: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", MappingProxyType(dict(self.kwargs)))
        object.__setattr__(self, "static_kwargs", MappingProxyType(dict(self.static_kwargs)))


@dataclass(frozen=True, slots=True)
class Tensor:
    _trace: Trace
    value_id: int

    @property
    def spec(self) -> TensorSpec:
        return self._trace.values[self.value_id].spec

    @property
    def shape(self):
        return self.spec.shape

    @property
    def dtype(self) -> DType:
        return self.spec.dtype

    @property
    def T(self) -> Tensor:
        from .ops import transpose

        return transpose(self, tuple(reversed(range(self.spec.rank))))

    def __bool__(self) -> bool:
        raise TypeError("a dynamic Tensor cannot control Python execution")

    def __add__(self, other: Tensor | int | float) -> Tensor:
        from .ops import add

        return add(self, other)

    def __radd__(self, other: Tensor | int | float) -> Tensor:
        return self + other

    def __sub__(self, other: Tensor | int | float) -> Tensor:
        from .ops import subtract

        return subtract(self, other)

    def __rsub__(self, other: Tensor | int | float) -> Tensor:
        from .ops import subtract

        return subtract(other, self)

    def __mul__(self, other: Tensor | int | float) -> Tensor:
        from .ops import multiply

        return multiply(self, other)

    def __rmul__(self, other: Tensor | int | float) -> Tensor:
        return self * other

    def __truediv__(self, other: Tensor | int | float) -> Tensor:
        from .ops import divide

        return divide(self, other)

    def __matmul__(self, other: Tensor) -> Tensor:
        from .ops import matmul

        return matmul(self, other)

    def reshape(self, *shape: int) -> Tensor:
        from .ops import reshape

        return reshape(self, shape)

    def astype(self, dtype: DType) -> Tensor:
        from .ops import cast

        return cast(self, dtype)


class Trace:
    def __init__(self, name: str):
        self.name = name
        self.values: list[Value] = []
        self.nodes: list[Node] = []
        self.inputs: list[int] = []
        self.constants: list[int] = []
        self.resources: list[int] = []
        self._resource_versions: dict[int, int] = {}
        self.formula_calls: list[Any] = []
        self.formula_stack: list[int] = []
        self.formula_quantities: dict[int, list[Any]] = {}

    def argument(self, argument: Argument, position: int | str) -> Tensor:
        name = argument.name or (f"arg{position}" if isinstance(position, int) else position)
        resource_id = None
        resource_version = None
        if argument.kind == ValueKind.RESOURCE:
            resource_id = len(self.resources)
            resource_version = 0
            self._resource_versions[resource_id] = 0
        value = Value(
            len(self.values),
            argument.spec,
            argument.kind,
            name=name,
            resource_id=resource_id,
            resource_version=resource_version,
        )
        self.values.append(value)
        getattr(self, f"{argument.kind.value}s").append(value.id)
        return Tensor(self, value.id)

    def literal(self, value: int | float | bool, *, dtype: DType | None = None) -> Tensor:
        from .ops import scalar

        return scalar(value, dtype=dtype)

    def emit(
        self, name: str, inputs: tuple[Tensor, ...], attributes: Mapping[str, Any]
    ) -> tuple[Tensor, ...]:
        if any(tensor._trace is not self for tensor in inputs):
            raise ValueError("cannot combine tensors from different traces")
        definition = primitives.get(name)
        input_values = tuple(self.values[tensor.value_id] for tensor in inputs)
        for value in input_values:
            if value.resource_id is None:
                continue
            current = self._resource_versions[value.resource_id]
            if value.resource_version != current:
                raise ValueError(
                    f"{name} reads stale resource version {value.resource_version}; "
                    f"current version is {current}"
                )
        specs = definition.infer(tuple(value.spec for value in input_values), attributes)
        effects = definition.effects(input_values, len(specs))
        node_id = len(self.nodes)
        source = _source_location()
        outputs = []
        alias_by_output = dict(definition.aliases)
        write_by_resource = {resource: after for resource, _, after in effects.writes}
        for index, spec in enumerate(specs):
            resource_id = None
            resource_version = None
            source_index = alias_by_output.get(index)
            if source_index is not None:
                source_value = input_values[source_index]
                resource_id = source_value.resource_id
                resource_version = source_value.resource_version
                if resource_id in write_by_resource:
                    resource_version = write_by_resource[resource_id]
                    self._resource_versions[resource_id] = resource_version
            value = Value(
                len(self.values),
                spec,
                ValueKind.NODE,
                producer=node_id,
                output_index=index,
                resource_id=resource_id,
                resource_version=resource_version,
            )
            self.values.append(value)
            outputs.append(value.id)
        self.nodes.append(
            Node(
                node_id,
                name,
                tuple(item.value_id for item in inputs),
                attributes,
                tuple(outputs),
                effects,
                source,
            )
        )
        return tuple(Tensor(self, value_id) for value_id in outputs)

    def finish(self, outputs: Any) -> Graph:
        from ..formula import FormulaIndex

        flat = _flatten_outputs(outputs)
        if any(tensor._trace is not self for tensor in flat):
            raise ValueError("trace output belongs to another trace")
        return Graph(
            self.name,
            tuple(self.values),
            tuple(self.nodes),
            tuple(self.inputs),
            tuple(self.constants),
            tuple(self.resources),
            tuple(tensor.value_id for tensor in flat),
            FormulaIndex(tuple(self.formula_calls)),
        )


def trace(function, signature: Signature, *, name: str | None = None) -> Graph:
    active = _ACTIVE_TRACE.get()
    if active is not None:
        raise RuntimeError("nested Ops tracing is not supported")
    state = Trace(name or getattr(function, "__name__", "tensor_function"))
    token = _ACTIVE_TRACE.set(state)
    try:
        from .tree import map_tree

        def bind(argument, path):
            if isinstance(argument, Argument):
                return state.argument(argument, ".".join(map(str, path)))
            return argument

        args = tuple(map_tree(argument, bind, (f"arg{index}",))
                     for index, argument in enumerate(signature.args))
        kwargs = {key: map_tree(argument, bind, (key,)) for key, argument in signature.kwargs.items()}
        kwargs.update(signature.static_kwargs)
        return state.finish(function(*args, **kwargs))
    finally:
        _ACTIVE_TRACE.reset(token)


def active_trace() -> Trace:
    trace = _ACTIVE_TRACE.get()
    if trace is None:
        raise RuntimeError("Ops primitives may only run while tracing")
    return trace


def as_tensor(value: Tensor | int | float | bool, *, like: Tensor | None = None) -> Tensor:
    if isinstance(value, Tensor):
        return value
    dtype = like.dtype if like is not None else None
    return active_trace().literal(value, dtype=dtype)


def emit(name: str, *inputs: Tensor, **attributes: Any) -> Tensor | tuple[Tensor, ...]:
    outputs = active_trace().emit(name, inputs, attributes)
    return outputs[0] if len(outputs) == 1 else outputs


def _flatten_outputs(value: Any) -> tuple[Tensor, ...]:
    from .tree import leaves

    result = []
    for _, leaf in leaves(value):
        if not isinstance(leaf, Tensor):
            raise TypeError("tensor function output leaves must be tensors")
        result.append(leaf)
    if not result:
        raise ValueError("tensor functions must have at least one tensor output")
    return tuple(result)


def _source_location() -> SourceLocation | None:
    frame = inspect.currentframe()
    try:
        while frame is not None:
            filename = frame.f_code.co_filename
            if "/ops/" not in filename:
                return SourceLocation(filename, frame.f_lineno, frame.f_code.co_name)
            frame = frame.f_back
    finally:
        del frame
    return None
