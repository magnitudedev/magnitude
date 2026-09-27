"""Immutable authored programs, built once and reused at materialization."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from threading import RLock
from types import FunctionType
from typing import Any

from ..representations import Dense
from ..tensor.graph import Graph, Node, SourceLocation, ValueKind, _stable
from ..tensor.types import TensorSpec
from .dependencies import CodeDependency, code_dependencies, code_identity


class PortRole(StrEnum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read-write"
    WORKSPACE = "workspace"


@dataclass(frozen=True, slots=True)
class KernelPort:
    spec: TensorSpec
    role: PortRole
    offset: bool = True


@dataclass(frozen=True, slots=True)
class KernelDefinition:
    """The exact authored program consumed by module assembly and compilation."""

    identity: str
    name: str
    ports: tuple[KernelPort, ...]
    program: Any
    dependencies: tuple[CodeDependency, ...] = ()


def annotation(T, spec: TensorSpec, *, offset: bool = True) -> Any:
    # Resources include views into retained outputs and partially consumed input
    # features. Their DLPack byte offset is dynamic, even when shape is static.
    representation = spec.representation
    if representation is not None and not isinstance(representation, Dense):
        shape, dtype = ((spec.storage_nbytes + 3) // 4,), "uint32"
    else:
        shape = spec.shape
        dtype = (
            spec.dtype if not isinstance(representation, Dense) else representation.dtype
        ).value
    if not offset:
        return T.Tensor(shape, dtype)
    strides = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return T.buffer(shape, dtype, strides=tuple(reversed(strides)), offset_factor=1)


def _identity(value):
    """Structural authoring inputs, not repr(), addresses, or graph occurrence IDs."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, SourceLocation):
        return None
    if isinstance(value, (TensorSpec, Graph, Node)):
        return _stable(value)
    if isinstance(value, Mapping):
        return sorted((str(key), _identity(item)) for key, item in value.items())
    if isinstance(value, (set, frozenset)):
        return sorted((_identity(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (tuple, list)):
        return tuple(_identity(item) for item in value)
    if isinstance(value, FunctionType):
        return (value.__qualname__, code_identity(value), _identity(value.__defaults__),
                tuple(_identity(cell.cell_contents) for cell in value.__closure__ or ()))
    original = getattr(value, "orig_func", None)
    if isinstance(original, FunctionType):
        return _identity(original)
    if callable(getattr(value, "specialization_key", None)):
        return (type(value).__qualname__, code_identity(type(value)), _identity(value.specialization_key()))
    if is_dataclass(value):
        return (type(value).__qualname__, code_identity(type(value)),
                tuple((field.name, _identity(getattr(value, field.name))) for field in fields(value)))
    if hasattr(value, "__dict__"):
        return (type(value).__qualname__, code_identity(type(value)), _identity(vars(value)))
    raise TypeError(f"{type(value).__qualname__} cannot be an operation specialization parameter")


_DEFINITIONS: OrderedDict[str, KernelDefinition] = OrderedDict()
_DEFINITION_LIMIT = 512
_LOCK = RLock()


def define_kernel(emitter, ports: tuple[KernelPort, ...]) -> KernelDefinition:
    """Build symbolic TileLang only; never lowers, compiles, allocates or executes."""
    import tilelang.language as T

    import tilelang

    dependencies = code_dependencies(emitter)
    payload = (
        _identity(emitter),
        _identity(ports),
        tuple((item.module, item.symbol, item.fingerprint) for item in dependencies),
        tilelang.__version__,
    )
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with _LOCK:
        cached = _DEFINITIONS.get(identity)
        if cached is not None:
            _DEFINITIONS.move_to_end(identity)
            return cached
        name = f"ops_{identity[:24]}"
        buffers = tuple(
            (f"p{index}", annotation(T, port.spec, offset=port.offset))
            for index, port in enumerate(ports)
        )
        # Private launchers receive raw buffer pointers. Carry their element
        # offsets explicitly instead of leaving free variables in the launcher.
        parameters = (
            *buffers,
            *(
                (f"offset{index}", buffer.elem_offset)
                for index, (_, buffer) in enumerate(buffers)
                if ports[index].offset
            ),
        )

        def body(*bound):
            emitter(tuple(bound[: len(ports)]))

        program = T.build_prim_func(name, parameters, body)
        definition = KernelDefinition(identity, name, ports, program, dependencies)
        _DEFINITIONS[identity] = definition
        # Plans retain their own references; retiring this cache entry cannot
        # invalidate an executable or a still-live prepared formula.
        if len(_DEFINITIONS) > _DEFINITION_LIMIT:
            _DEFINITIONS.popitem(last=False)
        return definition


def operation_definition(graph: Graph, operation) -> KernelDefinition:
    reads, writes = set(operation.inputs), set(operation.outputs)

    def offset(value):
        tensor = graph.value(value)
        if tensor.kind in (ValueKind.INPUT, ValueKind.RESOURCE) or tensor.resource_id is not None:
            return True
        if tensor.producer is not None:
            for result, source in graph.nodes[tensor.producer].effects.aliases:
                if result == value:
                    return offset(source)
        return False

    ports = tuple(
        KernelPort(
            graph.value(value).spec,
            PortRole.READ_WRITE
            if value in reads and value in writes
            else PortRole.READ
            if value in reads
            else PortRole.WRITE,
            offset(value),
        )
        for value in (*operation.inputs, *operation.outputs)
    )
    ports += tuple(KernelPort(spec, PortRole.WORKSPACE, False) for spec in operation.workspace)
    return define_kernel(operation.emitter, ports)
