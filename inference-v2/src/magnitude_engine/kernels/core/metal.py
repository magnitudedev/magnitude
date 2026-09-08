"""Typed boundaries around handwritten Metal. Evaluated only at specialization.

Index expressions describe tensor access, never numerical algorithms. Numerical
functions and ordered drivers live in Metal; this module declares their interfaces.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from types import MappingProxyType
from typing import Any

import mlx.core as mx

from ._emitter import Binary, Expression, Literal, Symbol, expression
from .graph import Tensor, Value
from .plan import Source, identifier


class Unsupported:
    pass


UNSUPPORTED = Unsupported()


@dataclass(frozen=True)
class Index:
    value: Expression

    def __add__(self, other):
        return Index(Binary("+", self.value, _index(other)))

    def __sub__(self, other):
        return Index(Binary("-", self.value, _index(other)))

    def __mul__(self, other):
        return Index(Binary("*", self.value, _index(other)))

    def __floordiv__(self, other):
        return Index(Binary("/", self.value, _index(other)))

    def __xor__(self, other):
        return Index(Binary("^", self.value, _index(other)))

    def __mod__(self, other):
        return Index(Binary("%", self.value, _index(other)))


def _index(value):
    if isinstance(value, Index):
        return value.value
    if type(value) is int and value >= 0:
        return Literal(value)
    raise TypeError("view coordinates require nonnegative integers or domain indices")


@dataclass(frozen=True, init=False)
class Domain:
    axes: tuple[tuple[str, int], ...]

    def __init__(self, **axes: int):
        if not axes or any(type(n) is not int or n <= 0 for n in axes.values()):
            raise ValueError("a tile domain requires positive static extents")
        for name in axes:
            identifier(name)
        object.__setattr__(self, "axes", tuple(axes.items()))

    @property
    def indices(self):
        return tuple(Index(Symbol(f"coord_{name}")) for name, _ in self.axes)

    @property
    def shape(self):
        return tuple(n for _, n in self.axes)

    @property
    def size(self):
        return prod(self.shape)


@dataclass(frozen=True)
class TensorSpec:
    value: Value

    @property
    def shape(self):
        return self.value.tensor.shape

    @property
    def dtype(self):
        return self.value.tensor.dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return self.value.tensor.size

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        if len(key) != self.ndim:
            raise ValueError("view must specify every tensor axis")
        stride, offset, remaining = 1, Literal(0), []
        saw_coordinate = False
        for extent, item in reversed(tuple(zip(self.shape, key, strict=True))):
            if isinstance(item, slice):
                if item != slice(None) or saw_coordinate:
                    raise ValueError(
                        "only contiguous suffix views are supported; use an explicit copy"
                    )
                remaining.append(extent)
            else:
                saw_coordinate = True
                if isinstance(item, int) and not 0 <= item < extent:
                    raise ValueError("view coordinate out of bounds")
                offset = Binary("+", Binary("*", _index(item), Literal(stride)), offset)
            stride *= extent
        return View(
            self,
            offset,
            tuple(reversed(remaining)),
            tuple(None if isinstance(k, slice) else _index(k) for k in key),
        )


@dataclass(frozen=True)
class View:
    tensor: TensorSpec
    offset: Expression
    shape: tuple[int, ...]
    coordinates: tuple[Expression | None, ...]


@dataclass(frozen=True)
class ReadOnly:
    view: View


@dataclass(frozen=True)
class Load:
    view: View

    def __post_init__(self):
        if self.view.shape:
            raise ValueError("Load requires one logical element")


@dataclass(frozen=True)
class UInt:
    value: int

    def __post_init__(self):
        if type(self.value) is not int or not 0 <= self.value < 2**32:
            raise ValueError("UInt requires a uint32 literal")


@dataclass(frozen=True)
class Float:
    value: float

    def __post_init__(self):
        import math

        if not math.isfinite(self.value):
            raise ValueError("Float requires a finite literal")


@dataclass(frozen=True)
class Lane:
    pass


@dataclass(frozen=True)
class SIMDGroup:
    size: int = 32

    def __post_init__(self):
        if self.size != 32:
            raise ValueError("the qualified Metal SIMD layout requires 32 lanes")


@dataclass(frozen=True)
class Thread:
    size: int = 1

    def __post_init__(self):
        if self.size != 1:
            raise ValueError("a thread scope has one participant")


@dataclass(frozen=True)
class Replicated:
    coordinate: tuple[Index, ...]
    dtype: mx.Dtype


@dataclass(frozen=True)
class Distributed(Replicated):
    """One completed value per lane; consecutive final-axis coordinates share a SIMD group."""


@dataclass(frozen=True)
class TileCall:
    domain: Domain
    scope: SIMDGroup | Thread
    arguments: Mapping[str, Any]
    result: Replicated
    template: tuple[Any, ...] = ()

    def __post_init__(self):
        if self.result.coordinate != self.domain.indices:
            raise ValueError("this backend requires one completed result per domain coordinate")
        if isinstance(self.result, Distributed) and (
            not isinstance(self.scope, SIMDGroup) or self.domain.shape[-1] % 32
        ):
            raise ValueError(
                "distributed scalar tiles require complete 32-element final-axis blocks"
            )
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        for name in self.arguments:
            identifier(name)
        for arg in self.arguments.values():
            if not isinstance(arg, (ReadOnly, Load, UInt, Float, Lane)):
                raise TypeError(f"unsupported Metal call argument: {type(arg).__name__}")
            if isinstance(arg, Lane) and not isinstance(self.scope, SIMDGroup):
                raise ValueError("Lane requires SIMD participation")


@dataclass(frozen=True)
class Iteration:
    axis: str
    extent: int
    start: Lane
    step: int = 32

    def __post_init__(self):
        identifier(self.axis)
        if self.extent < 0 or not isinstance(self.start, Lane) or self.step != 32:
            raise ValueError("ordered fold requires ascending lane-strided iteration")

    @property
    def index(self):
        return Index(Symbol(self.axis))


@dataclass(frozen=True)
class Sample:
    name: str
    view: View

    def __post_init__(self):
        identifier(self.name)
        if self.view.shape:
            raise ValueError("fold sample must be a scalar")


@dataclass(frozen=True)
class State:
    name: str
    dtype: mx.Dtype

    def __post_init__(self):
        identifier(self.name)


@dataclass(frozen=True)
class Call:
    function: str
    arguments: tuple[Any, ...]

    def __init__(self, function, *arguments):
        identifier(function)
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "arguments", arguments)


@dataclass(frozen=True)
class OrderedFold:
    domain: Domain
    scope: SIMDGroup
    arguments: Mapping[str, Any]
    iteration: Iteration
    shared_inputs: tuple[Sample, ...]
    state: tuple[State, ...]
    initial: Mapping[State, Float]
    step: Call
    finish: Call
    result: Replicated

    def __post_init__(self):
        call = TileCall(self.domain, self.scope, self.arguments, self.result)
        object.__setattr__(self, "arguments", call.arguments)
        object.__setattr__(self, "initial", MappingProxyType(dict(self.initial)))
        if len(self.state) != 1 or len(self.shared_inputs) != 1:
            raise ValueError("qualified scalar fold requires one state and one shared sample")
        if set(self.initial) != set(self.state):
            raise ValueError("every carried state needs exactly one initializer")
        allowed = (*self.state, *self.shared_inputs)
        for call in (self.step, self.finish):
            if any(
                not isinstance(arg, (Load, Float, UInt)) and arg not in allowed
                for arg in call.arguments
            ):
                raise ValueError("fold hook uses an undeclared state or sample")


@dataclass(frozen=True)
class Binding:
    source: Source
    function: str
    interface: Interface

    def __post_init__(self):
        identifier(self.function)
        validate_interface(self.interface)


def dtype_name(dtype):
    names = {
        mx.float32: "float",
        mx.float16: "half",
        mx.bfloat16: "bfloat",
        mx.int32: "int",
        mx.uint32: "uint",
        mx.bool_: "bool",
        mx.int64: "long",
        mx.uint64: "ulong",
        mx.int16: "short",
        mx.uint16: "ushort",
        mx.int8: "char",
        mx.uint8: "uchar",
    }
    try:
        return names[dtype]
    except KeyError:
        raise ValueError(f"unsupported Metal dtype: {dtype}") from None


def argument(value, *, fields=False):
    if isinstance(value, ReadOnly):
        return f"({value.view.tensor.value.name} + {expression(value.view.offset)})"
    if isinstance(value, Load):
        return f"{value.view.tensor.value.name}[{expression(value.view.offset)}]"
    if isinstance(value, UInt):
        return f"{value.value}u"
    if isinstance(value, Float):
        return f"{float(value.value)!r}f"
    if isinstance(value, Lane):
        return "lane"
    raise TypeError(f"unsupported Metal argument: {type(value).__name__}")


@dataclass(frozen=True)
class Threadgroup:
    size: int

    def __post_init__(self):
        if not 32 <= self.size <= 1024 or self.size % 32:
            raise ValueError("cooperative scope requires complete SIMD groups")


@dataclass(frozen=True)
class RowTransform:
    """A qualified driver consumes scalar inputs and returns completed row elements.

    The driver calls Body.input(index) and Body.output(index, result, original),
    uniformly according to its declared row coverage. It owns the reduction;
    output hooks run only after all participants complete that reduction.
    """

    output: Tensor
    input: Value
    scope: Threadgroup
    arguments: Mapping[str, Any]
    template: tuple[Any, ...] = ()

    def __post_init__(self):
        if not self.output.shape or self.output.shape[-1] < 1:
            raise ValueError("row driver requires a nonempty feature dimension")
        if self.output != self.input.tensor:
            raise ValueError("row transform must preserve input geometry and native dtype")
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        for name, value in self.arguments.items():
            identifier(name)
            if not isinstance(
                value,
                (ReadOnly, UInt, Float, GroupPosition, ThreadPosition, Lane, SIMDIndex, Scratch),
            ):
                raise TypeError("unsupported cooperative driver argument")
        # This profile deliberately limits statically allocated shared storage.
        # It is a feasibility rule, not an estimate of occupancy or a performance ceiling.
        sizes = {mx.float32: 4, mx.float16: 2, mx.bfloat16: 2, mx.uint32: 4, mx.int32: 4}
        if (
            sum(a.count * sizes[a.dtype] for a in self.arguments.values() if isinstance(a, Scratch))
            > 32768
        ):
            raise ValueError("cooperative driver exceeds qualified shared storage")


@dataclass(frozen=True)
class BlockedRows:
    """Complete row/channel fragments, optionally addressed by an explicit permutation.

    Each SIMD group owns consecutive channels for a block of rows. Permutation
    operands must be a bijection of the row domain; their construction is qualified
    by the caller. Matching layouts require the same actual permutation value.
    """

    rows: int
    columns: int
    tile_rows: int
    permutation: Value | None = None
    channels: int = 4
    groups: int = 2

    def __post_init__(self):
        if min(self.rows, self.columns, self.tile_rows) < 1 or self.tile_rows > 4:
            raise ValueError("blocked rows require positive extents and at most four live rows")
        if self.channels != 4 or self.groups != 2:
            raise ValueError("unqualified blocked fragment layout")
        if self.permutation is not None and (
            self.permutation.tensor.size != self.rows or self.permutation.tensor.dtype != mx.uint32
        ):
            raise ValueError("row permutation must contain one uint32 entry per logical row")

    @property
    def items(self):
        return self.tile_rows * self.channels


@dataclass(frozen=True)
class FragmentCall:
    """A function returns completed native values in a declared blocked layout."""

    output: Tensor
    layout: BlockedRows
    arguments: Mapping[str, Any]
    template: tuple[Any, ...]

    def __post_init__(self):
        if self.output.size != self.layout.rows * self.layout.columns:
            raise ValueError("fragment layout does not cover operation output")
        if self.output.shape[-1] != self.layout.columns:
            raise ValueError("blocked fragment columns must match final tensor axis")
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        for name in self.arguments:
            identifier(name)


@dataclass(frozen=True)
class GroupPosition:
    axis: str

    def __post_init__(self):
        if self.axis not in ("x", "y", "z"):
            raise ValueError("invalid grid axis")


@dataclass(frozen=True)
class ThreadPosition:
    axis: str = "x"

    def __post_init__(self):
        if self.axis not in ("x", "y", "z"):
            raise ValueError("invalid threadgroup axis")


@dataclass(frozen=True)
class Scratch:
    dtype: mx.Dtype
    count: int

    def __post_init__(self):
        if self.dtype not in (mx.float32, mx.float16, mx.bfloat16, mx.uint32, mx.int32):
            raise ValueError("unqualified scratch element type")
        if type(self.count) is not int or self.count < 1:
            raise ValueError("scratch requires a positive static extent")


@dataclass(frozen=True)
class SIMDIndex:
    pass


@dataclass(frozen=True)
class RowIndices:
    """Private logical row indices; negative indices are masked tail rows."""

    pass


@dataclass(frozen=True)
class ColumnStart:
    pass


@dataclass(frozen=True)
class MetalType:
    """Typed aggregate constructor for a handwritten step implementation."""

    name: str
    template: tuple[Any, ...]
    arguments: tuple[Any, ...]

    def __post_init__(self):
        identifier(self.name)


@dataclass(frozen=True)
class FragmentFold(FragmentCall):
    """Qualified pack driver: prepare(k, first), step(state, row, pack, sum), finish.

    Branch states are independent. The driver owns the exact lane/K traversal and
    input preparation. Compatible branches share it without changing either order.
    """

    body: MetalType
    pack: int

    def __post_init__(self):
        super().__post_init__()
        if self.pack < 1 or self.pack > 32:
            raise ValueError("unqualified pack extent")


@dataclass(frozen=True)
class OrderedReduction:
    """An ascending penultimate-axis fold of completed producer elements."""

    input: Value
    output: Tensor
    body: MetalType

    def __post_init__(self):
        shape = self.input.tensor.shape
        if len(shape) < 2 or shape[-2] < 1 or self.output.shape != (*shape[:-2], shape[-1]):
            raise ValueError("ordered reduction must remove its nonempty penultimate axis")
        if self.output.dtype != self.input.tensor.dtype:
            raise ValueError("qualified ordered fragment fold preserves native dtype")


@dataclass(frozen=True)
class ArgumentType:
    """Preserve MLX's device/constant address space in a Metal template argument."""

    argument: Any


def _range(value, axes):
    """Conservative integer intervals; rejection never substitutes sampling for proof."""
    match value:
        case Literal(n) if type(n) is int:
            return n, n
        case Symbol(name) if name in axes:
            return axes[name]
        case Binary(op, left, right):
            a, b = _range(left, axes), _range(right, axes)
            if op == "+":
                return a[0] + b[0], a[1] + b[1]
            if op == "-":
                return a[0] - b[1], a[1] - b[0]
            if op == "*" and min(*a, *b) >= 0:
                return a[0] * b[0], a[1] * b[1]
            if op == "/" and a[0] >= 0 and b[0] > 0:
                return a[0] // b[1], a[1] // b[0]
            if op == "%" and a[0] >= 0 and b[0] > 0:
                return 0, b[1] - 1
            if op == "^" and a[0] >= 0 and b[0] == b[1] and b[0] >= 0:
                # Highest possible changed bit gives a conservative enclosing interval.
                mask = (1 << b[0].bit_length()) - 1
                return 0, a[1] | mask
    raise ValueError("view indexing has no qualified range proof")


def validate_interface(interface):
    from dataclasses import fields, is_dataclass

    axes = {}
    if isinstance(interface, (TileCall, OrderedFold)):
        axes.update((f"coord_{name}", (0, extent - 1)) for name, extent in interface.domain.axes)
    if isinstance(interface, OrderedFold):
        axes[interface.iteration.axis] = (0, interface.iteration.extent - 1)
        if isinstance(interface.result, Distributed):
            raise ValueError("ordered scalar fold returns a replicated completed result")

    def validate(value):
        if isinstance(value, View):
            for coordinate, extent in zip(value.coordinates, value.tensor.shape, strict=True):
                if coordinate is not None:
                    lo, hi = _range(coordinate, axes)
                    if not 0 <= lo <= hi < extent:
                        raise ValueError("view coordinate can leave its declared tensor domain")
        elif isinstance(value, Mapping):
            for child in value.values():
                validate(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                validate(child)
        elif is_dataclass(value):
            for field in fields(value):
                validate(getattr(value, field.name))

    validate(interface)


type Interface = TileCall | OrderedFold | RowTransform | FragmentCall | OrderedReduction
