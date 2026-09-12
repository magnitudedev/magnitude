"""Tensor types and immutable abstract values."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..representations import Representation


class DType(StrEnum):
    BOOL = "bool"
    U8 = "uint8"
    U16 = "uint16"
    U32 = "uint32"
    I8 = "int8"
    I16 = "int16"
    I32 = "int32"
    I64 = "int64"
    F16 = "float16"
    BF16 = "bfloat16"
    F32 = "float32"

    @property
    def itemsize(self) -> int:
        return _ITEM_BYTES[self]

    @property
    def floating(self) -> bool:
        return self in (DType.F16, DType.BF16, DType.F32)

    @property
    def integer(self) -> bool:
        return self not in (DType.BOOL, DType.F16, DType.BF16, DType.F32)


_ITEM_BYTES = {
    DType.BOOL: 1,
    DType.U8: 1,
    DType.U16: 2,
    DType.U32: 4,
    DType.I8: 1,
    DType.I16: 2,
    DType.I32: 4,
    DType.I64: 8,
    DType.F16: 2,
    DType.BF16: 2,
    DType.F32: 4,
}


@dataclass(frozen=True, slots=True)
class Dim:
    """A named positive dimension bound at specialization time."""

    name: str
    minimum: int = 1
    maximum: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dimension name must not be empty")
        if self.minimum <= 0 or (self.maximum is not None and self.maximum < self.minimum):
            raise ValueError("invalid symbolic dimension bounds")

    def bind(self, value: int) -> int:
        if type(value) is not int or value < self.minimum:
            raise ValueError(f"{self.name} must be at least {self.minimum}, got {value!r}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.name} must be at most {self.maximum}, got {value}")
        return value


type ShapeDim = int | Dim
type Shape = tuple[ShapeDim, ...]


@dataclass(frozen=True, slots=True)
class Layout:
    """Observable logical strides; ``None`` means dense row-major."""

    strides: tuple[int, ...] | None = None
    tag: str = "dense"

    def __post_init__(self) -> None:
        if not self.tag:
            raise ValueError("layout tag must not be empty")
        if self.strides is not None and any(type(v) is not int or v < 0 for v in self.strides):
            raise ValueError("layout strides must be non-negative integers")


DENSE = Layout()


@dataclass(frozen=True, slots=True)
class TensorSpec:
    shape: Shape
    dtype: DType
    layout: Layout = DENSE
    representation: Representation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple):
            raise TypeError("tensor shape must be an immutable tuple")
        if any((type(v) is int and v <= 0) or not isinstance(v, (int, Dim)) for v in self.shape):
            raise ValueError("tensor extents must be positive integers or dimensions")
        if not isinstance(self.dtype, DType):
            raise TypeError("tensor dtype must be a DType")
        if self.layout.strides is not None and len(self.layout.strides) != len(self.shape):
            raise ValueError("layout rank differs from tensor rank")

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def static(self) -> bool:
        return all(type(v) is int for v in self.shape)

    @property
    def elements(self) -> int:
        if not self.static:
            raise ValueError("symbolic tensor has no fixed element count")
        return math.prod(self.shape)  # type: ignore[arg-type]

    @property
    def nbytes(self) -> int:
        return self.elements * self.dtype.itemsize

    @property
    def storage_nbytes(self) -> int:
        """Physical bytes required by the declared representation."""
        if self.representation is None:
            return self.nbytes
        from ..representations import Dense, represented_nbytes

        if isinstance(self.representation, Dense):
            return represented_nbytes(self.representation, self.elements)
        return represented_nbytes(self.representation, self.elements)

    def bind(self, bindings: dict[str, int]) -> TensorSpec:
        shape = tuple(v if isinstance(v, int) else v.bind(bindings[v.name]) for v in self.shape)
        return TensorSpec(shape, self.dtype, self.layout, self.representation)

    def with_layout(self, layout: Layout) -> TensorSpec:
        return TensorSpec(self.shape, self.dtype, layout, self.representation)

    def with_representation(self, representation: Representation | None) -> TensorSpec:
        return TensorSpec(self.shape, self.dtype, self.layout, representation)


def dense_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


def broadcast_shape(*shapes: Shape) -> Shape:
    if not shapes:
        return ()
    width = max(map(len, shapes))
    result: list[ShapeDim] = []
    for axis in range(width):
        values = [shape[-axis - 1] if axis < len(shape) else 1 for shape in shapes]
        concrete = {v for v in values if type(v) is int and v != 1}
        symbolic = {v for v in values if isinstance(v, Dim)}
        if len(concrete) > 1 or len(symbolic) > 1 or (concrete and symbolic):
            raise ValueError(f"cannot broadcast dimensions {values!r}")
        result.append(next(iter(symbolic or concrete), 1))
    return tuple(reversed(result))


def normalize_axes(rank: int, axes: int | tuple[int, ...]) -> tuple[int, ...]:
    values = (axes,) if isinstance(axes, int) else axes
    normalized = tuple(axis + rank if axis < 0 else axis for axis in values)
    if len(set(normalized)) != len(normalized) or any(not 0 <= axis < rank for axis in normalized):
        raise ValueError(f"invalid axes {axes!r} for rank {rank}")
    return normalized
