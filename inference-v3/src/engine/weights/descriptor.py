"""Logical weight values, and what a container yields for one of them.

A descriptor names a role the model asked for. A ``Stored`` value is what a
format found on disk for that role: a byte source, an offset, and the layout the
bytes are in. Neither says anything about how a kernel will read the weight;
that is a ``Representation``, decided at residency.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pydantic import PositiveInt

import ops
from engine.data import Record
from ops.binding import ByteSource


class WeightTransform(StrEnum):
    IDENTITY = "identity"
    NEGATIVE_EXP = "negative_exp"


class WeightDescriptor(Record):
    name: str
    shape: tuple[PositiveInt, ...]
    transform: WeightTransform = WeightTransform.IDENTITY


class TraceValue(Protocol):
    """Scalar expression accepted by the import kernel's tracing frontend."""

    def astype(self, dtype: str) -> TraceValue: ...
    def __add__(self, other: int | TraceValue) -> TraceValue: ...
    def __radd__(self, other: int | TraceValue) -> TraceValue: ...
    def __mul__(self, other: int | TraceValue) -> TraceValue: ...
    def __rmul__(self, other: int | TraceValue) -> TraceValue: ...
    def __floordiv__(self, other: int | TraceValue) -> TraceValue: ...
    def __mod__(self, other: int | TraceValue) -> TraceValue: ...
    def __lshift__(self, other: int | TraceValue) -> TraceValue: ...
    def __rshift__(self, other: int | TraceValue) -> TraceValue: ...
    def __and__(self, other: int | TraceValue) -> TraceValue: ...
    def __or__(self, other: int | TraceValue) -> TraceValue: ...
    def __neg__(self) -> TraceValue: ...
    def __lt__(self, other: int | TraceValue) -> TraceValue: ...


class TraceBuffer(Protocol):
    """Indexable source operand visible to a format codec during tracing."""

    def __getitem__(self, index: int | TraceValue) -> TraceValue: ...


class SourceCodec(Protocol):
    """Format-owned logical reader used only by residency-time relayout."""

    @property
    def block_elements(self) -> int: ...

    @property
    def block_bytes(self) -> int: ...

    def code(
        self, data: TraceBuffer, base: int | TraceValue, index: int | TraceValue
    ) -> TraceValue: ...
    def local_scale(
        self,
        data: TraceBuffer,
        base: int | TraceValue,
        group: int | TraceValue,
        select: Callable[[Any, Any, Any], Any],
    ) -> TraceValue: ...
    def local_bias(
        self,
        data: TraceBuffer,
        base: int | TraceValue,
        group: int | TraceValue,
        select: Callable[[Any, Any, Any], Any],
    ) -> TraceValue: ...
    def scale_byte(
        self, data: TraceBuffer, base: int | TraceValue, byte_index: int | TraceValue
    ) -> TraceValue: ...
    def bias_byte(
        self, data: TraceBuffer, base: int | TraceValue, byte_index: int | TraceValue
    ) -> TraceValue: ...
    def super_scale(self, data: TraceBuffer, base: int | TraceValue,
                    reinterpret: Callable[[Any, str], Any]) -> TraceValue: ...
    def super_bias(self, data: TraceBuffer, base: int | TraceValue,
                   reinterpret: Callable[[Any, str], Any]) -> TraceValue: ...
    def direct_scale(
        self,
        data: TraceBuffer,
        base: int | TraceValue,
        group: int | TraceValue,
        select: Callable[[Any, Any, Any], Any],
        reinterpret: Callable[[Any, str], Any],
    ) -> TraceValue: ...
    def direct_bias(
        self,
        data: TraceBuffer,
        base: int | TraceValue,
        group: int | TraceValue,
        select: Callable[[Any, Any, Any], Any],
        reinterpret: Callable[[Any, str], Any],
    ) -> TraceValue: ...


@dataclass(frozen=True)
class StoredQuantized:
    """Container bytes plus their one import-only logical reader."""

    representation: ops.Affine | ops.Codebook
    codec: SourceCodec
    source: ByteSource
    offset: int


@dataclass(frozen=True)
class StoredDense:
    dtype: ops.DType
    source: ByteSource
    offset: int
    nbytes: int


@dataclass(frozen=True)
class StoredAffinePlanes:
    """Affine codes and per-group coefficients held as three separate tensors."""

    bits: int
    group: int
    codes: StoredDense
    scales: StoredDense
    biases: StoredDense


type Stored = StoredQuantized | StoredDense | StoredAffinePlanes
