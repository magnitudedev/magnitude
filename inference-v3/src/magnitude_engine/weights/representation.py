"""Numerical weight representations and their canonical engine layouts.

Containers stop at residency. Quantized representations describe values; the
layout below is the sole definition of the bytes inference kernels read.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeGuard

from magnitude_engine.platform.execution import DType

__all__ = [
    "Affine",
    "CanonicalLayout",
    "Code",
    "CodeInterpretation",
    "Codebook",
    "CoefficientScheme",
    "Dense",
    "DirectCoefficients",
    "HierarchicalCoefficients",
    "Representation",
    "WeightLayout",
    "canonical_layout",
    "resident_bytes",
    "is_scale_min_hierarchy",
]


@dataclass(frozen=True)
class Dense:
    dtype: DType


class CodeInterpretation(StrEnum):
    UNSIGNED = "unsigned"
    OFFSET_BINARY = "offset_binary"
    TWOS_COMPLEMENT = "twos_complement"


@dataclass(frozen=True)
class Code:
    low_bits: int
    high_bits: int = 0
    interpretation: CodeInterpretation = CodeInterpretation.UNSIGNED
    zero_point: int = 0

    def __post_init__(self):
        if self.low_bits not in (4, 8) or self.high_bits not in (0, 1, 2):
            raise ValueError("unsupported canonical code planes")
        if self.low_bits == 8 and self.high_bits:
            raise ValueError("eight-bit codes have no high plane")
        if self.interpretation == CodeInterpretation.OFFSET_BINARY:
            if not 0 < self.zero_point < 1 << self.bits:
                raise ValueError("offset-binary codes require an in-range zero point")
        elif self.zero_point:
            raise ValueError("only offset-binary codes carry a zero point")
        if self.interpretation == CodeInterpretation.TWOS_COMPLEMENT and self.bits != 8:
            raise ValueError("canonical two's-complement codes are eight bit")

    @property
    def bits(self) -> int:
        return self.low_bits + self.high_bits


@dataclass(frozen=True)
class DirectCoefficients:
    scale_dtype: DType
    bias_dtype: DType | None = None

    def __post_init__(self):
        if self.scale_dtype not in (DType.F16, DType.BF16, DType.F32):
            raise ValueError("direct affine scales must be floating point")
        if self.bias_dtype not in (None, DType.F16, DType.BF16, DType.F32):
            raise ValueError("direct affine biases must be floating point")

    @property
    def has_bias(self) -> bool:
        return self.bias_dtype is not None


@dataclass(frozen=True)
class HierarchicalCoefficients:
    supergroup: int
    local_scale_bits: int
    local_scale_interpretation: CodeInterpretation
    super_scale_dtype: DType
    local_scale_zero_point: int = 0
    local_bias_bits: int | None = None
    super_bias_dtype: DType | None = None
    bias_sign: int = 0

    def __post_init__(self):
        if self.supergroup <= 0 or self.local_scale_bits not in (6, 8):
            raise ValueError("invalid hierarchical coefficient geometry")
        if self.super_scale_dtype not in (DType.F16, DType.BF16, DType.F32):
            raise ValueError("hierarchical super scale must be floating point")
        if self.local_scale_interpretation == CodeInterpretation.OFFSET_BINARY:
            if not 0 < self.local_scale_zero_point < 1 << self.local_scale_bits:
                raise ValueError("offset-binary local scales require a zero point")
        elif self.local_scale_zero_point:
            raise ValueError("only offset-binary local scales carry a zero point")
        bias = self.local_bias_bits is not None
        if bias != (self.super_bias_dtype is not None):
            raise ValueError("local and super bias must be present together")
        if bias:
            if self.local_bias_bits not in (6, 8):
                raise ValueError("invalid hierarchical local bias width")
            if self.super_bias_dtype not in (DType.F16, DType.BF16, DType.F32):
                raise ValueError("hierarchical super bias must be floating point")
            if self.bias_sign not in (-1, 1):
                raise ValueError("hierarchical bias requires an explicit sign")
        elif self.bias_sign:
            raise ValueError("a bias sign requires bias coefficients")

    @property
    def has_bias(self) -> bool:
        return self.local_bias_bits is not None


type CoefficientScheme = DirectCoefficients | HierarchicalCoefficients


@dataclass(frozen=True)
class Affine:
    code: Code
    group: int
    coefficients: CoefficientScheme

    def __post_init__(self):
        if self.group <= 0:
            raise ValueError("an affine group must be positive")
        coefficients = self.coefficients
        if isinstance(coefficients, HierarchicalCoefficients):
            if coefficients.supergroup % self.group:
                raise ValueError("affine groups must tile their supergroup")

    @property
    def has_bias(self) -> bool:
        return self.coefficients.has_bias

    @property
    def supergroup(self) -> int | None:
        coefficients = self.coefficients
        return (
            coefficients.supergroup if isinstance(coefficients, HierarchicalCoefficients) else None
        )


@dataclass(frozen=True)
class Codebook:
    code_bits: int
    table: tuple[int, ...]
    group: int
    coefficients: CoefficientScheme

    def __post_init__(self):
        if self.code_bits not in (2, 4, 8) or len(self.table) != 1 << self.code_bits:
            raise ValueError("a codebook must define every packed code")
        if any(value < -128 or value > 127 for value in self.table):
            raise ValueError("canonical codebook entries are signed bytes")
        if self.group <= 0:
            raise ValueError("a codebook group must be positive")
        coefficients = self.coefficients
        if isinstance(coefficients, HierarchicalCoefficients):
            if coefficients.supergroup % self.group:
                raise ValueError("codebook groups must tile their supergroup")

    @property
    def supergroup(self) -> int | None:
        coefficients = self.coefficients
        return (
            coefficients.supergroup if isinstance(coefficients, HierarchicalCoefficients) else None
        )


type Representation = Dense | Affine | Codebook


def is_scale_min_hierarchy(representation: Representation) -> TypeGuard[Affine]:
    """Whether the representation matches the grouped Q4/Q5 scale/min family."""
    coefficients = representation.coefficients if isinstance(representation, Affine) else None
    return (
        isinstance(representation, Affine)
        and isinstance(coefficients, HierarchicalCoefficients)
        and representation.code.low_bits == 4
        and representation.code.high_bits in (0, 1)
        and representation.code.interpretation == CodeInterpretation.UNSIGNED
        and representation.group == 32
        and coefficients.supergroup == 256
        and coefficients.local_scale_bits == 6
        and coefficients.local_scale_interpretation == CodeInterpretation.UNSIGNED
        and coefficients.local_bias_bits == 6
        and coefficients.bias_sign == -1
        and coefficients.super_scale_dtype == DType.F16
        and coefficients.super_bias_dtype == DType.F16
    )


@dataclass(frozen=True)
class WeightLayout:
    """One logical row range over a complete resident allocation."""

    representation: Representation
    rows: int
    columns: int
    first_row: int = 0
    row_count: int | None = None

    def __post_init__(self):
        count = self.rows if self.row_count is None else self.row_count
        if min(self.rows, self.columns, count) <= 0 or not 0 <= self.first_row <= self.rows - count:
            raise ValueError("invalid resident weight row range")

    @property
    def logical_rows(self) -> int:
        return self.rows if self.row_count is None else self.row_count

    @property
    def elements(self) -> int:
        return self.rows * self.columns

    @property
    def nbytes(self) -> int:
        return resident_bytes(self.representation, self.elements)


@dataclass(frozen=True)
class CanonicalLayout:
    """Byte offsets in one canonical allocation.

    Hierarchical fields repeat once per supergroup. Direct fields are planes
    spanning the complete allocation. Offsets are relative to a tile for the
    former and absolute for the latter.
    """

    elements: int
    tile_elements: int
    tile_bytes: int
    low: int
    high: int | None
    scales: int
    biases: int | None
    super_scale: int | None
    super_bias: int | None
    nbytes: int
    hierarchical: bool


def _bytes(bits: int) -> int:
    if bits < 0 or bits % 8:
        raise ValueError("canonical fields must occupy whole bytes")
    return bits // 8


def canonical_layout(representation: Affine | Codebook, elements: int) -> CanonicalLayout:
    if elements <= 0 or elements % representation.group:
        raise ValueError("quantized storage requires complete groups")
    coefficients = representation.coefficients
    code_bits = (
        representation.code.low_bits
        if isinstance(representation, Affine)
        else representation.code_bits
    )
    high_bits = representation.code.high_bits if isinstance(representation, Affine) else 0

    if isinstance(coefficients, DirectCoefficients):
        groups = elements // representation.group
        low = 0
        high = _bytes(elements * code_bits) if high_bits else None
        scales = _bytes(elements * (code_bits + high_bits))
        biases = (
            scales + groups * coefficients.scale_dtype.itemsize if coefficients.has_bias else None
        )
        end = scales + groups * coefficients.scale_dtype.itemsize
        if coefficients.bias_dtype is not None:
            end += groups * coefficients.bias_dtype.itemsize
        return CanonicalLayout(
            elements, elements, end, low, high, scales, biases, None, None, end, False
        )

    tile_elements = coefficients.supergroup
    if elements % tile_elements:
        raise ValueError("hierarchical storage requires complete supergroups")
    groups = tile_elements // representation.group
    low = 0
    high = _bytes(tile_elements * code_bits) if high_bits else None
    scales = _bytes(tile_elements * (code_bits + high_bits))
    local_scale_bytes = _bytes(groups * coefficients.local_scale_bits)
    biases = scales + local_scale_bytes if coefficients.has_bias else None
    end = scales + local_scale_bytes
    if coefficients.local_bias_bits is not None:
        end += _bytes(groups * coefficients.local_bias_bits)
    super_scale = end
    end += coefficients.super_scale_dtype.itemsize
    super_bias = end if coefficients.super_bias_dtype is not None else None
    if coefficients.super_bias_dtype is not None:
        end += coefficients.super_bias_dtype.itemsize
    tiles = elements // tile_elements
    return CanonicalLayout(
        elements,
        tile_elements,
        end,
        low,
        high,
        scales,
        biases,
        super_scale,
        super_bias,
        tiles * end,
        True,
    )


def resident_bytes(representation: Representation, elements: int) -> int:
    if isinstance(representation, Dense):
        return elements * representation.dtype.itemsize
    return canonical_layout(representation, elements).nbytes
