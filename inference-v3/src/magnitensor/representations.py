"""Numerical representations independent of artifact containers and targets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .tensor.types import DType


@dataclass(frozen=True, slots=True)
class Dense:
    dtype: DType


class CodeInterpretation(StrEnum):
    UNSIGNED = "unsigned"
    OFFSET_BINARY = "offset_binary"
    TWOS_COMPLEMENT = "twos_complement"


@dataclass(frozen=True, slots=True)
class Code:
    low_bits: int
    high_bits: int = 0
    interpretation: CodeInterpretation = CodeInterpretation.UNSIGNED
    zero_point: int = 0

    def __post_init__(self) -> None:
        if self.low_bits not in (2, 4, 8) or self.high_bits not in (0, 1, 2):
            raise ValueError("unsupported encoded value width")
        if self.low_bits == 8 and self.high_bits:
            raise ValueError("eight-bit values cannot have a high plane")
        if self.interpretation == CodeInterpretation.OFFSET_BINARY:
            if not 0 < self.zero_point < 1 << self.bits:
                raise ValueError("offset-binary values need an in-range zero point")
        elif self.zero_point:
            raise ValueError("only offset-binary values carry a zero point")

    @property
    def bits(self) -> int:
        return self.low_bits + self.high_bits


@dataclass(frozen=True, slots=True)
class DirectCoefficients:
    scale_dtype: DType
    bias_dtype: DType | None = None

    def __post_init__(self) -> None:
        if not self.scale_dtype.floating or (
            self.bias_dtype is not None and not self.bias_dtype.floating
        ):
            raise ValueError("affine coefficients must be floating point")

    @property
    def has_bias(self) -> bool:
        return self.bias_dtype is not None


@dataclass(frozen=True, slots=True)
class HierarchicalCoefficients:
    supergroup: int
    local_scale_bits: int
    local_scale_interpretation: CodeInterpretation
    super_scale_dtype: DType
    local_scale_zero_point: int = 0
    local_bias_bits: int | None = None
    super_bias_dtype: DType | None = None
    bias_sign: int = 0

    def __post_init__(self) -> None:
        if self.supergroup <= 0 or self.local_scale_bits not in (6, 8):
            raise ValueError("invalid hierarchical coefficient geometry")
        if not self.super_scale_dtype.floating:
            raise ValueError("super scale must be floating point")
        if self.local_scale_interpretation == CodeInterpretation.OFFSET_BINARY:
            if not 0 < self.local_scale_zero_point < 1 << self.local_scale_bits:
                raise ValueError("offset-binary local scales need a zero point")
        elif self.local_scale_zero_point:
            raise ValueError("only offset-binary local scales carry a zero point")
        has_bias = self.local_bias_bits is not None
        if has_bias != (self.super_bias_dtype is not None):
            raise ValueError("local and super bias must appear together")
        if has_bias and (self.local_bias_bits not in (6, 8) or self.bias_sign not in (-1, 1)):
            raise ValueError("invalid hierarchical bias")
        if not has_bias and self.bias_sign:
            raise ValueError("bias sign requires bias coefficients")

    @property
    def has_bias(self) -> bool:
        return self.local_bias_bits is not None


type CoefficientScheme = DirectCoefficients | HierarchicalCoefficients


@dataclass(frozen=True, slots=True)
class Affine:
    code: Code
    group: int
    coefficients: CoefficientScheme

    def __post_init__(self) -> None:
        if self.group <= 0:
            raise ValueError("affine group must be positive")
        if isinstance(self.coefficients, HierarchicalCoefficients):
            if self.coefficients.supergroup % self.group:
                raise ValueError("affine groups must tile their supergroup")


@dataclass(frozen=True, slots=True)
class Codebook:
    code_bits: int
    table: tuple[int, ...]
    group: int
    coefficients: CoefficientScheme

    def __post_init__(self) -> None:
        if self.code_bits not in (2, 4, 8) or len(self.table) != 1 << self.code_bits:
            raise ValueError("codebook must define every code")
        if self.group <= 0 or any(not -128 <= value <= 127 for value in self.table):
            raise ValueError("invalid codebook representation")


type Representation = Dense | Affine | Codebook


@dataclass(frozen=True, slots=True)
class CanonicalLayout:
    """Byte geometry of one canonical encoded allocation.

    Direct coefficients occupy allocation-wide planes. Hierarchical
    coefficients remain interleaved with each supergroup so a consuming
    subgroup can fetch one complete quantization tile locally.
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


def _whole_bytes(bits: int) -> int:
    if bits < 0 or bits % 8:
        raise ValueError("canonical encoded fields must occupy whole bytes")
    return bits // 8


def canonical_layout(
    representation: Affine | Codebook, elements: int
) -> CanonicalLayout:
    """Return the sole physical layout consumed by Magnitensor schedules."""
    if elements <= 0 or elements % representation.group:
        raise ValueError("encoded storage requires complete quantization groups")
    coefficients = representation.coefficients
    low_bits = (
        representation.code.low_bits
        if isinstance(representation, Affine)
        else representation.code_bits
    )
    high_bits = representation.code.high_bits if isinstance(representation, Affine) else 0

    if isinstance(coefficients, DirectCoefficients):
        groups = elements // representation.group
        low = 0
        high = _whole_bytes(elements * low_bits) if high_bits else None
        scales = _whole_bytes(elements * (low_bits + high_bits))
        biases = scales + groups * coefficients.scale_dtype.itemsize if coefficients.has_bias else None
        end = scales + groups * coefficients.scale_dtype.itemsize
        if coefficients.bias_dtype is not None:
            end += groups * coefficients.bias_dtype.itemsize
        return CanonicalLayout(
            elements,
            elements,
            end,
            low,
            high,
            scales,
            biases,
            None,
            None,
            end,
            False,
        )

    tile_elements = coefficients.supergroup
    if elements % tile_elements:
        raise ValueError("hierarchical storage requires complete supergroups")
    tile_groups = tile_elements // representation.group
    low = 0
    high = _whole_bytes(tile_elements * low_bits) if high_bits else None
    scales = _whole_bytes(tile_elements * (low_bits + high_bits))
    local_scale_bytes = _whole_bytes(tile_groups * coefficients.local_scale_bits)
    biases = scales + local_scale_bytes if coefficients.has_bias else None
    end = scales + local_scale_bytes
    if coefficients.local_bias_bits is not None:
        end += _whole_bytes(tile_groups * coefficients.local_bias_bits)
    super_scale = end
    end += coefficients.super_scale_dtype.itemsize
    super_bias = end if coefficients.super_bias_dtype is not None else None
    if coefficients.super_bias_dtype is not None:
        end += coefficients.super_bias_dtype.itemsize
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
        elements // tile_elements * end,
        True,
    )


def represented_nbytes(representation: Representation, elements: int) -> int:
    if elements <= 0:
        raise ValueError("element count must be positive")
    if isinstance(representation, Dense):
        return elements * representation.dtype.itemsize
    return canonical_layout(representation, elements).nbytes
