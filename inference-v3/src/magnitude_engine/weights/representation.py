"""What a kernel reads. Containers are forgotten once a weight is resident.

A Q4_K matrix repacked into planes and an MLX affine matrix uploaded from three
tensors are both ``PlanarAffine``; they differ only in the parameters below, and
one kernel family covers both by specializing on them.

Planar storage is one allocation: the low plane, then the scale plane, then the
bias plane if present, then the high plane if present. ``plane_offsets`` is the
single definition of where those start, shared by the repack kernel that writes
them, the upload that places MLX tensors at the same offsets, and every kernel
that reads them.
"""

from __future__ import annotations

from dataclasses import dataclass

from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.formats.gguf import Encoding

__all__ = [
    "Blocked",
    "Dense",
    "Encoding",
    "PlanarAffine",
    "PlaneOffsets",
    "Representation",
    "plane_offsets",
    "resident_bytes",
]


@dataclass(frozen=True)
class Dense:
    dtype: DType


@dataclass(frozen=True)
class Blocked:
    encoding: Encoding


@dataclass(frozen=True)
class PlanarAffine:
    bits: int
    """Low-plane bits per value."""

    high_bits: int
    """Extra plane bits per value: 0, 1 or 2."""

    group: int
    """Values sharing one scale."""

    coefficient_dtype: DType
    """F32 for repacked K-quants, BF16 for MLX."""

    signed: bool
    """The zero point is ``2 ** (bits + high_bits - 1)`` rather than zero."""

    has_bias: bool
    """A bias plane follows the scales."""

    def __post_init__(self):
        if self.bits != 4 or self.high_bits not in (0, 1, 2):
            raise ValueError("planar affine storage carries a 4-bit low plane")
        if self.group <= 0 or self.group % 8:
            raise ValueError("an affine group must cover whole low-plane words")
        if self.coefficient_dtype not in (DType.F32, DType.BF16):
            raise ValueError("affine coefficients are stored as F32 or BF16")
        if self.signed and self.has_bias:
            raise ValueError("a signed affine representation has no bias plane")

    @property
    def zero_point(self) -> int:
        return (1 << (self.bits + self.high_bits - 1)) if self.signed else 0

    @property
    def bits_per_value(self) -> int:
        """Payload and coefficient bits amortized over one value."""
        coefficients = self.coefficient_dtype.itemsize * 8 * (2 if self.has_bias else 1)
        return self.bits + self.high_bits + coefficients // self.group


type Representation = Dense | Blocked | PlanarAffine


@dataclass(frozen=True)
class PlaneOffsets:
    """Word offsets of each plane within one ``uint32`` allocation."""

    low: int
    scales: int
    biases: int | None
    high: int | None
    words: int


def plane_offsets(representation: PlanarAffine, elements: int) -> PlaneOffsets:
    if elements <= 0 or elements % representation.group:
        raise ValueError("planar affine storage requires complete groups")
    groups = elements // representation.group
    coefficient_words = groups * representation.coefficient_dtype.itemsize // 4
    if groups * representation.coefficient_dtype.itemsize % 4:
        raise ValueError("affine coefficient planes must fill whole words")
    low = 0
    scales = elements * representation.bits // 32
    biases = scales + coefficient_words if representation.has_bias else None
    end = scales + coefficient_words * (2 if representation.has_bias else 1)
    high = end if representation.high_bits else None
    if representation.high_bits:
        end += elements * representation.high_bits // 32
    return PlaneOffsets(low=low, scales=scales, biases=biases, high=high, words=end)


def resident_bytes(representation: Representation, elements: int) -> int:
    if isinstance(representation, PlanarAffine):
        return plane_offsets(representation, elements).words * 4
    if isinstance(representation, Dense):
        return elements * representation.dtype.itemsize
    encoding = representation.encoding
    if elements % encoding.block_elements:
        raise ValueError("blocked storage requires complete container blocks")
    return elements // encoding.block_elements * encoding.block_bytes
