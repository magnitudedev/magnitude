"""What a kernel reads. Containers are forgotten once a weight is resident.

Flat affine planes and compact hierarchical affine blocks are distinct resident
meanings. A container may already store either one; kernels specialize on the
representation parameters and never on that container's wire names.

Planar storage is one allocation: the low plane, then the scale plane, then the
bias plane if present, then the high plane if present. ``plane_offsets`` is the
single definition of where those start, shared by the repack kernel that writes
them, the upload that places MLX tensors at the same offsets, and every kernel
that reads them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from magnitude_engine.platform.execution import DType

__all__ = [
    "BlockCodec",
    "EncodedBlocks",
    "Dense",
    "HierarchicalAffine",
    "HierarchyPacking",
    "PlanarAffine",
    "PlaneOffsets",
    "Representation",
    "plane_offsets",
    "resident_bytes",
]


@dataclass(frozen=True)
class Dense:
    dtype: DType


class BlockCodec(StrEnum):
    """Container-neutral byte interpretations without affine hierarchy."""

    F16 = "f16"
    GROUPED_I8 = "grouped_i8"
    CODEBOOK_I4 = "codebook_i4"


@dataclass(frozen=True)
class EncodedBlocks:
    codec: BlockCodec
    block_elements: int
    block_bytes: int

    def __post_init__(self):
        if self.block_elements <= 0 or self.block_bytes <= 0:
            raise ValueError("encoded block geometry must be positive")

    @property
    def bits_per_value(self) -> float:
        return self.block_bytes * 8 / self.block_elements


class HierarchyPacking(StrEnum):
    """Physical organizations of hierarchical affine coefficients and codes."""

    SCALE_MIN_I6 = "scale_min_i6"
    SIGNED_SCALE_I8 = "signed_scale_i8"


@dataclass(frozen=True)
class HierarchicalAffine:
    """Compact two-level affine blocks consumed without persistent expansion."""

    bits: int
    high_bits: int
    group: int
    supergroup: int
    local_bits: int
    signed: bool
    has_bias: bool
    packing: HierarchyPacking

    def __post_init__(self):
        if self.bits != 4 or self.high_bits not in (0, 1, 2):
            raise ValueError("hierarchical affine storage carries a 4-bit low code")
        if self.group <= 0 or self.supergroup <= 0 or self.supergroup % self.group:
            raise ValueError("hierarchical affine groups must tile their supergroup")
        if self.local_bits not in (6, 8):
            raise ValueError("hierarchical affine coefficients are six or eight bit")
        if self.packing == HierarchyPacking.SCALE_MIN_I6:
            if self.local_bits != 6 or self.signed or not self.has_bias or self.group != 32:
                raise ValueError("scale/min packing requires unsigned 32-value groups")
        elif self.packing == HierarchyPacking.SIGNED_SCALE_I8:
            if self.local_bits != 8 or not self.signed or self.has_bias or self.group != 16:
                raise ValueError("signed-scale packing requires signed 16-value groups")

    @property
    def block_elements(self) -> int:
        return self.supergroup

    @property
    def block_bytes(self) -> int:
        payload = self.supergroup * (self.bits + self.high_bits) // 8
        groups = self.supergroup // self.group
        local = groups * self.local_bits * (2 if self.has_bias else 1) // 8
        super_coefficients = 2 * (2 if self.has_bias else 1)
        return payload + local + super_coefficients

    @property
    def bits_per_value(self) -> float:
        return self.block_bytes * 8 / self.block_elements


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


type Representation = Dense | EncodedBlocks | HierarchicalAffine | PlanarAffine


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
    if not isinstance(representation, (EncodedBlocks, HierarchicalAffine)):
        raise TypeError("unknown resident weight representation")
    if elements % representation.block_elements:
        raise ValueError("blocked storage requires complete container blocks")
    return elements // representation.block_elements * representation.block_bytes
