"""Physical encoded operand layouts, independent of artifact/model identity."""

from dataclasses import dataclass
from enum import StrEnum

from magnitude_engine.artifacts.gguf import Encoding


class EncodedLayout(StrEnum):
    GGUF = "gguf"
    AFFINE_PLANES = "affine_planes"


@dataclass(frozen=True)
class AffineGeometry:
    group_elements: int
    high_bits: int
    zero_point: int


def affine_geometry(encoding: Encoding) -> AffineGeometry:
    if encoding == Encoding.Q4_K:
        return AffineGeometry(32, 0, 0)
    if encoding == Encoding.Q5_K:
        return AffineGeometry(32, 1, 0)
    if encoding == Encoding.Q6_K:
        return AffineGeometry(16, 2, 32)
    raise ValueError("affine planes require Q4_K, Q5_K or Q6_K")


def storage_bytes(elements: int, encoding: Encoding, layout: EncodedLayout) -> int:
    if elements <= 0 or elements % encoding.block_elements:
        raise ValueError("encoded storage requires complete artifact blocks")
    if layout == EncodedLayout.AFFINE_PLANES:
        geometry = affine_geometry(encoding)
        # Low nibbles plus two metadata bits/value, followed by any high bits.
        # Q6_K stores one scale/16 values; Q4_K/Q5_K store scale+bias/32 values.
        return elements * (6 + geometry.high_bits) // 8
    if layout != EncodedLayout.GGUF:
        raise ValueError("unknown encoded operand layout")
    return elements // encoding.block_elements * encoding.block_bytes
