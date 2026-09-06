"""Validate and locate affine embedding rows across tensor shards."""

from __future__ import annotations

from dataclasses import dataclass

from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.artifacts.tensors import DTYPE_BYTES, TensorRegion


@dataclass(frozen=True)
class AffineRowTable:
    shards: tuple[tuple[TensorRegion, TensorRegion, TensorRegion], ...]
    encoding: AffineEncoding

    def __post_init__(self) -> None:
        if not self.shards:
            raise ValueError("embedding table requires at least one shard")
        geometry = None
        for weight, scale, bias in self.shards:
            records = weight, scale, bias
            if (
                any(len(r.shape) != 2 or r.shape[0] <= 0 for r in records)
                or len({r.shape[0] for r in records}) != 1
            ):
                raise ValueError("embedding components require matching nonempty row axes")
            actual = tuple((r.shape[1], r.dtype) for r in records)
            if geometry is not None and geometry != actual:
                raise ValueError("embedding shards disagree on row geometry")
            geometry = actual
            if (
                weight.dtype != "U32"
                or scale.dtype not in ("F16", "BF16", "F32")
                or bias.dtype != scale.dtype
                or bias.shape != scale.shape
                or weight.shape[1] * 32 // self.encoding.bits
                != scale.shape[1] * self.encoding.group_size
            ):
                raise ValueError("invalid affine weight/scale/bias geometry")

    @property
    def rows(self) -> int:
        return sum(shard[0].shape[0] for shard in self.shards)

    @property
    def width(self) -> int:
        return self.shards[0][0].shape[1] * 32 // self.encoding.bits

    @property
    def row_bytes(self) -> int:
        return sum(r.shape[1] * DTYPE_BYTES[r.dtype] for r in self.shards[0])
