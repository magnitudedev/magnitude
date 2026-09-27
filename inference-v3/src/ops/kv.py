"""Persistent attention geometry, codec identity and atomic plane layout.

Logical state contains concatenated key/value vectors. Physical planes belong to
one resource: sharing, copies, completion pins and reclamation cover them together.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from .tensor.types import DType


@dataclass(frozen=True, slots=True)
class AttentionGeometry:
    query_heads: int
    kv_heads: int
    key_width: int
    value_width: int

    def __post_init__(self):
        if min(self.query_heads, self.kv_heads, self.key_width, self.value_width) <= 0:
            raise ValueError("attention dimensions must be positive")
        if self.query_heads % self.kv_heads:
            raise ValueError("query heads must form complete KV-head groups")


@dataclass(frozen=True, slots=True)
class AttentionSemantics:
    scale: float
    causal: bool = True

    def __post_init__(self):
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("attention scale must be finite and positive")


@dataclass(frozen=True, slots=True)
class DenseKVCodec:
    dtype: DType

    def __post_init__(self):
        if not self.dtype.floating:
            raise ValueError("dense KV requires a floating dtype")


@dataclass(frozen=True, slots=True)
class AffineKVCodec:
    bits: int
    group_size: int = 0
    scale_dtype: DType = DType.F16

    def __post_init__(self):
        if self.bits not in (4, 8) or self.group_size < 0:
            raise ValueError("affine KV requires four/eight bits and a valid group size")
        if self.scale_dtype not in (DType.F16, DType.F32):
            raise ValueError("KV affine metadata requires FP16 or FP32")


@dataclass(frozen=True, slots=True)
class RotatedLloydMax:
    bits: int = 4
    norm_dtype: DType = DType.F16
    sign_seed: int = 42
    transform_version: int = 1
    codebook_version: int = 1

    def __post_init__(self):
        if self.bits != 4 or self.norm_dtype not in (DType.F16, DType.F32):
            raise ValueError("rotated KV requires four-bit codes and floating norms")
        if self.transform_version != 1 or self.codebook_version != 1:
            raise ValueError("unknown persistent rotation/codebook convention")
        if not 0 <= self.sign_seed < 2**32:
            raise ValueError("rotation seed must be an unsigned 32-bit integer")


type KVCodec = DenseKVCodec | AffineKVCodec | RotatedLloydMax


@dataclass(frozen=True, slots=True)
class KVPlane:
    name: str
    offset: int
    nbytes: int
    dtype: DType
    row_elements: int


@dataclass(frozen=True, slots=True)
class KVRepresentation:
    key_width: int
    value_width: int
    key: KVCodec
    value: KVCodec
    packing_version: int = 1

    def __post_init__(self):
        if min(self.key_width, self.value_width) <= 0 or self.packing_version not in (1, 2):
            raise ValueError("invalid KV geometry or packing version")
        for codec, width in ((self.key, self.key_width), (self.value, self.value_width)):
            if isinstance(codec, RotatedLloydMax) and (width < 32 or width & (width - 1)):
                raise ValueError("signed WHT requires a power-of-two width of at least 32")
            if isinstance(codec, AffineKVCodec) and codec.group_size and width % codec.group_size:
                raise ValueError("affine groups must cover complete KV vectors")
        if isinstance(self.value, RotatedLloydMax):
            raise ValueError("rotated value codecs require an explicit output-basis realization")

    @property
    def digest(self) -> str:
        # Codec tags matter: equal-shaped planes need not have equal semantics.
        record = asdict(self)
        record['key_kind'] = type(self.key).__name__
        record['value_kind'] = type(self.value).__name__
        return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    @property
    def logical_width(self) -> int:
        return self.key_width + self.value_width

    def planes(self, vectors: int) -> tuple[KVPlane, ...]:
        if vectors <= 0:
            raise ValueError("KV allocation requires at least one vector")
        result = []
        cursor = 0
        for prefix, codec, width in (("key", self.key, self.key_width),
                                     ("value", self.value, self.value_width)):
            if isinstance(codec, DenseKVCodec):
                descriptions = (("dense", codec.dtype, width),)
            else:
                # Every packed vector begins at 16-byte alignment. Metadata is
                # independently aligned; fetching codes never fetches metadata.
                words = ((width * codec.bits + 127) // 128) * 4
                descriptions = (("codes", DType.U32, words),)
                if isinstance(codec, AffineKVCodec):
                    groups = width // (codec.group_size or width)
                    descriptions += (("scale", codec.scale_dtype, groups),
                                     ("zero", codec.scale_dtype, groups))
                else:
                    descriptions += (("norm", codec.norm_dtype, 1),)
            for name, dtype, row_elements in descriptions:
                cursor = (cursor + 15) // 16 * 16
                nbytes = vectors * row_elements * dtype.itemsize
                result.append(KVPlane(f"{prefix}.{name}", cursor, nbytes, dtype, row_elements))
                cursor += nbytes
        return tuple(result)

    def storage_nbytes(self, elements: int) -> int:
        if elements % self.logical_width:
            raise ValueError("KV logical storage must contain complete key/value pairs")
        last = self.planes(elements // self.logical_width)[-1]
        return (last.offset + last.nbytes + 15) // 16 * 16


def dense_kv(key_width: int, value_width: int, dtype: DType) -> KVRepresentation:
    return KVRepresentation(key_width, value_width, DenseKVCodec(dtype), DenseKVCodec(dtype))


def affine_k8_uniform_v4(key_width: int, value_width: int) -> KVRepresentation:
    return KVRepresentation(key_width, value_width, AffineKVCodec(8), AffineKVCodec(4), packing_version=2)


def rotated_k4_uniform_v4(key_width: int, value_width: int) -> KVRepresentation:
    return KVRepresentation(key_width, value_width, RotatedLloydMax(), AffineKVCodec(4), packing_version=2)


def default_kv_representation(key_width: int, value_width: int) -> KVRepresentation:
    """Minimum-work compact tier; callers can select maximum compression."""
    return affine_k8_uniform_v4(key_width, value_width)


def kv_state_spec(capacity: int, heads: int, dtype: DType, representation: KVRepresentation):
    from .tensor.types import TensorSpec

    if capacity <= 0 or heads <= 0 or not dtype.floating:
        raise ValueError("invalid logical KV state geometry")
    return TensorSpec((capacity, heads, representation.logical_width), dtype,
                      representation=representation)
