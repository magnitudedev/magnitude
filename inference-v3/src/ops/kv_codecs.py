"""Versioned codec constants and independent logical references.

NumPy is used only to construct specialization constants and evaluate reference
values. Production persistence and attention execute in TileLang kernels.
"""
from __future__ import annotations

import math
from functools import lru_cache

from .kv import AffineKVCodec, DenseKVCodec, RotatedLloydMax


@lru_cache(maxsize=None)
def rotation_signs(width: int, seed: int) -> tuple[int, ...]:
    """Portable uint32 avalanche convention, independent of host RNG versions."""
    signs = []
    for channel in range(width):
        word = (seed ^ channel ^ 0x9E3779B9) & 0xFFFFFFFF
        word = ((word ^ (word >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
        word = ((word ^ (word >> 15)) * 0x846CA68B) & 0xFFFFFFFF
        word ^= word >> 16
        signs.append(1 if word & 1 else -1)
    return tuple(signs)


@lru_cache(maxsize=None)
def lloyd_max_centroids(width: int) -> tuple[float, ...]:
    """16-level quantizer for sqrt(width) times a unit-sphere coordinate.

    Integrate the finite-dimensional sphere density, rather than substituting
    a normal distribution for narrow heads. Symmetry is imposed explicitly.
    The grid, iteration count and convergence rule belong to codebook version 1.
    """
    import numpy as np

    if width < 32 or width & (width - 1):
        raise ValueError("Lloyd-Max KV requires a power-of-two width >= 32")
    edge = math.sqrt(width)
    grid = np.linspace(-edge, edge, 65537, dtype=np.float64)
    density = np.maximum(0, 1 - grid * grid / width) ** ((width - 3) / 2)
    cumulative = np.cumsum(density)
    centers = np.interp((np.arange(16) + 0.5) / 16, cumulative / cumulative[-1], grid)
    for _ in range(512):
        bins = np.searchsorted((centers[:-1] + centers[1:]) / 2, grid)
        mass = np.bincount(bins, weights=density, minlength=16)
        moment = np.bincount(bins, weights=density * grid, minlength=16)
        updated = moment / mass
        updated = (updated - updated[::-1]) / 2
        delta = np.max(np.abs(updated - centers))
        centers = updated
        if delta < 1e-10:
            break
    return tuple(float(value) for value in centers.astype(np.float32))


def rotate_reference(values, seed: int, *, inverse: bool = False):
    import numpy as np

    result = np.array(values, dtype=np.float32, copy=True)
    width = result.shape[-1]
    signs = np.asarray(rotation_signs(width, seed), dtype=np.float32)
    if not inverse:
        result *= signs
    step = 1
    while step < width:
        view = result.reshape(*result.shape[:-1], -1, 2 * step)
        left = view[..., :step].copy()
        right = view[..., step:].copy()
        view[..., :step] = left + right
        view[..., step:] = left - right
        step *= 2
    result *= np.float32(width ** -0.5)
    if inverse:
        result *= signs
    return result


def quantize_reference(values, codec):
    """Return codes and metadata using the same published metadata precision."""
    import numpy as np

    x = np.asarray(values, dtype=np.float32)
    if isinstance(codec, DenseKVCodec):
        from .tensor.primitive import round_reference
        return (round_reference(x, codec.dtype),)
    if isinstance(codec, AffineKVCodec):
        groups = x.reshape(*x.shape[:-1], -1, codec.group_size or x.shape[-1])
        zero = groups.min(axis=-1).astype(codec.scale_dtype.value)
        scale = ((groups.max(axis=-1) - groups.min(axis=-1)) / ((1 << codec.bits) - 1)).astype(codec.scale_dtype.value)
        divisor = np.where(scale > 0, scale, 1).astype(np.float32)
        codes = np.clip(np.rint((groups - zero.astype(np.float32)[..., None]) / divisor[..., None]),
                        0, (1 << codec.bits) - 1).astype(np.uint8)
        return codes.reshape(x.shape), scale, zero
    if isinstance(codec, RotatedLloydMax):
        norm = np.sqrt(np.sum(x * x, axis=-1, dtype=np.float32))
        unit = x / np.where(norm > 0, norm, 1)[..., None]
        projected = rotate_reference(unit, codec.sign_seed) * np.float32(math.sqrt(x.shape[-1]))
        centers = np.asarray(lloyd_max_centroids(x.shape[-1]), dtype=np.float32)
        codes = np.searchsorted((centers[:-1] + centers[1:]) * np.float32(0.5), projected).astype(np.uint8)
        return codes, norm.astype(codec.norm_dtype.value)
    raise TypeError("unknown KV codec")


def dequantize_reference(encoded, codec):
    import numpy as np

    if isinstance(codec, DenseKVCodec):
        return np.asarray(encoded[0], dtype=np.float32)
    if isinstance(codec, AffineKVCodec):
        codes, scale, zero = encoded
        grouped = codes.reshape(*codes.shape[:-1], -1, codec.group_size or codes.shape[-1])
        result = grouped.astype(np.float32) * scale.astype(np.float32)[..., None] + zero.astype(np.float32)[..., None]
        return result.reshape(codes.shape)
    if isinstance(codec, RotatedLloydMax):
        codes, norm = encoded
        centers = np.asarray(lloyd_max_centroids(codes.shape[-1]), dtype=np.float32)
        projected = centers[codes] * (norm.astype(np.float32) / np.float32(math.sqrt(codes.shape[-1])))[..., None]
        return rotate_reference(projected, codec.sign_seed, inverse=True)
    raise TypeError("unknown KV codec")


def encode_kv_reference(logical, representation) -> bytes:
    import numpy as np

    logical = np.asarray(logical, dtype=np.float32)
    vectors = logical.size // representation.logical_width
    planes = {p.name: p for p in representation.planes(vectors)}
    output = bytearray(representation.storage_nbytes(logical.size))
    for prefix, codec, values in (
        ("key", representation.key, logical[..., :representation.key_width]),
        ("value", representation.value, logical[..., representation.key_width:]),
    ):
        encoded = quantize_reference(values, codec)
        if isinstance(codec, DenseKVCodec):
            from .tensor.types import DType
            dense = np.ascontiguousarray(encoded[0]).reshape(vectors, -1)
            if codec.dtype == DType.BF16:
                dense = (dense.view(np.uint32) >> 16).astype(np.uint16)
            entries = (("dense", dense),)
        else:
            codes = encoded[0].reshape(vectors, -1).astype(np.uint32)
            per_word = 32 // codec.bits
            packed = np.zeros((vectors, planes[prefix + ".codes"].row_elements), dtype=np.uint32)
            for channel in range(codes.shape[1]):
                packed[:, channel // per_word] |= codes[:, channel] << ((channel % per_word) * codec.bits)
            if representation.packing_version == 2:
                heads = logical.shape[-2]
                packed = packed.reshape(vectors // heads, heads, -1, 4).transpose(1, 2, 0, 3).copy()
            entries = (("codes", packed),)
            entries += (("scale", encoded[1]), ("zero", encoded[2])) if isinstance(codec, AffineKVCodec) else (("norm", encoded[1]),)
        for suffix, value in entries:
            plane = planes[prefix + "." + suffix]
            raw = np.ascontiguousarray(value).tobytes()
            if len(raw) != plane.nbytes:
                raise ValueError("codec plane byte extent disagrees with representation")
            output[plane.offset:plane.offset + plane.nbytes] = raw
    return bytes(output)


def decode_kv_reference(content: bytes, spec):
    import numpy as np
    from .tensor.types import DType

    representation = spec.representation
    vectors = spec.elements // representation.logical_width
    if len(content) != spec.storage_nbytes:
        raise ValueError("KV byte extent disagrees with representation")
    planes = {p.name: p for p in representation.planes(vectors)}

    def plane(prefix, suffix):
        descriptor = planes[prefix + "." + suffix]
        dtype = np.uint16 if descriptor.dtype == DType.BF16 else descriptor.dtype.value
        value = np.frombuffer(content, dtype=dtype, count=descriptor.nbytes // descriptor.dtype.itemsize,
                              offset=descriptor.offset).reshape(vectors, descriptor.row_elements)
        if descriptor.dtype == DType.BF16:
            value = (value.astype(np.uint32) << 16).view(np.float32)
        return value

    outputs = []
    for prefix, codec, width in (("key", representation.key, representation.key_width),
                                 ("value", representation.value, representation.value_width)):
        if isinstance(codec, DenseKVCodec):
            encoded = (plane(prefix, "dense"),)
        else:
            packed = plane(prefix, "codes")
            if representation.packing_version == 2:
                packed = packed.reshape(spec.shape[1], -1, spec.shape[0], 4).transpose(2, 0, 1, 3).reshape(vectors, -1)
            channel = np.arange(width)
            codes = ((packed[:, channel // (32 // codec.bits)] >> ((channel % (32 // codec.bits)) * codec.bits))
                     & ((1 << codec.bits) - 1)).astype(np.uint8)
            encoded = (codes, plane(prefix, "scale"), plane(prefix, "zero")) if isinstance(codec, AffineKVCodec) else (codes, plane(prefix, "norm")[:, 0])
        outputs.append(dequantize_reference(encoded, codec))
    return np.concatenate(outputs, axis=-1).reshape(spec.shape)
