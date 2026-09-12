"""Canonical encoded-weight loads expressed with portable TileLang operations."""

from __future__ import annotations

import math
from typing import Any

import tilelang.language as T

from ..representations import (
    Affine,
    Code,
    Codebook,
    CodeInterpretation,
    DirectCoefficients,
    HierarchicalCoefficients,
)
from ..tensor.types import DType, TensorSpec


def represented_load(storage: Any, spec: TensorSpec, index: Any) -> Any:
    """Return one logical value from Magnitensor's canonical encoded byte layout."""
    representation = spec.representation
    if isinstance(representation, Affine):
        low_bytes = math.ceil(spec.elements * representation.code.low_bits / 8)
        high_bytes = math.ceil(spec.elements * representation.code.high_bits / 8)
        code_bytes = low_bytes + high_bytes
        raw: Any = _packed(storage, 0, index, representation.code.low_bits)
        if representation.code.high_bits:
            raw += (
                _packed(storage, low_bytes, index, representation.code.high_bits)
                << representation.code.low_bits
            )
        value = _interpret(raw, representation.code)
        scale, bias = _coefficients(
            storage,
            code_bytes,
            spec.elements,
            representation.group,
            representation.coefficients,
            index,
        )
        return T.cast(value, "float32") * scale + bias
    if isinstance(representation, Codebook):
        code_bytes = math.ceil(spec.elements * representation.code_bits / 8)
        raw = _packed(storage, 0, index, representation.code_bits)
        value: Any = T.cast(representation.table[-1], "float32")
        for code, entry in reversed(tuple(enumerate(representation.table[:-1]))):
            value = T.if_then_else(raw == code, T.cast(entry, "float32"), value)
        scale, bias = _coefficients(
            storage,
            code_bytes,
            spec.elements,
            representation.group,
            representation.coefficients,
            index,
        )
        return value * scale + bias
    raise TypeError("represented_load requires an encoded tensor")


def _packed(storage: Any, base: int, index: Any, bits: int) -> Any:
    start = index * bits
    value: Any = T.cast(0, "uint16")
    for position in range(bits):
        bit = start + position
        value |= (T.cast(storage[base + bit // 8], "uint16") >> (bit % 8) & 1) << position
    return value


def _interpret(raw: Any, code: Code) -> Any:
    if code.interpretation == CodeInterpretation.UNSIGNED:
        return raw
    if code.interpretation == CodeInterpretation.OFFSET_BINARY:
        return T.cast(raw, "int32") - code.zero_point
    sign = 1 << (code.bits - 1)
    return T.if_then_else(raw >= sign, T.cast(raw, "int32") - (1 << code.bits), raw)


def _coefficients(
    storage: Any, base: int, elements: int, group: int, coefficients: Any, index: Any
) -> tuple[Any, Any]:
    groups = math.ceil(elements / group)
    group_index = index // group
    if isinstance(coefficients, DirectCoefficients):
        scale = _float_at(storage, base, group_index, coefficients.scale_dtype)
        if coefficients.bias_dtype is None:
            return T.cast(scale, "float32"), T.cast(0, "float32")
        bias_base = base + groups * coefficients.scale_dtype.itemsize
        bias = _float_at(storage, bias_base, group_index, coefficients.bias_dtype)
        return T.cast(scale, "float32"), T.cast(bias, "float32")
    if not isinstance(coefficients, HierarchicalCoefficients):
        raise TypeError("unknown coefficient scheme")
    local_scale_bytes = math.ceil(groups * coefficients.local_scale_bits / 8)
    local_scale = _packed(storage, base, group_index, coefficients.local_scale_bits)
    local_scale = _interpret_bits(
        local_scale,
        coefficients.local_scale_bits,
        coefficients.local_scale_interpretation,
        coefficients.local_scale_zero_point,
    )
    cursor = base + local_scale_bytes
    local_bias = None
    if coefficients.local_bias_bits is not None:
        local_bias_bytes = math.ceil(groups * coefficients.local_bias_bits / 8)
        local_bias = _packed(storage, cursor, group_index, coefficients.local_bias_bits)
        cursor += local_bias_bytes
    groups_per_supergroup = coefficients.supergroup // group
    supergroup = group_index // groups_per_supergroup
    supergroups = math.ceil(elements / coefficients.supergroup)
    super_scale = _float_at(storage, cursor, supergroup, coefficients.super_scale_dtype)
    scale = T.cast(local_scale, "float32") * T.cast(super_scale, "float32")
    if local_bias is None or coefficients.super_bias_dtype is None:
        return scale, T.cast(0, "float32")
    super_bias_base = cursor + supergroups * coefficients.super_scale_dtype.itemsize
    super_bias = _float_at(storage, super_bias_base, supergroup, coefficients.super_bias_dtype)
    bias = coefficients.bias_sign * T.cast(local_bias, "float32") * T.cast(super_bias, "float32")
    return scale, bias


def _interpret_bits(raw: Any, bits: int, interpretation: Any, zero_point: int) -> Any:
    if interpretation == CodeInterpretation.UNSIGNED:
        return raw
    if interpretation == CodeInterpretation.OFFSET_BINARY:
        return T.cast(raw, "int32") - zero_point
    sign = 1 << (bits - 1)
    return T.if_then_else(raw >= sign, T.cast(raw, "int32") - (1 << bits), raw)


def _float_at(storage: Any, base: int, index: Any, dtype: DType) -> Any:
    offset = base + index * dtype.itemsize
    if dtype in (DType.F16, DType.BF16):
        bits = T.cast(storage[offset], "uint16")
        bits |= T.cast(storage[offset + 1], "uint16") << 8
        return T.reinterpret(T.cast(bits, "uint16"), dtype.value)
    if dtype == DType.F32:
        bits = T.cast(storage[offset], "uint32")
        bits |= T.cast(storage[offset + 1], "uint32") << 8
        bits |= T.cast(storage[offset + 2], "uint32") << 16
        bits |= T.cast(storage[offset + 3], "uint32") << 24
        return T.reinterpret(T.cast(bits, "uint32"), "float32")
    raise TypeError(f"unsupported coefficient dtype {dtype}")
