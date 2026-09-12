"""Packet-native access to the packed representations used by production kernels.

Every helper in this module consumes one complete quantization packet.  There is
deliberately no scalar logical-element decoder: production schedules must expose
the packet geometry in their work decomposition so codes and coefficients are
loaded once and reused by the dot product.
"""

from __future__ import annotations

from dataclasses import dataclass

import tilelang.language as T

from ..representations import (
    Affine,
    CodeInterpretation,
    DirectCoefficients,
    HierarchicalCoefficients,
    canonical_layout,
)
from ..tensor.types import DType, TensorSpec


@dataclass(frozen=True, slots=True)
class PacketFormat:
    name: str
    packet: int
    tile: int


def packet_format(spec: TensorSpec) -> PacketFormat | None:
    """Classify only representations with a production packet schedule."""
    value = spec.representation
    if not isinstance(value, Affine):
        return None
    coefficients = value.coefficients
    if (
        value.code.low_bits == 4
        and value.code.high_bits == 0
        and value.code.interpretation == CodeInterpretation.UNSIGNED
        and isinstance(coefficients, DirectCoefficients)
        and coefficients.scale_dtype == DType.BF16
        and coefficients.bias_dtype == DType.BF16
        and value.group == 64
    ):
        return PacketFormat("mlx-q4-group64", 16, 512)
    if (
        value.code.bits == 8
        and value.code.interpretation == CodeInterpretation.TWOS_COMPLEMENT
        and isinstance(coefficients, DirectCoefficients)
        and coefficients.scale_dtype == DType.F16
        and coefficients.bias_dtype is None
        and value.group == 32
    ):
        return PacketFormat("gguf-q8-0", 8, 256)
    if (
        value.code.low_bits == 4
        and value.code.high_bits in (0, 1)
        and value.code.interpretation == CodeInterpretation.UNSIGNED
        and isinstance(coefficients, HierarchicalCoefficients)
        and value.group == 32
        and coefficients.supergroup == 256
        and coefficients.local_scale_bits == 6
        and coefficients.local_scale_interpretation == CodeInterpretation.UNSIGNED
        and coefficients.local_bias_bits == 6
        and coefficients.super_scale_dtype == DType.F16
        and coefficients.super_bias_dtype == DType.F16
        and coefficients.bias_sign == -1
    ):
        return PacketFormat("gguf-q5-k" if value.code.high_bits else "gguf-q4-k", 8, 256)
    if (
        value.code.low_bits == 4
        and value.code.high_bits == 2
        and value.code.interpretation == CodeInterpretation.OFFSET_BINARY
        and value.code.zero_point == 32
        and isinstance(coefficients, HierarchicalCoefficients)
        and value.group == 16
        and coefficients.supergroup == 256
        and coefficients.local_scale_bits == 8
        and coefficients.local_scale_interpretation == CodeInterpretation.TWOS_COMPLEMENT
        and not coefficients.has_bias
        and coefficients.super_scale_dtype == DType.F16
    ):
        return PacketFormat("gguf-q6-k", 8, 256)
    return None


@T.macro
def byte(words, offset):
    word = T.cast(words[offset // 4], "uint32")
    return (word >> ((offset % 4) * 8)) & T.uint32(255)


@T.macro
def half(words, offset):
    bits = T.cast(byte(words, offset) | (byte(words, offset + 1) << 8), "uint16")
    return T.cast(T.reinterpret(bits, "float16"), "float32")


@T.macro
def bfloat(words, offset):
    bits = T.cast(byte(words, offset) | (byte(words, offset + 1) << 8), "uint16")
    # BF16 is its bit-identical FP32 prefix.  Reconstruct FP32 directly so the
    # generated program does not require a native bfloat type (Metal has none).
    return T.reinterpret(T.cast(bits, "uint32") << 16, "float32")


@T.macro
def field(words, offset, bits, index):
    bit = index * bits
    packed = byte(words, offset + bit // 8) | (byte(words, offset + bit // 8 + 1) << 8)
    return (packed >> (bit % 8)) & T.uint32((1 << bits) - 1)


@T.macro
def packet_dot(values, words, spec, row, chunk, lane):
    """Dot one lane's packet and return its affine contribution."""
    representation = spec.representation
    assert isinstance(representation, Affine)
    layout = canonical_layout(representation, spec.elements)
    columns = spec.shape[-1]
    packet = packet_format(spec)
    assert packet is not None
    first = row * columns + chunk * packet.tile + lane * packet.packet
    dot = T.alloc_local((1,), "float32")
    total = T.alloc_local((1,), "float32")
    dot[0] = 0.0
    total[0] = 0.0
    coefficients = representation.coefficients

    if packet.name == "mlx-q4-group64":
        low = T.cast(words[(layout.low + first // 2) // 4], "uint32")
        high = T.cast(words[(layout.low + first // 2) // 4 + 1], "uint32")
        for item in T.unroll(4, explicit=True):
            packed = low if item < 2 else high
            word = (packed >> ((item % 2) * 16)) & T.uint32(65535)
            for offset in T.unroll(4, explicit=True):
                index = item * 4 + offset
                code = (word >> (offset * 4)) & T.uint32(15)
                value = T.cast(values[index], "float32")
                dot[0] += value * T.cast(code, "float32")
                total[0] += value
        group = first // representation.group
        scale = bfloat(words, layout.scales + group * 2)
        assert layout.biases is not None
        bias = bfloat(words, layout.biases + group * 2)
        return dot[0] * scale + total[0] * bias

    if packet.name == "gguf-q8-0":
        low = T.cast(words[(layout.low + first) // 4], "uint32")
        high = T.cast(words[(layout.low + first) // 4 + 1], "uint32")
        for offset in T.unroll(8, explicit=True):
            word = low if offset < 4 else high
            shift = offset * 8 if offset < 4 else (offset - 4) * 8
            code = T.cast((word >> shift) & T.uint32(255), "uint8")
            dot[0] += T.cast(values[offset], "float32") * T.cast(
                T.cast(code, "int8"), "float32"
            )
        scale = half(words, layout.scales + (first // 32) * 2)
        return dot[0] * scale

    assert isinstance(coefficients, HierarchicalCoefficients)
    tile = first // layout.tile_elements
    within = first % layout.tile_elements
    base = tile * layout.tile_bytes
    low = T.cast(words[(base + layout.low + within // 2) // 4], "uint32")
    shift = (within % 8) * 4
    assert not representation.code.high_bits or layout.high is not None
    if representation.code.high_bits == 1:
        high = byte(words, base + layout.high + within // 8)
    elif representation.code.high_bits == 2:
        high = field(words, base + layout.high, 2, within)
    else:
        high = T.uint32(0)
    for offset in T.unroll(8, explicit=True):
        low_code = (low >> (shift + offset * 4)) & T.uint32(15)
        if representation.code.high_bits == 1:
            code = low_code | (((high >> offset) & T.uint32(1)) << 4)
        elif representation.code.high_bits == 2:
            code = low_code | (
                field(words, base + layout.high, 2, within + offset) << 4
            )
        else:
            code = low_code
        value = T.cast(values[offset], "float32")
        interpreted = (
            T.cast(code, "int32") - representation.code.zero_point
            if representation.code.interpretation == CodeInterpretation.OFFSET_BINARY
            else T.cast(code, "float32")
        )
        dot[0] += value * T.cast(interpreted, "float32")
        total[0] += value
    group = within // representation.group
    if packet.name == "gguf-q6-k":
        local = T.cast(byte(words, base + layout.scales + group), "uint8")
        assert layout.super_scale is not None
        return dot[0] * T.cast(T.cast(local, "int8"), "float32") * half(
            words, base + layout.super_scale
        )
    local_scale = T.cast(field(words, base + layout.scales, 6, group), "float32")
    assert layout.biases is not None
    local_bias = T.cast(field(words, base + layout.biases, 6, group), "float32")
    assert layout.super_scale is not None and layout.super_bias is not None
    return (
        half(words, base + layout.super_scale) * local_scale * dot[0]
        - half(words, base + layout.super_bias) * local_bias * total[0]
    )


@T.macro
def decode_packet(destination, tile_row, tile_column, words, spec, row, first):
    """Decode one naturally aligned packet into a shared matrix tile."""
    representation = spec.representation
    assert isinstance(representation, Affine)
    layout = canonical_layout(representation, spec.elements)
    packet = packet_format(spec)
    assert packet is not None
    element = row * spec.shape[-1] + first
    coefficients = representation.coefficients

    if packet.name == "mlx-q4-group64":
        low = T.cast(words[(layout.low + element // 2) // 4], "uint32")
        high = T.cast(words[(layout.low + element // 2) // 4 + 1], "uint32")
        group = element // representation.group
        scale = bfloat(words, layout.scales + group * 2)
        assert layout.biases is not None
        bias = bfloat(words, layout.biases + group * 2)
        for index in T.unroll(16, explicit=True):
            packed = low if index < 8 else high
            code = (packed >> ((index % 8) * 4)) & T.uint32(15)
            destination[tile_row, tile_column + index] = T.cast(code, "float32") * scale + bias
    elif packet.name == "gguf-q8-0":
        low = T.cast(words[(layout.low + element) // 4], "uint32")
        high = T.cast(words[(layout.low + element) // 4 + 1], "uint32")
        scale = half(words, layout.scales + (element // 32) * 2)
        for index in T.unroll(8, explicit=True):
            packed = low if index < 4 else high
            code = T.cast((packed >> ((index % 4) * 8)) & T.uint32(255), "uint8")
            destination[tile_row, tile_column + index] = T.cast(
                T.cast(code, "int8"), "float32"
            ) * scale
    else:
        assert isinstance(coefficients, HierarchicalCoefficients)
        tile = element // layout.tile_elements
        within = element % layout.tile_elements
        base = tile * layout.tile_bytes
        low = T.cast(words[(base + layout.low + within // 2) // 4], "uint32")
        assert not representation.code.high_bits or layout.high is not None
        if representation.code.high_bits == 1:
            high = byte(words, base + layout.high + within // 8)
        else:
            high = T.uint32(0)
        group = within // representation.group
        assert layout.super_scale is not None
        if packet.name == "gguf-q6-k":
            local = T.cast(byte(words, base + layout.scales + group), "uint8")
            scale = half(words, base + layout.super_scale) * T.cast(
                T.cast(local, "int8"), "float32"
            )
            bias = T.cast(0, "float32")
        else:
            scale = half(words, base + layout.super_scale) * T.cast(
                field(words, base + layout.scales, 6, group), "float32"
            )
            assert layout.biases is not None and layout.super_bias is not None
            bias = -half(words, base + layout.super_bias) * T.cast(
                field(words, base + layout.biases, 6, group), "float32"
            )
        for index in T.unroll(8, explicit=True):
            low_code = (low >> (index * 4)) & T.uint32(15)
            if representation.code.high_bits == 1:
                code = low_code | (((high >> index) & T.uint32(1)) << 4)
            elif representation.code.high_bits == 2:
                code = low_code | (
                    field(words, base + layout.high, 2, within + index) << 4
                )
            else:
                code = low_code
            interpreted = (
                T.cast(code, "int32") - representation.code.zero_point
                if representation.code.interpretation == CodeInterpretation.OFFSET_BINARY
                else T.cast(code, "float32")
            )
            destination[tile_row, tile_column + index] = T.cast(
                interpreted, "float32"
            ) * scale + bias


def require_packet(spec: TensorSpec) -> PacketFormat:
    result = packet_format(spec)
    if result is None:
        raise ValueError(f"no optimized packed schedule for {spec.representation!r}")
    return result
