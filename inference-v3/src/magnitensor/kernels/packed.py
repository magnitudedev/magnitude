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
    dot_packet: int
    matrix_packet: int
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
        # GEMV amortizes one affine coefficient over sixteen codes per lane.
        # Matrix kernels instead distribute eight-code words across the whole
        # workgroup; coupling those two granularities halves cooperative load
        # parallelism for every prefill projection.
        return PacketFormat("mlx-q4-group64", 16, 8, 512)
    if (
        value.code.bits == 8
        and value.code.interpretation == CodeInterpretation.TWOS_COMPLEMENT
        and isinstance(coefficients, DirectCoefficients)
        and coefficients.scale_dtype == DType.F16
        and coefficients.bias_dtype is None
        and value.group == 32
    ):
        return PacketFormat("gguf-q8-0", 8, 8, 256)
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
        return PacketFormat("gguf-q5-k" if value.code.high_bits else "gguf-q4-k", 8, 8, 256)
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
        return PacketFormat("gguf-q6-k", 8, 8, 256)
    return None


@T.macro
def byte(words, offset):
    packed = words[offset // 4]
    return (packed >> ((offset % 4) * 8)) & T.uint32(255)


@T.macro
def word(words, offset):
    index = offset // 4
    shift = (offset % 4) * 8
    # Most production packets are word aligned. Q6_K's 210-byte tile makes
    # alternating tiles cross a word boundary, requiring one adjacent load.
    packed = T.alloc_local((1,), "uint32")
    if shift == 0:
        packed[0] = words[index]
    else:
        packed[0] = (words[index] >> shift) | (words[index + 1] << (32 - shift))
    return packed[0]


@T.macro
def halfword(words, offset):
    index = offset // 4
    shift = (offset % 4) * 8
    packed = T.alloc_local((1,), "uint32")
    packed[0] = words[index] >> shift
    if shift > 16:
        packed[0] |= words[index + 1] << (32 - shift)
    return packed[0] & T.uint32(65535)


@T.macro
def half(words, offset):
    bits = T.cast(halfword(words, offset), "uint16")
    return T.cast(T.reinterpret(bits, "float16"), "float32")


@T.macro
def bfloat(words, offset):
    bits = T.cast(halfword(words, offset), "uint16")
    # BF16 is its bit-identical FP32 prefix. Reconstruct directly without
    # depending on a target's support for native BF16 reinterpretation.
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
    first = row * columns + chunk * packet.tile + lane * packet.dot_packet
    dot = T.alloc_local((1,), "float32")
    total = T.alloc_local((1,), "float32")
    dot[0] = 0.0
    total[0] = 0.0
    coefficients = representation.coefficients

    if packet.name == "mlx-q4-group64":
        low = word(words, layout.low + first // 2)
        high = word(words, layout.low + first // 2 + 4)
        for item in T.unroll(4, explicit=True):
            packed_word = low if item < 2 else high
            value_word = (packed_word >> ((item % 2) * 16)) & T.uint32(65535)
            for offset in T.unroll(4, explicit=True):
                index = item * 4 + offset
                code = (value_word >> (offset * 4)) & T.uint32(15)
                value = T.cast(values[index], "float32")
                dot[0] += value * T.cast(code, "float32")
                total[0] += value
        group = first // representation.group
        scale = bfloat(words, layout.scales + group * 2)
        assert layout.biases is not None
        bias = bfloat(words, layout.biases + group * 2)
        return dot[0] * scale + total[0] * bias

    if packet.name == "gguf-q8-0":
        low = word(words, layout.low + first)
        high = word(words, layout.low + first + 4)
        for offset in T.unroll(8, explicit=True):
            packed_word = low if offset < 4 else high
            shift = offset * 8 if offset < 4 else (offset - 4) * 8
            code = T.cast((packed_word >> shift) & T.uint32(255), "uint8")
            dot[0] += T.cast(values[offset], "float32") * T.cast(T.cast(code, "int8"), "float32")
        scale = half(words, layout.scales + (first // 32) * 2)
        return dot[0] * scale

    assert isinstance(coefficients, HierarchicalCoefficients)
    tile = first // layout.tile_elements
    within = first % layout.tile_elements
    base = tile * layout.tile_bytes
    low = word(words, base + layout.low + within // 2)
    shift = (within % 8) * 4
    assert not representation.code.high_bits or layout.high is not None
    if representation.code.high_bits == 1:
        high = byte(words, base + layout.high + within // 8)
    elif representation.code.high_bits == 2:
        high = word(words, base + layout.high + within // 4)
    else:
        high = T.uint32(0)
    for offset in T.unroll(8, explicit=True):
        low_code = (low >> (shift + offset * 4)) & T.uint32(15)
        if representation.code.high_bits == 1:
            code = low_code | (((high >> offset) & T.uint32(1)) << 4)
        elif representation.code.high_bits == 2:
            code = low_code | (((high >> (offset * 2)) & T.uint32(3)) << 4)
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
        return (
            dot[0]
            * T.cast(T.cast(local, "int8"), "float32")
            * half(words, base + layout.super_scale)
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
def group_coefficients(words, spec, element):
    representation = spec.representation
    assert isinstance(representation, Affine)
    layout = canonical_layout(representation, spec.elements)
    packet = packet_format(spec)
    assert packet is not None
    if packet.name == "mlx-q4-group64":
        group = element // representation.group
        assert layout.biases is not None
        scale = bfloat(words, layout.scales + group * 2)
        bias = bfloat(words, layout.biases + group * 2)
    elif packet.name == "gguf-q8-0":
        scale = half(words, layout.scales + (element // 32) * 2)
        bias = T.float32(0)
    else:
        base = element // layout.tile_elements * layout.tile_bytes
        group = element % layout.tile_elements // representation.group
        assert layout.super_scale is not None
        if packet.name == "gguf-q6-k":
            local = T.cast(byte(words, base + layout.scales + group), "uint8")
            scale = half(words, base + layout.super_scale) * T.cast(
                T.cast(local, "int8"), "float32"
            )
            bias = T.float32(0)
        else:
            assert layout.biases is not None and layout.super_bias is not None
            scale = half(words, base + layout.super_scale) * T.cast(
                field(words, base + layout.scales, 6, group), "float32"
            )
            bias = -half(words, base + layout.super_bias) * T.cast(
                field(words, base + layout.biases, 6, group), "float32"
            )
    return scale, bias


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
    scale, bias = group_coefficients(words, spec, element)

    if packet.name == "mlx-q4-group64":
        low = word(words, layout.low + element // 2)
        for index in T.unroll(8, explicit=True):
            code = (low >> (index * 4)) & T.uint32(15)
            destination[tile_row, tile_column + index] = T.cast(code, "float32") * scale + bias
    elif packet.name == "gguf-q8-0":
        low = word(words, layout.low + element)
        high = word(words, layout.low + element + 4)
        for index in T.unroll(8, explicit=True):
            packed = low if index < 4 else high
            code = T.cast((packed >> ((index % 4) * 8)) & T.uint32(255), "uint8")
            destination[tile_row, tile_column + index] = (
                T.cast(T.cast(code, "int8"), "float32") * scale
            )
    else:
        assert isinstance(coefficients, HierarchicalCoefficients)
        tile = element // layout.tile_elements
        within = element % layout.tile_elements
        base = tile * layout.tile_bytes
        low = word(words, base + layout.low + within // 2)
        assert not representation.code.high_bits or layout.high is not None
        if representation.code.high_bits == 1:
            high = byte(words, base + layout.high + within // 8)
        elif representation.code.high_bits == 2:
            high = word(words, base + layout.high + within // 4)
        else:
            high = T.uint32(0)
        for index in T.unroll(8, explicit=True):
            low_code = (low >> (index * 4)) & T.uint32(15)
            if representation.code.high_bits == 1:
                code = low_code | (((high >> index) & T.uint32(1)) << 4)
            elif representation.code.high_bits == 2:
                code = low_code | (((high >> (index * 2)) & T.uint32(3)) << 4)
            else:
                code = low_code
            interpreted = (
                T.cast(code, "int32") - representation.code.zero_point
                if representation.code.interpretation == CodeInterpretation.OFFSET_BINARY
                else T.cast(code, "float32")
            )
            destination[tile_row, tile_column + index] = (
                T.cast(interpreted, "float32") * scale + bias
            )


@T.macro
def load_matrix_tile(
    destination,
    words,
    spec,
    first_row,
    first_column,
    rows,
    columns,
    bn,
    bk,
    threads,
    destination_row_stride=1,
    destination_row_offset=0,
):
    """One writer per packet, including tails; never clear then overwrite."""
    packet = packet_format(spec)
    assert packet is not None
    packets = bn * bk // packet.matrix_packet
    for iteration in T.serial(T.ceildiv(packets, threads)):
        linear = iteration * threads + T.get_thread_binding()
        row = linear // (bk // packet.matrix_packet)
        column = linear % (bk // packet.matrix_packet) * packet.matrix_packet
        if row < bn:
            if first_row + row < rows and first_column + column < columns:
                decode_packet(
                    destination,
                    row * destination_row_stride + destination_row_offset,
                    column,
                    words,
                    spec,
                    first_row + row,
                    first_column + column,
                )
            else:
                for item in T.unroll(packet.matrix_packet, explicit=True):
                    destination[
                        row * destination_row_stride + destination_row_offset, column + item
                    ] = 0
