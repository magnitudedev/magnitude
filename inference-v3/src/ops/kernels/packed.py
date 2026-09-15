"""Packet-native access to the packed representations used by production kernels.

Every helper in this module consumes one complete quantization packet.  There is
deliberately no scalar logical-element decoder: production schedules must expose
the packet geometry in their work decomposition so codes and coefficients are
loaded once and reused by the dot product.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import cast

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
    if (isinstance(coefficients, HierarchicalCoefficients)
            and coefficients.supergroup == 256
            and coefficients.local_scale_bits == 8
            and coefficients.super_scale_dtype == DType.F32
            and coefficients.local_scale_interpretation in
                (CodeInterpretation.UNSIGNED, CodeInterpretation.TWOS_COMPLEMENT)
            and (not coefficients.has_bias or
                 (coefficients.local_bias_bits == 8 and coefficients.super_bias_dtype == DType.F32))
            and value.code.low_bits == 4 and value.code.high_bits in (0, 1, 2)
            and value.code.interpretation in
                (CodeInterpretation.UNSIGNED, CodeInterpretation.OFFSET_BINARY)
            and value.group in (16, 32)):
        return PacketFormat("affine-factored", 8, 8, 256)
    if (isinstance(coefficients, DirectCoefficients)
            and coefficients.scale_dtype == DType.F32
            and coefficients.bias_dtype in (None, DType.F32)
            and value.code.low_bits == 4 and value.code.high_bits in (0, 1, 2)
            and value.code.interpretation in (CodeInterpretation.UNSIGNED, CodeInterpretation.OFFSET_BINARY)
            and value.group in (16, 32)):
        return PacketFormat("affine-direct-grouped", 8, 8, 256)
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


def affine_group_width(reduction: int, *specs: TensorSpec) -> int:
    assert specs and all(isinstance(spec.representation, Affine) for spec in specs)
    return min(reduction, math.gcd(*(cast(Affine, spec.representation).group for spec in specs)))


def affine_shared_bytes(rows: int, columns: int, reduction: int, dtype: DType,
                        *specs: TensorSpec) -> int:
    """Native activations, exact F16 codes and original FP32 coefficient pairs."""
    group = affine_group_width(reduction, *specs)
    return (rows * reduction * dtype.itemsize + columns * reduction * DType.F16.itemsize
            + columns * (reduction // group) * 2 * DType.F32.itemsize)


def affine_code_center(spec: TensorSpec) -> float:
    representation = spec.representation
    assert isinstance(representation, Affine)
    # Unsigned min/max encodings put their numerical zero near the midpoint.
    # Half-integer centered codes remain exact in F16/BF16 and avoid subtracting
    # a large nearly-equal bias contribution after the matrix contraction.
    return ((2**representation.code.bits - 1) / 2
            if representation.code.interpretation == CodeInterpretation.UNSIGNED else 0.0)


@T.macro
def affine_storage(bm, bn, bk, dtype, reduction_step, specs):
    left = T.alloc_shared((bm, bk), dtype)
    codes = T.alloc_shared((bn, bk), "float16")
    group = affine_group_width(bk, *specs)
    coefficients = T.alloc_shared((bn, bk // group, 2), "float32")
    accum = T.alloc_fragment((bm, bn), "float32")
    b = T.alloc_fragment((bn, bk), dtype)
    return left, codes, coefficients, accum, b


@T.macro
def affine_gemm(storage, bm, bn, bk, valid_m):
    """Reconstruct immediate operands; keep one FP32 accumulator across all K."""
    left, codes, coefficients, accum, b = storage
    group = bk // coefficients.shape[1]
    T.sync_threads()
    for j, k in T.Parallel(bn, bk):
        b[j, k] = T.cast(
            T.cast(codes[j, k], "float32") * coefficients[j, k // group, 0]
            + coefficients[j, k // group, 1],
            left.dtype,
        )
    # Keep dequantization in the portable IR and let the target select its
    # native matrix instruction for the complete reduction tile.
    with T.attr(0, "pragma_auto_unroll_max_step", 4096):
        with T.attr(0, "pragma_unroll_explicit", 1):
            T.gemm(left, b, accum, transpose_B=True, valid_m=valid_m,
                   policy=T.GemmWarpPolicy.Square)
    T.sync_threads()


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
def prepare_packet_activation(values, dtype, packet):
    """Share affine activation work across independent output rows.

    Masked nibbles and reciprocal powers of two have the same exact product.
    Keep the original path when rescaling would create an FP32 subnormal.
    Other packet formats retain their unmodified activation representation.
    """
    if packet.name in ("affine-direct-grouped", "affine-factored"):
        total = T.alloc_local((1,), "float32")
        total[0] = 0.0
        for item in T.unroll(packet.dot_packet, explicit=True):
            total[0] += values[item]
        return total[0], False
    if packet.name == "mlx-q4-group64":
        total = T.alloc_local((1,), "float32")
        normal = T.alloc_local((1,), "bool")
        total[0] = 0.0
        normal[0] = True
        for item in T.unroll(packet.dot_packet, explicit=True):
            total[0] += values[item]
            if dtype != "float16":
                normal[0] = normal[0] and (values[item] == 0 or T.abs(values[item]) >= 2**-114)
        if normal[0]:
            for group in T.unroll(4, explicit=True):
                values[group * 4 + 1] *= 1 / 16
                values[group * 4 + 2] *= 1 / 256
                values[group * 4 + 3] *= 1 / 4096
        return total[0], normal[0]
    else:
        return None, None


@T.macro
def _affine_nibble_dot(values, low, high, masked):
    dot = T.alloc_local((1,), "float32")
    dot[0] = 0.0
    for item in T.unroll(4, explicit=True):
        packed_word = low if item < 2 else high
        value_word = (packed_word >> ((item % 2) * 16)) & T.uint32(65535)
        for offset in T.unroll(4, explicit=True):
            if masked:
                code = value_word & (T.uint32(15) << (offset * 4))
            else:
                code = (value_word >> (offset * 4)) & T.uint32(15)
            dot[0] += T.cast(values[item * 4 + offset], "float32") * T.cast(code, "float32")
    return dot[0]


@T.macro
def packet_dot(values, words, spec, row, chunk, lane, prepared_sum=None, prepared_mask=None):
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
        if prepared_sum is not None:
            total[0] = prepared_sum
            if prepared_mask:
                dot[0] = _affine_nibble_dot(values, low, high, True)
            else:
                dot[0] = _affine_nibble_dot(values, low, high, False)
        else:
            dot[0] = _affine_nibble_dot(values, low, high, False)
            for item in T.unroll(16, explicit=True):
                total[0] += T.cast(values[item], "float32")
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

    if packet.name in ("affine-direct-grouped", "affine-factored"):
        decoded = T.alloc_local((1, 8), "float32")
        decode_packet(decoded, 0, 0, words, spec, row, chunk * packet.tile + lane * packet.dot_packet, False)
        for offset in T.unroll(8, explicit=True):
            dot[0] += T.cast(values[offset], "float32") * decoded[0, offset]
            if prepared_sum is None:
                total[0] += T.cast(values[offset], "float32")
        scale, bias = group_coefficients(words, spec, first)
        activation_sum = total[0] if prepared_sum is None else prepared_sum
        return dot[0] * scale + activation_sum * (bias + scale * affine_code_center(spec))

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
    if packet.name == "affine-direct-grouped":
        group = element // representation.group
        scale = T.reinterpret(words[layout.scales // 4 + group], "float32")
        bias = (T.reinterpret(words[layout.biases // 4 + group], "float32")
                if layout.biases is not None else T.float32(0))
    elif packet.name == "affine-factored":
        base = element // layout.tile_elements * layout.tile_bytes
        group = element % layout.tile_elements // representation.group
        assert layout.super_scale is not None
        local = T.cast(byte(words, base + layout.scales + group), "uint8")
        if representation.coefficients.local_scale_interpretation == CodeInterpretation.TWOS_COMPLEMENT:
            factor = T.cast(T.cast(local, "int8"), "float32")
        else:
            factor = T.cast(local, "float32")
        scale = T.reinterpret(words[(base + layout.super_scale) // 4], "float32") * factor
        if layout.biases is not None:
            assert layout.super_bias is not None
            bias = (representation.coefficients.bias_sign
                    * T.reinterpret(words[(base + layout.super_bias) // 4], "float32")
                    * T.cast(byte(words, base + layout.biases + group), "float32"))
        else:
            bias = T.float32(0)
    elif packet.name == "mlx-q4-group64":
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
def decode_packet(destination, tile_row, tile_column, words, spec, row, first,
                  apply_coefficients=True, center_codes=True):
    """Shared packet interpretation for value publication and raw-code contraction."""
    representation = spec.representation
    assert isinstance(representation, Affine)
    layout = canonical_layout(representation, spec.elements)
    packet = packet_format(spec)
    assert packet is not None
    element = row * spec.shape[-1] + first
    coefficients = representation.coefficients
    if apply_coefficients:
        scale_value, bias_value = group_coefficients(words, spec, element)
        scale = T.alloc_var("float32", init=scale_value)
        bias = T.alloc_var("float32", init=bias_value)
    else:
        scale, bias = T.float32(1), T.float32(-affine_code_center(spec) if center_codes else 0)

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
        within = element % layout.tile_elements if layout.hierarchical else element
        base = element // layout.tile_elements * layout.tile_bytes if layout.hierarchical else 0
        low = word(words, base + layout.low + within // 2)
        assert not representation.code.high_bits or layout.high is not None
        if representation.code.high_bits == 1:
            high = byte(words, base + layout.high + within // 8)
        elif representation.code.high_bits == 2:
            # Eight two-bit fields occupy exactly one aligned halfword. Loading
            # a full unaligned word would fetch an unused neighbour and branch
            # on alternating lanes just to discard its upper sixteen bits.
            high = halfword(words, base + layout.high + within // 4)
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
    coefficients,
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
    """One writer per code packet and per coefficient pair, including tails."""
    packet = packet_format(spec)
    assert packet is not None
    assert isinstance(spec.representation, Affine)
    group = bk // coefficients.shape[1]
    assert spec.representation.group % group == 0
    for index, coefficient in T.Parallel(bn, coefficients.shape[1]):
        destination_row = index * destination_row_stride + destination_row_offset
        if first_row + index < rows and first_column + coefficient * group < columns:
            scale, bias = group_coefficients(words, spec, (first_row + index) * columns + first_column + coefficient * group)
            coefficients[destination_row, coefficient, 0] = scale
            coefficients[destination_row, coefficient, 1] = bias
        else:
            coefficients[destination_row, coefficient, 0] = 0
            coefficients[destination_row, coefficient, 1] = 0
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
                    False,
                    False,
                )
            else:
                for item in T.unroll(packet.matrix_packet, explicit=True):
                    destination[
                        row * destination_row_stride + destination_row_offset, column + item
                    ] = 0
