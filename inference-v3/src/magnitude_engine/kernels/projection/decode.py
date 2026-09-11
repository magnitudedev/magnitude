"""Shared scalar and cooperative readers for canonical resident layouts."""

import tilelang.language as T

from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import (
    Affine,
    Codebook,
    CodeInterpretation,
    Dense,
    DirectCoefficients,
    Representation,
    canonical_layout,
)


def _reader(representation: Representation, elements: int, *, words: bool):
    layout = (
        None if isinstance(representation, Dense) else canonical_layout(representation, elements)
    )

    @T.macro
    def byte(data, offset):
        if words:
            return (data[offset // 2].astype("uint32") >> ((offset % 2) * 8)) & 255
        return data[offset].astype("uint32")

    @T.macro
    def unsigned(data, offset, size):
        value = T.alloc_local((1,), "uint32")
        value[0] = 0
        for j in T.unroll(size, explicit=True):
            value[0] |= byte(data, offset + j) << (8 * j)
        return value[0]

    @T.macro
    def floating(data, offset, dtype):
        bits = unsigned(data, offset, dtype.itemsize)
        if dtype == DType.F16:
            return T.reinterpret(bits.astype("uint16"), "float16").astype("float32")
        if dtype == DType.BF16:
            return T.reinterpret(bits.astype("uint16"), "bfloat16").astype("float32")
        return T.reinterpret(bits, "float32")

    @T.macro
    def packed(data, offset, index, bits):
        bit = index * bits
        raw = byte(data, offset + bit // 8) | (byte(data, offset + bit // 8 + 1) << 8)
        return (raw >> (bit % 8)) & ((1 << bits) - 1)

    @T.macro
    def interpreted(value, bits, interpretation, zero_point=0):
        if interpretation == CodeInterpretation.OFFSET_BINARY:
            return (value.astype("int32") - zero_point).astype("float32")
        if interpretation == CodeInterpretation.TWOS_COMPLEMENT:
            if bits != 8:
                raise ValueError("two's-complement canonical values are eight bit")
            return value.astype("uint8").astype("int8").astype("float32")
        return value.astype("float32")

    @T.macro
    def raw_code(data, index):
        assert layout is not None
        if layout.hierarchical:
            tile = index // layout.tile_elements
            local = index % layout.tile_elements
            base = tile * layout.tile_bytes
        else:
            local = index
            base = 0
        if isinstance(representation, Affine):
            code = representation.code
            low = packed(data, base + layout.low, local, code.low_bits)
            if code.high_bits:
                high = packed(data, base + layout.high, local, code.high_bits)
                low |= high << code.low_bits
            return low
        return packed(data, base + layout.low, local, representation.code_bits)

    def codebook_value(code, table):
        table_words = tuple(
            sum((table[first + offset] & 255) << (offset * 8) for offset in range(4))
            for first in range(0, len(table), 4)
        )
        word_index = code // 4
        word = T.uint32(table_words[-1])
        for index in reversed(range(len(table_words) - 1)):
            word = T.if_then_else(word_index == index, T.uint32(table_words[index]), word)
        return (
            ((word >> ((code % 4) * 8)) & 255)
            .astype("uint8")
            .astype("int8")
            .astype("float32")
        )

    @T.macro
    def parameters(data, index):
        scale = T.alloc_local((1,), "float32")
        bias = T.alloc_local((1,), "float32")
        scale[0], bias[0] = 1, 0
        if isinstance(representation, Dense):
            return scale[0], bias[0]
        assert layout is not None
        coefficients = representation.coefficients
        if isinstance(coefficients, DirectCoefficients):
            group = index // representation.group
            scale[0] = floating(
                data,
                layout.scales + group * coefficients.scale_dtype.itemsize,
                coefficients.scale_dtype,
            )
            if coefficients.bias_dtype is not None:
                bias[0] = floating(
                    data,
                    layout.biases + group * coefficients.bias_dtype.itemsize,
                    coefficients.bias_dtype,
                )
        else:
            tile = index // coefficients.supergroup
            local = index % coefficients.supergroup
            group = local // representation.group
            base = tile * layout.tile_bytes
            local_scale = packed(
                data,
                base + layout.scales,
                group,
                coefficients.local_scale_bits,
            )
            local_scale_value = interpreted(
                local_scale,
                coefficients.local_scale_bits,
                coefficients.local_scale_interpretation,
                coefficients.local_scale_zero_point,
            )
            scale[0] = (
                floating(
                    data,
                    base + layout.super_scale,
                    coefficients.super_scale_dtype,
                )
                * local_scale_value
            )
            if coefficients.local_bias_bits is not None:
                local_bias = packed(
                    data,
                    base + layout.biases,
                    group,
                    coefficients.local_bias_bits,
                )
                bias[0] = (
                    coefficients.bias_sign
                    * floating(
                        data,
                        base + layout.super_bias,
                        coefficients.super_bias_dtype,
                    )
                    * local_bias.astype("float32")
                )
        return scale[0], bias[0]

    @T.macro
    def payload(data, index):
        if isinstance(representation, Dense):
            if representation.dtype != DType.F32:
                raise ValueError("dense inference parameters are resident as FP32")
            return floating(data, index * 4, DType.F32)
        raw = raw_code(data, index)
        if isinstance(representation, Codebook):
            return codebook_value(raw, representation.table)
        return interpreted(
            raw,
            representation.code.bits,
            representation.code.interpretation,
            representation.code.zero_point,
        )

    return parameters, payload


def decoder(representation: Representation, elements: int):
    """Scalar access to one coordinate of a canonical resident weight."""
    parameters, payload = _reader(representation, elements, words=False)

    @T.macro
    def decode(data, index):
        scale, bias = parameters(data, index)
        return scale * payload(data, index) + bias

    return decode


def interpretation(representation: Representation, elements: int):
    """Cooperative access over a uint16 view of the same canonical bytes."""
    return _reader(representation, elements, words=True)
