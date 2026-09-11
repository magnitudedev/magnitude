"""Reading one stored value, given only the representation it is resident in.

These macros are expanded inside consumers; nothing here allocates a
dequantized weight. ``decoder`` is scalar random access over bytes;
``interpretation`` is the cooperative form, where a worker owns an aligned
eight-coordinate group and fetches that group's affine parameters once.
"""

import tilelang.language as T

from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import (
    Blocked,
    Dense,
    Encoding,
    PlanarAffine,
    Representation,
    plane_offsets,
)


def _blocks(representation: Representation) -> Encoding | None:
    """The container encoding a blocked schedule reads, or None for planes.

    A dense FP32 weight is read by the same path as an FP32 container block:
    one reinterpretation of four bytes, with no group parameters.
    """
    if isinstance(representation, Blocked):
        return representation.encoding
    if isinstance(representation, Dense):
        if representation.dtype != DType.F32:
            raise ValueError("dense resident weights are read as FP32")
        return Encoding.F32
    if not isinstance(representation, PlanarAffine):
        raise TypeError("unknown resident weight representation")
    return None


@T.macro
def half(data, offset):
    # A typed temporary prevents C integer promotion from changing the operand
    # width at reinterpretation in the Metal code generator.
    bits = T.alloc_local((1,), "uint16")
    bits[0] = data[offset].astype("uint16") | (data[offset + 1].astype("uint16") << 8)
    return T.reinterpret(bits[0], "float16").astype("float32")


def decoder(representation: Representation, elements: int = 0):
    """Scalar access to one coordinate of a resident weight."""
    encoding = _blocks(representation)
    planes = None
    group_elements, high_bits, zero_point = 32, 0, 0
    if encoding is None:
        assert isinstance(representation, PlanarAffine)
        if representation.coefficient_dtype != DType.F32:
            raise ValueError("scalar plane access requires FP32 group coefficients")
        planes = plane_offsets(representation, elements)
        assert planes.low == 0  # the low plane is the origin of the allocation
        group_elements = representation.group
        high_bits = representation.high_bits
        zero_point = representation.zero_point

    @T.macro
    def decode(data, index):
        result = T.alloc_local((1,), "float32")
        if planes is not None:
            scale_offset = planes.scales * 4 + index // group_elements * 4
            bias_offset = (0 if planes.biases is None else planes.biases * 4) + (
                index // group_elements * 4
            )
            bits = T.alloc_local((2,), "uint32")
            quant = T.alloc_local((1,), "uint32")
            T.clear(bits)
            for byte_index in T.unroll(4, explicit=True):
                bits[0] |= data[scale_offset + byte_index].astype("uint32") << (byte_index * 8)
                if zero_point == 0:
                    bits[1] |= data[bias_offset + byte_index].astype("uint32") << (byte_index * 8)
            quant[0] = (data[index // 2].astype("uint32") >> ((index % 2) * 4)) & 15
            if high_bits:
                high = (
                    data[planes.high * 4 + index * high_bits // 8].astype("uint32")
                    >> ((index * high_bits) % 8)
                ) & ((1 << high_bits) - 1)
                quant[0] |= high << 4
            result[0] = T.reinterpret(bits[0], "float32") * (
                quant[0].astype("int32") - zero_point
            ).astype("float32") + T.reinterpret(bits[1], "float32")
        elif encoding == Encoding.F32:
            offset = index * 4
            bits = (
                data[offset].astype("uint32")
                | (data[offset + 1].astype("uint32") << 8)
                | (data[offset + 2].astype("uint32") << 16)
                | (data[offset + 3].astype("uint32") << 24)
            )
            result[0] = T.reinterpret(bits, "float32")
        elif encoding == Encoding.F16:
            result[0] = half(data, index * 2)
        elif encoding == Encoding.Q8_0:
            base = index // 32 * 34
            result[0] = half(data, base) * data[base + 2 + index % 32].astype("int8").astype(
                "float32"
            )
        elif encoding == Encoding.Q4_K or encoding == Encoding.Q5_K:
            base = index // 256 * encoding.block_bytes
            k = index % 256
            group = k // 32
            low = data[base + 4 + group % 4].astype("int32")
            minimum = data[base + 8 + group % 4].astype("int32")
            high = data[base + 12 + group % 4].astype("int32")
            scale = T.if_then_else(group < 4, low & 63, (high & 15) | ((low >> 6) << 4))
            bias = T.if_then_else(group < 4, minimum & 63, (high >> 4) | ((minimum >> 6) << 4))
            payload = 16 if encoding == Encoding.Q4_K else 48
            low_code = (
                data[base + payload + k // 64 * 32 + k % 32].astype("int32") >> (k % 64 // 32 * 4)
            ) & 15
            code = T.alloc_local((1,), "int32")
            code[0] = low_code
            if encoding == Encoding.Q5_K:
                code[0] |= ((data[base + 16 + k % 32].astype("int32") >> group) & 1) << 4
            result[0] = half(data, base) * scale.astype("float32") * code[0].astype(
                "float32"
            ) - half(data, base + 2) * bias.astype("float32")
        elif encoding == Encoding.Q6_K:
            base = index // 256 * 210
            k = index % 256
            low = (data[base + k // 128 * 64 + k % 64].astype("int32") >> (k % 128 // 64 * 4)) & 15
            high = (
                data[base + 128 + k // 128 * 32 + k % 32].astype("int32") >> (k % 128 // 32 * 2)
            ) & 3
            scale = data[base + 192 + k // 16].astype("int8").astype("float32")
            result[0] = (
                half(data, base + 208) * scale * ((low | (high << 4)) - 32).astype("float32")
            )
        elif encoding == Encoding.IQ4_XS:
            base = index // 256 * 136
            k = index % 256
            group = k // 32
            high = data[base + 2].astype("int32") | (data[base + 3].astype("int32") << 8)
            low = (data[base + 4 + group // 2].astype("int32") >> (group % 2 * 4)) & 15
            scale = (low | (((high >> (2 * group)) & 3) << 4)) - 32
            code = (data[base + 8 + group * 16 + k % 16].astype("int32") >> (k % 32 // 16 * 4)) & 15
            value = T.if_then_else(
                code == 0,
                -127,
                T.if_then_else(
                    code == 1,
                    -104,
                    T.if_then_else(
                        code == 2,
                        -83,
                        T.if_then_else(
                            code == 3,
                            -65,
                            T.if_then_else(
                                code == 4,
                                -49,
                                T.if_then_else(
                                    code == 5,
                                    -35,
                                    T.if_then_else(
                                        code == 6,
                                        -22,
                                        T.if_then_else(
                                            code == 7,
                                            -10,
                                            T.if_then_else(
                                                code == 8,
                                                1,
                                                T.if_then_else(
                                                    code == 9,
                                                    13,
                                                    T.if_then_else(
                                                        code == 10,
                                                        25,
                                                        T.if_then_else(
                                                            code == 11,
                                                            38,
                                                            T.if_then_else(
                                                                code == 12,
                                                                53,
                                                                T.if_then_else(
                                                                    code == 13,
                                                                    69,
                                                                    T.if_then_else(
                                                                        code == 14, 89, 113
                                                                    ),
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            )
            result[0] = half(data, base) * scale.astype("float32") * value.astype("float32")
        return result[0]

    return decode


@T.macro
def byte(words, offset):
    return (words[offset // 2].astype("uint32") >> ((offset % 2) * 8)) & 255


@T.macro
def iq_value(code):
    index = code.astype("int32")
    word = T.if_then_else(
        index < 8,
        T.if_then_else(index < 4, T.uint32(0xBFAD9881), T.uint32(0xF6EADDCF)),
        T.if_then_else(index < 12, T.uint32(0x26190D01), T.uint32(0x71594535)),
    )
    return ((word >> ((index % 4) * 8)) & 255).astype("int8").astype("float32")


@T.macro
def word32(words, index):
    return words[index * 2].astype("uint32") | (words[index * 2 + 1].astype("uint32") << 16)


def interpretation(representation: Representation, elements: int = 0):
    """Cooperative access: one worker owns an aligned group of eight values."""
    encoding = _blocks(representation)
    planes = None
    group_elements, high_bits, zero_point = 32, 0, 0
    if encoding is None:
        assert isinstance(representation, PlanarAffine)
        if representation.coefficient_dtype != DType.F32:
            raise ValueError("cooperative plane access requires FP32 group coefficients")
        planes = plane_offsets(representation, elements)
        assert planes.low == 0  # the low plane is the origin of the allocation
        group_elements = representation.group
        high_bits = representation.high_bits
        zero_point = representation.zero_point

    @T.macro
    def parameters(words, index):
        scale = T.alloc_local((1,), "float32")
        bias = T.alloc_local((1,), "float32")
        scale[0], bias[0] = 1, 0
        if planes is not None:
            scale[0] = T.reinterpret(
                word32(words, planes.scales + index // group_elements), "float32"
            )
            if zero_point == 0:
                bias[0] = T.reinterpret(
                    word32(words, planes.biases + index // group_elements),
                    "float32",
                )
        elif encoding == Encoding.Q4_K or encoding == Encoding.Q5_K:
            base = index // 256 * encoding.block_bytes
            group = index % 256 // 32
            lo = byte(words, base + 4 + group % 4)
            minimum = byte(words, base + 8 + group % 4)
            hi = byte(words, base + 12 + group % 4)
            s = T.if_then_else(group < 4, lo & 63, (hi & 15) | ((lo >> 6) << 4))
            m = T.if_then_else(group < 4, minimum & 63, (hi >> 4) | ((minimum >> 6) << 4))
            scale[0] = T.reinterpret(words[base // 2], "float16").astype("float32") * s.astype(
                "float32"
            )
            bias[0] = -T.reinterpret(words[base // 2 + 1], "float16").astype("float32") * m.astype(
                "float32"
            )
        elif encoding == Encoding.Q6_K:
            base = index // 256 * 210
            scale[0] = T.reinterpret(words[base // 2 + 104], "float16").astype("float32") * byte(
                words, base + 192 + index % 256 // 16
            ).astype("int8").astype("float32")
        elif encoding == Encoding.IQ4_XS:
            base = index // 256 * 136
            group = index % 256 // 32
            lo = (byte(words, base + 4 + group // 2) >> ((group % 2) * 4)) & 15
            hi = (words[base // 2 + 1].astype("uint32") >> (group * 2)) & 3
            scale[0] = T.reinterpret(words[base // 2], "float16").astype("float32") * (
                (lo | (hi << 4)).astype("int32") - 32
            ).astype("float32")
        elif encoding == Encoding.Q8_0:
            scale[0] = T.reinterpret(words[index // 32 * 17], "float16").astype("float32")
        return scale[0], bias[0]

    @T.macro
    def code(words, index):
        result = T.alloc_local((1,), "float32")
        if planes is not None:
            quant = T.alloc_local((1,), "uint32")
            quant[0] = (word32(words, index // 8) >> ((index % 8) * 4)) & 15
            if high_bits:
                high = (
                    word32(words, planes.high + index // group_elements)
                    >> ((index % group_elements) * high_bits)
                ) & ((1 << high_bits) - 1)
                quant[0] |= high << 4
            result[0] = (quant[0].astype("int32") - zero_point).astype("float32")
        elif encoding == Encoding.Q4_K or encoding == Encoding.Q5_K:
            base = index // 256 * encoding.block_bytes
            k = index % 256
            payload = 16 if encoding == Encoding.Q4_K else 48
            lo = (byte(words, base + payload + k // 64 * 32 + k % 32) >> (k % 64 // 32 * 4)) & 15
            high = T.alloc_local((1,), "uint32")
            high[0] = 0
            if encoding == Encoding.Q5_K:
                high[0] = ((byte(words, base + 16 + k % 32) >> (k // 32)) & 1) << 4
            result[0] = (lo | high[0]).astype("float32")
        elif encoding == Encoding.Q6_K:
            base = index // 256 * 210
            k = index % 256
            lo = (byte(words, base + k // 128 * 64 + k % 64) >> (k % 128 // 64 * 4)) & 15
            hi = (byte(words, base + 128 + k // 128 * 32 + k % 32) >> (k % 128 // 32 * 2)) & 3
            result[0] = ((lo | (hi << 4)).astype("int32") - 32).astype("float32")
        elif encoding == Encoding.IQ4_XS:
            base = index // 256 * 136
            k = index % 256
            packed = byte(words, base + 8 + k // 32 * 16 + k % 16)
            result[0] = iq_value((packed >> (k % 32 // 16 * 4)) & 15)
        elif encoding == Encoding.Q8_0:
            result[0] = (
                byte(words, index // 32 * 34 + 2 + index % 32).astype("int8").astype("float32")
            )
        elif encoding == Encoding.F16:
            result[0] = T.reinterpret(words[index], "float16").astype("float32")
        else:
            bits = words[index * 2].astype("uint32") | (words[index * 2 + 1].astype("uint32") << 16)
            result[0] = T.reinterpret(bits, "float32")
        return result[0]

    return parameters, code
