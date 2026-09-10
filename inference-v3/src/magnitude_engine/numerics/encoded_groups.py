"""Quantization-group interpretation for cooperative GPU contractions.

Every supported encoding has groups aligned to eight logical coordinates.
Consumers fetch the affine parameters once, then consume the group's payload.
The scalar random-access interpreter is separate from this cooperative layout.
"""

import tilelang.language as T

from magnitude_engine.artifacts.gguf import Encoding
from magnitude_engine.numerics.encoded_layout import EncodedLayout, affine_geometry, storage_bytes


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


def interpretation(
    encoding: Encoding, layout: EncodedLayout = EncodedLayout.GGUF, elements: int = 0
):
    if layout != EncodedLayout.GGUF:
        storage_bytes(elements, encoding, layout)
    geometry = affine_geometry(encoding) if layout == EncodedLayout.AFFINE_PLANES else None
    group_elements = geometry.group_elements if geometry is not None else 32
    high_bits = geometry.high_bits if geometry is not None else 0
    zero_point = geometry.zero_point if geometry is not None else 0

    @T.macro
    def parameters(words, index):
        scale = T.alloc_local((1,), "float32")
        bias = T.alloc_local((1,), "float32")
        scale[0], bias[0] = 1, 0
        if layout == EncodedLayout.AFFINE_PLANES:
            scale[0] = T.reinterpret(
                word32(words, elements // 8 + index // group_elements), "float32"
            )
            if zero_point == 0:
                bias[0] = T.reinterpret(
                    word32(
                        words, elements // 8 + elements // group_elements + index // group_elements
                    ),
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
        if layout == EncodedLayout.AFFINE_PLANES:
            quant = T.alloc_local((1,), "uint32")
            quant[0] = (word32(words, index // 8) >> ((index % 8) * 4)) & 15
            if high_bits:
                high = (
                    word32(words, elements * 3 // 16 + index // group_elements)
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
