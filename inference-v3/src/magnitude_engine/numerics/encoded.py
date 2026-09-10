"""Inlined GGML block interpretation, shared by encoded operations.

The input remains packed bytes throughout residency. These macros are expanded
inside consumers; they do not allocate a dequantized weight tensor.
"""

import math

import tilelang.language as T

from magnitude_engine.artifacts.gguf import Encoding
from magnitude_engine.numerics.encoded_layout import EncodedLayout, affine_geometry, storage_bytes
from magnitude_engine.platform.execution import DType


@T.macro
def half(data, offset):
    # A typed temporary prevents C integer promotion from changing the operand
    # width at reinterpretation in the Metal code generator.
    bits = T.alloc_local((1,), "uint16")
    bits[0] = data[offset].astype("uint16") | (data[offset + 1].astype("uint16") << 8)
    return T.reinterpret(bits[0], "float16").astype("float32")


def decoder(encoding: Encoding, layout: EncodedLayout = EncodedLayout.GGUF, elements: int = 0):
    if layout != EncodedLayout.GGUF:
        storage_bytes(elements, encoding, layout)
    geometry = affine_geometry(encoding) if layout == EncodedLayout.AFFINE_PLANES else None
    group_elements = geometry.group_elements if geometry is not None else 32
    high_bits = geometry.high_bits if geometry is not None else 0
    zero_point = geometry.zero_point if geometry is not None else 0

    @T.macro
    def decode(data, index):
        result = T.alloc_local((1,), "float32")
        if layout == EncodedLayout.AFFINE_PLANES:
            scale_offset = elements // 2 + index // group_elements * 4
            bias_offset = (
                elements // 2 + elements // group_elements * 4 + index // group_elements * 4
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
                    data[elements * 3 // 4 + index * high_bits // 8].astype("uint32")
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


def projection(
    rows: int,
    outputs: int,
    inputs: int,
    encoding: Encoding,
    *,
    output_tile: int = 4,
    reduction_lanes: int = 32,
    row_tile: int = 1,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
    layout: EncodedLayout = EncodedLayout.GGUF,
):
    """A portable fused GEMV baseline, also valid for small batched inputs.

    Prefill matrix-tiled candidates will share the decoder and operation contract.
    Reduction lanes and output tile are candidate parameters, not device facts.
    """
    if min(rows, outputs, inputs, output_tile, reduction_lanes, row_tile) <= 0:
        raise ValueError("projection extents and tile sizes must be positive")
    if inputs % encoding.block_elements:
        raise ValueError("encoded rows must contain complete blocks")
    if reduction_lanes & (reduction_lanes - 1):
        raise ValueError("reduction lane count must be a power of two")
    decode = decoder(encoding, layout, outputs * inputs)
    size = storage_bytes(outputs * inputs, encoding, layout)

    @T.prim_func
    def main(
        A: T.Tensor((rows, inputs), dtype.value),
        B: T.Tensor((size,), "uint8"),
        C: T.Tensor((rows, outputs), output_dtype.value),
    ):
        with T.Kernel(
            T.ceildiv(outputs, output_tile),
            T.ceildiv(rows, row_tile),
            threads=output_tile * reduction_lanes,
        ) as (block, row_group):
            partial = T.alloc_fragment((row_tile, output_tile, reduction_lanes), "float32")
            shared = T.alloc_shared((row_tile, output_tile, reduction_lanes), "float32")
            T.clear(partial)
            for chunk in T.serial(T.ceildiv(inputs, reduction_lanes)):
                for i, j in T.Parallel(output_tile, reduction_lanes):
                    out = block * output_tile + i
                    k = chunk * reduction_lanes + j
                    if out < outputs and k < inputs:
                        weight = decode(B, out * inputs + k)
                        for r in T.unroll(row_tile, explicit=True):
                            row = row_group * row_tile + r
                            if row < rows:
                                partial[r, i, j] += A[row, k].astype("float32") * weight
            for r, i, j in T.Parallel(row_tile, output_tile, reduction_lanes):
                shared[r, i, j] = partial[r, i, j]
            T.sync_threads()
            for step in T.unroll(int(math.log2(reduction_lanes))):
                for r, i, j in T.Parallel(row_tile, output_tile, reduction_lanes):
                    if j < (reduction_lanes >> (step + 1)):
                        shared[r, i, j] += shared[r, i, j + (reduction_lanes >> (step + 1))]
                T.sync_threads()
            for r, i in T.Parallel(row_tile, output_tile):
                row = row_group * row_tile + r
                if block * output_tile + i < outputs and row < rows:
                    C[row, block * output_tile + i] = shared[r, i, 0]

    return main


def projection_cpu(
    rows: int,
    outputs: int,
    inputs: int,
    encoding: Encoding,
    *,
    row_tile: int = 1,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
    layout: EncodedLayout = EncodedLayout.GGUF,
):
    """CPU schedule for the same encoded contraction; no GPU fragment semantics."""
    if min(rows, outputs, inputs, row_tile) <= 0 or inputs % encoding.block_elements:
        raise ValueError("invalid encoded projection geometry")
    decode = decoder(encoding, layout, outputs * inputs)
    size = storage_bytes(outputs * inputs, encoding, layout)

    @T.prim_func
    def main(
        A: T.Tensor((rows, inputs), dtype.value),
        B: T.Tensor((size,), "uint8"),
        C: T.Tensor((rows, outputs), output_dtype.value),
    ):
        for row_group, out in T.Parallel(T.ceildiv(rows, row_tile), outputs):
            total = T.alloc_local((row_tile,), "float32")
            for r in T.unroll(row_tile, explicit=True):
                total[r] = 0
            for k in T.serial(inputs):
                weight = decode(B, out * inputs + k)
                for r in T.unroll(row_tile, explicit=True):
                    row = row_group * row_tile + r
                    if row < rows:
                        total[r] += A[row, k].astype("float32") * weight
            for r in T.unroll(row_tile, explicit=True):
                row = row_group * row_tile + r
                if row < rows:
                    C[row, out] = total[r]

    return main


def gather(
    rows: int,
    vocabulary: int,
    width: int,
    encoding: Encoding,
    *,
    cpu: bool,
    threads=128,
    dtype: DType = DType.F32,
    layout: EncodedLayout = EncodedLayout.GGUF,
):
    """Read selected embedding rows while leaving the shared table encoded."""
    if min(rows, vocabulary, width, threads) <= 0 or width % encoding.block_elements:
        raise ValueError("invalid encoded embedding geometry")
    decode = decoder(encoding, layout, vocabulary * width)
    size = storage_bytes(vocabulary * width, encoding, layout)

    @T.prim_func
    def main(
        Indices: T.Tensor((rows,), "int32"),
        W: T.Tensor((size,), "uint8"),
        C: T.Tensor((rows, width), dtype.value),
    ):
        if cpu:
            for row, col in T.Parallel(rows, width):
                if Indices[row] >= 0 and Indices[row] < vocabulary:
                    C[row, col] = decode(W, Indices[row] * width + col)
                else:
                    C[row, col] = T.reinterpret(T.uint32(0x7FC00000), "float32")
        else:
            with T.Kernel(T.ceildiv(width, threads), rows, threads=threads) as (block, row):
                col = block * threads + T.get_thread_binding(0)
                if col < width:
                    if Indices[row] >= 0 and Indices[row] < vocabulary:
                        C[row, col] = decode(W, Indices[row] * width + col)
                    else:
                        C[row, col] = T.reinterpret(T.uint32(0x7FC00000), "float32")

    return main
