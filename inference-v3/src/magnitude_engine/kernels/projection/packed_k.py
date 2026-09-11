"""Q4_K/Q5_K contraction that loads packed words and corrects once per group."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import (
    Blocked,
    Encoding,
    Representation,
    resident_bytes,
)


@T.macro
def scale_min(first, second, third, group):
    shift = (group % 4) * 8
    low = (first >> shift) & T.uint32(255)
    minimum = (second >> shift) & T.uint32(255)
    high = (third >> shift) & T.uint32(255)
    scale = T.if_then_else(group < 4, low & T.uint32(63), (high & T.uint32(15)) | ((low >> 6) << 4))
    bias = T.if_then_else(group < 4, minimum & T.uint32(63), (high >> 4) | ((minimum >> 6) << 4))
    return scale.astype("float32"), bias.astype("float32")


def projection(
    rows: int,
    outputs: int,
    inputs: int,
    representation: Representation,
    *,
    capability: Capability,
    output_tile: int = 4,
    row_tile: int = 1,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
):
    """Packed integer loads and shared affine corrections for Q4_K/Q5_K.

    Each lane consumes one packed word (eight weights) from a 256-element
    superblock. The subgroup width is an explicit eligibility requirement.
    """
    subgroup_width = capability.subgroup_width
    if subgroup_width != 32:
        raise ValueError("this block schedule requires a queried 32-lane subgroup")
    encoding = representation.encoding if isinstance(representation, Blocked) else None
    if encoding not in (Encoding.Q4_K, Encoding.Q5_K) or inputs % 256:
        raise ValueError("packed K projection requires complete Q4_K/Q5_K blocks")
    if min(rows, outputs, inputs, output_tile, row_tile) <= 0:
        raise ValueError("projection extents must be positive")
    block_words = encoding.block_bytes // 4
    payload_words = 4 if encoding == Encoding.Q4_K else 12
    size = resident_bytes(representation, outputs * inputs) // 4

    @T.prim_func
    def main(
        A: T.Tensor((rows, inputs), dtype.value),
        B: T.Tensor((size,), "uint32"),
        C: T.Tensor((rows, outputs), output_dtype.value),
    ):
        with T.Kernel(
            T.ceildiv(outputs, output_tile),
            T.ceildiv(rows, row_tile),
            threads=output_tile * subgroup_width,
        ) as (block, row_group):
            thread = T.get_thread_binding()
            lane = thread % subgroup_width
            out = block * output_tile + thread // subgroup_width
            partial = T.alloc_local((row_tile,), "float32")
            half_bits = T.alloc_local((2,), "uint16")
            dot = T.alloc_local((row_tile, 2), "float32")
            sums = T.alloc_local((row_tile, 2), "float32")
            high_bits = T.alloc_local((1,), "uint32")
            T.clear(partial)
            if out < outputs:
                for chunk in T.serial(inputs // 256):
                    base = (out * (inputs // 256) + chunk) * block_words
                    header = B[base]
                    half_bits[0] = (header & T.uint32(65535)).astype("uint16")
                    half_bits[1] = (header >> 16).astype("uint16")
                    d = T.reinterpret(half_bits[0], "float16").astype("float32")
                    minimum = T.reinterpret(half_bits[1], "float16").astype("float32")
                    first, second, third = B[base + 1], B[base + 2], B[base + 3]
                    group = lane // 8 * 2
                    scale0, bias0 = scale_min(first, second, third, group)
                    scale1, bias1 = scale_min(first, second, third, group + 1)
                    word = B[base + payload_words + lane]
                    high_bits[0] = 0
                    if encoding == Encoding.Q5_K:
                        high_bits[0] = B[base + 4 + lane % 8]
                    T.clear(dot)
                    T.clear(sums)
                    for j in T.unroll(4, explicit=True):
                        k = chunk * 256 + lane // 8 * 64 + lane % 8 * 4 + j
                        q0 = ((word >> (j * 8)) & T.uint32(15)) | (
                            ((high_bits[0] >> (j * 8 + group)) & T.uint32(1)) << 4
                        )
                        q1 = ((word >> (j * 8 + 4)) & T.uint32(15)) | (
                            ((high_bits[0] >> (j * 8 + group + 1)) & T.uint32(1)) << 4
                        )
                        for r in T.unroll(row_tile, explicit=True):
                            row = row_group * row_tile + r
                            if row < rows:
                                x0, x1 = (
                                    A[row, k].astype("float32"),
                                    A[row, k + 32].astype("float32"),
                                )
                                dot[r, 0] += x0 * q0.astype("float32")
                                dot[r, 1] += x1 * q1.astype("float32")
                                sums[r, 0] += x0
                                sums[r, 1] += x1
                    for r in T.unroll(row_tile, explicit=True):
                        partial[r] += d * (scale0 * dot[r, 0] + scale1 * dot[r, 1]) - minimum * (
                            bias0 * sums[r, 0] + bias1 * sums[r, 1]
                        )
            for r in T.unroll(row_tile, explicit=True):
                total = T.warp_reduce_sum(partial[r])
                row = row_group * row_tile + r
                if lane == 0 and out < outputs and row < rows:
                    C[row, out] = total

    return main
