"""Gate and up over one compact scale/min hierarchy, sharing every input load."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.precision import Precision, Rounding, native_sigmoid
from magnitude_engine.kernels.projection.packed_k import scale_min
from magnitude_engine.weights.representation import HierarchicalAffine, HierarchyPacking


def gated_vector(
    M: int,
    N: int,
    K: int,
    representation: HierarchicalAffine,
    *,
    capability: Capability,
    precision: Precision,
):
    if capability.subgroup_width != 32:
        raise ValueError("the hierarchical gate reduces across a 32-lane subgroup")
    if precision.rounding != Rounding.NATIVE_BF16:
        raise ValueError("the hierarchical gate rounds the way NATIVE_BF16 does")
    if representation.packing != HierarchyPacking.SCALE_MIN_I6 or K % 256:
        raise ValueError("the hierarchical gate requires complete scale/min superblocks")
    block_words = representation.block_bytes // 4
    payload_words = 4 + representation.high_bits * 8
    words = 2 * N * K // 256 * block_words

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((words,), "uint32"),
        C: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(N, M, threads=32) as (out, row):
            lane = T.get_thread_binding()
            partial = T.alloc_local((2,), "float32")
            dot = T.alloc_local((2, 2), "float32")
            sums = T.alloc_local((2,), "float32")
            input0 = T.alloc_local((4,), "float32")
            input1 = T.alloc_local((4,), "float32")
            half_bits = T.alloc_local((2,), "uint16")
            high = T.alloc_local((1,), "uint32")
            T.clear(partial)
            for chunk in T.serial(K // 256):
                group = lane // 8 * 2
                T.clear(sums)
                for j in T.unroll(4, explicit=True):
                    k = chunk * 256 + lane // 8 * 64 + lane % 8 * 4 + j
                    input0[j] = A[row, k].astype("float32")
                    input1[j] = A[row, k + 32].astype("float32")
                    sums[0] += input0[j]
                    sums[1] += input1[j]
                for branch in T.unroll(2, explicit=True):
                    base = ((branch * N + out) * (K // 256) + chunk) * block_words
                    header = B[base]
                    half_bits[0] = (header & T.uint32(65535)).astype("uint16")
                    half_bits[1] = (header >> 16).astype("uint16")
                    d = T.reinterpret(half_bits[0], "float16").astype("float32")
                    minimum = T.reinterpret(half_bits[1], "float16").astype("float32")
                    scale0, bias0 = scale_min(B[base + 1], B[base + 2], B[base + 3], group)
                    scale1, bias1 = scale_min(B[base + 1], B[base + 2], B[base + 3], group + 1)
                    word = B[base + payload_words + lane]
                    high[0] = 0
                    if representation.high_bits:
                        high[0] = B[base + 4 + lane % 8]
                    T.clear(dot)
                    for j in T.unroll(4, explicit=True):
                        q0 = ((word >> (j * 8)) & T.uint32(15)) | (
                            ((high[0] >> (j * 8 + group)) & T.uint32(1)) << 4
                        )
                        q1 = ((word >> (j * 8 + 4)) & T.uint32(15)) | (
                            ((high[0] >> (j * 8 + group + 1)) & T.uint32(1)) << 4
                        )
                        dot[branch, 0] += input0[j] * q0.astype("float32")
                        dot[branch, 1] += input1[j] * q1.astype("float32")
                    partial[branch] += d * (
                        scale0 * dot[branch, 0] + scale1 * dot[branch, 1]
                    ) - minimum * (bias0 * sums[0] + bias1 * sums[1])
            gate = T.warp_reduce_sum(partial[0]).astype("bfloat16").astype("float32")
            up = T.warp_reduce_sum(partial[1]).astype("bfloat16").astype("float32")
            if lane == 0:
                activated = (
                    (gate * native_sigmoid(gate).astype("float32"))
                    .astype("bfloat16")
                    .astype("float32")
                )
                C[row, out] = activated * up

    return main
