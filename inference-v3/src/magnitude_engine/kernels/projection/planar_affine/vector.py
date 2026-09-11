"""One GEMV family over the planar affine representation.

The two branches below are the same contraction against different group
parameters, and they are one factory because a representation — not a container
and not a backend — decides which one is traced. FP32 coefficients are read out
of the flat word plane by lanes that each own a 512-coordinate fold; BF16
coefficients are read per row, so their planes are bound as separate views and
each lane owns a strided pack. Both accumulate integer-code dot products and
apply the affine correction once per group.
"""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.projection.layout import output_index
from magnitude_engine.kernels.projection.planar_affine.layout import check
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import PlanarAffine, plane_offsets


def vector(
    rows: int,
    widths: tuple[int, ...],
    inputs: int,
    representation: PlanarAffine,
    *,
    capability: Capability,
    dtype: DType = DType.BF16,
    output_dtype: DType = DType.BF16,
):
    check(representation)
    if capability.subgroup_width != 32:
        raise ValueError("the affine GEMV reduces across a 32-lane subgroup")
    if min(rows, inputs) <= 0 or not widths or any(width <= 0 for width in widths):
        raise ValueError("affine GEMV extents must be positive")
    if representation.coefficient_dtype == DType.BF16:
        return _row_planes(rows, widths, inputs, representation, output_dtype)
    if len(widths) != 1:
        raise ValueError("flat-plane affine GEMV computes one output segment")
    return _flat_plane(rows, widths[0], inputs, representation, dtype, output_dtype)


def _flat_plane(
    rows: int,
    outputs: int,
    inputs: int,
    representation: PlanarAffine,
    dtype: DType,
    output_dtype: DType,
):
    if inputs % 512:
        raise ValueError("complete 512-coordinate folds required")
    group_elements = representation.group
    high_bits = representation.high_bits
    zero_point = representation.zero_point
    elements = outputs * inputs
    offsets = plane_offsets(representation, elements)
    words = offsets.words

    @T.prim_func
    def main(
        A: T.Tensor((rows, inputs), dtype.value),
        B: T.Tensor((words,), "uint32"),
        C: T.Tensor((rows, outputs), output_dtype.value),
    ):
        with T.Kernel(T.ceildiv(outputs, 8), rows, threads=64) as (block, row):
            thread = T.get_thread_binding()
            lane = thread % 32
            first = block * 8 + thread // 32 * 4
            values = T.alloc_local((16,), "float32")
            high_values = T.alloc_local((16,), "float32")
            accum = T.alloc_local((4,), "float32")
            total = T.alloc_local((1,), "float32")
            dot = T.alloc_local((1,), "float32")
            T.clear(accum)
            for step in T.serial(inputs // 512):
                k = step * 512 + lane * 16
                total[0] = 0
                for j in T.unroll(16, explicit=True):
                    x = A[row, k + j].astype("float32")
                    total[0] += x
                    values[j] = x * (1.0 / (1 << (4 * (j % 4))))
                    if high_bits:
                        high_values[j] = x * (16.0 / (1 << (j * high_bits)))
                for out in T.unroll(4, explicit=True):
                    if first + out < outputs:
                        g = (first + out) * (inputs // group_elements) + k // group_elements
                        scale = T.reinterpret(B[offsets.scales + g], "float32")
                        dot[0] = 0
                        for packet in T.unroll(2, explicit=True):
                            packed = B[(first + out) * (inputs // 8) + k // 8 + packet]
                            for half in T.unroll(2, explicit=True):
                                word = (packed >> T.uint32(half * 16)) & T.uint32(65535)
                                for j in T.unroll(4, explicit=True):
                                    dot[0] += values[packet * 8 + half * 4 + j] * (
                                        word & T.uint32(15 << (j * 4))
                                    ).astype("float32")
                        if high_bits:
                            high_word = B[offsets.high + g] >> T.uint32(
                                (k % group_elements) * high_bits
                            )
                            for j in T.unroll(16, explicit=True):
                                coefficient = (
                                    high_word
                                    & (T.uint32((1 << high_bits) - 1) << T.uint32(j * high_bits))
                                ).astype("float32") - (
                                    T.uint32(zero_point // 16) << T.uint32(j * high_bits)
                                ).astype("float32")
                                dot[0] += high_values[j] * coefficient
                        if zero_point:
                            accum[out] += scale * dot[0]
                        else:
                            bias = T.reinterpret(B[offsets.biases + g], "float32")
                            accum[out] += scale * dot[0] + bias * total[0]
            for out in T.unroll(4, explicit=True):
                result = T.warp_reduce_sum(accum[out])
                if lane == 0 and first + out < outputs:
                    C[row, first + out] = result

    return main


def _row_planes(
    rows: int,
    widths: tuple[int, ...],
    inputs: int,
    representation: PlanarAffine,
    output_dtype: DType,
):
    if representation.high_bits or not representation.has_bias:
        raise ValueError("row-addressed planes carry a low plane, scales and biases")
    M, K, group = rows, inputs, representation.group
    if K % group:
        raise ValueError("row-addressed planes require complete groups per row")
    N = sum(widths)
    output_at = output_index(M, widths)
    # The reference uses eight coordinates/lane for output tails (e.g. the
    # 257-row MoE router). This changes where affine corrections are rounded.
    pack = 16 if N % 8 == 0 else 8

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K // 8), "uint32"),
        C: T.Tensor((N, K // group), "bfloat16"),
        D: T.Tensor((N, K // group), "bfloat16"),
        E: T.Tensor((M * N,), output_dtype.value),
    ):
        with T.Kernel(N, M, threads=32) as (row, token):
            lane = T.get_thread_binding()
            partial = T.alloc_local((1,), "float32")
            dot = T.alloc_local((1,), "float32")
            bias_sum = T.alloc_local((1,), "float32")
            partial[0] = 0
            for i in T.serial(T.ceildiv(K, 32 * pack)):
                base = (i * 32 + lane) * pack
                dot[0] = 0
                bias_sum[0] = 0
                for j in T.serial(pack // 4):
                    k = base + j * 4
                    if k < K:
                        word = (B[row, k // 8] >> ((k % 8) * 4)) & T.uint32(65535)
                        x0 = A[token, k].astype("float32")
                        x1 = A[token, k + 1].astype("float32")
                        x2 = A[token, k + 2].astype("float32")
                        x3 = A[token, k + 3].astype("float32")
                        xs = (x0 + x1).astype("bfloat16").astype("float32")
                        xs2 = (xs + x2).astype("bfloat16").astype("float32")
                        bias_sum[0] += (xs2 + x3).astype("bfloat16").astype("float32")
                        dot[0] += (
                            x0 * (word & T.uint32(15)).astype("float32")
                            + x1 * ((word >> 4) & T.uint32(15)).astype("float32")
                            + x2 * ((word >> 8) & T.uint32(15)).astype("float32")
                            + x3 * ((word >> 12) & T.uint32(15)).astype("float32")
                        )
                if base < K:
                    partial[0] += dot[0] * C[row, base // group].astype(
                        "float32"
                    ) + bias_sum[0] * D[row, base // group].astype("float32")
            total = T.warp_reduce_sum(partial[0])
            if lane == 0:
                E[output_at(token, row)] = total.astype("bfloat16")

    return main
