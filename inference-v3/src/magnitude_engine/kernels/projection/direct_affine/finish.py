"""Reconcile the partial sums a partitioned direct-affine contraction produced."""

import tilelang.language as T

from magnitude_engine.kernels.projection.layout import output_index
from magnitude_engine.platform.execution import DType


def finish(M, widths: tuple[int, ...], PARTS, output_dtype: DType = DType.BF16):
    """Reconcile partial contraction sums, rounding the way the reference does."""
    N = sum(widths)
    output_at = output_index(M, widths)
    # Native partial outputs and strided local sums round to BF16.
    # The SIMD reduction of32 local sums uses FP32, then rounds once.
    groups = min(PARTS, 8)

    @T.prim_func
    def main(A: T.Tensor((PARTS, M * N), "bfloat16"), B: T.Tensor((M * N,), output_dtype.value)):
        with T.Kernel(
            T.ceildiv(M * N, 128) if PARTS < 32 else M * N, threads=128 if PARTS < 32 else 32
        ) as block:
            lane = T.get_thread_binding()
            idx = block * 128 + lane if PARTS < 32 else block
            value = T.alloc_local((1,), "float32")
            part_sum = T.alloc_local((1,), "float32")
            value[0] = 0
            if idx < M * N:
                if PARTS < 32:
                    for group in T.unroll(groups, explicit=True):
                        part_sum[0] = 0
                        for step in T.serial(T.ceildiv(PARTS, groups)):
                            part = step * groups + group
                            if part < PARTS:
                                part_sum[0] = (
                                    (part_sum[0] + A[part, idx].astype("float32"))
                                    .astype("bfloat16")
                                    .astype("float32")
                                )
                        value[0] = (value[0] + part_sum[0]).astype("bfloat16").astype("float32")
                    B[output_at(idx // N, idx % N)] = value[0].astype("bfloat16")
                else:
                    for step in T.serial(T.ceildiv(PARTS, 32)):
                        part = step * 32 + lane
                        if part < PARTS:
                            value[0] = (
                                (value[0] + A[part, idx].astype("float32"))
                                .astype("bfloat16")
                                .astype("float32")
                            )
                    total = T.warp_reduce_sum(value[0])
                    if lane == 0:
                        B[output_at(idx // N, idx % N)] = total.astype("bfloat16")

    return main
