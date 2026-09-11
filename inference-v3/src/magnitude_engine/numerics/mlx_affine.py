"""Affine Q4/group-64 contractions ported from tilelang-poc.

Codes and BF16 coefficients stay encoded in global memory. GEMV accumulates
integer-code dot products and affine corrections; GEMM stages only its current
BF16 operand tile. Grouped outputs are placed directly in contiguous segments.
"""

import tilelang.language as T

from magnitude_engine.artifacts.weights import WeightTransform
from magnitude_engine.platform.execution import DType


def output_index(rows, widths):
    def index(row, col):
        result = row * widths[0] + col
        start = widths[0]
        for width in widths[1:]:
            result = T.if_then_else(col >= start, rows * start + row * width + col - start, result)
            start += width
        return result

    return index


def partitions(m, n, k):
    parts = min(max(1, 512 // (((m + 31) // 32) * ((n + 31) // 32))), k // 64)
    while k % (parts * 64):
        parts -= 1
    return parts


def vector(M, widths: tuple[int, ...], K, output_dtype: DType = DType.BF16):
    N = sum(widths)
    output_at = output_index(M, widths)
    # The reference uses eight coordinates/lane for output tails (e.g. the
    # 257-row MoE router). This changes where affine corrections are rounded.
    pack = 16 if N % 8 == 0 else 8

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K // 8), "uint32"),
        C: T.Tensor((N, K // 64), "bfloat16"),
        D: T.Tensor((N, K // 64), "bfloat16"),
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
                    partial[0] += dot[0] * C[row, base // 64].astype("float32") + bias_sum[0] * D[
                        row, base // 64
                    ].astype("float32")
            total = T.warp_reduce_sum(partial[0])
            if lane == 0:
                E[output_at(token, row)] = total.astype("bfloat16")

    return main


def matrix(M, widths: tuple[int, ...], K, PARTS, BM, BN, BK, PAD, output_dtype: DType = DType.BF16):
    N = sum(widths)
    output_at = output_index(M, widths)
    assert K % (PARTS * 64) == 0 and (K // PARTS) % BK == 0
    span = K // PARTS

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K // 8), "uint32"),
        C: T.Tensor((N, K // 64), "bfloat16"),
        D: T.Tensor((N, K // 64), "bfloat16"),
        E: T.Tensor((PARTS, M * N), output_dtype.value),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), PARTS, threads=128) as (bx, by, part):
            tid = T.get_thread_binding()
            inputs = T.alloc_shared((BM, BK + PAD), "bfloat16")
            weights = T.alloc_shared((BN, BK + PAD), "bfloat16")
            result = T.alloc_shared((BM, BN), "float32")
            accum = T.alloc_fragment((BM, BN), "float32")
            T.clear(accum)
            for block in T.serial(span // BK):
                for i, j in T.Parallel(BM, BK):
                    if by * BM + i < M:
                        inputs[i, j] = A[by * BM + i, part * span + block * BK + j]
                    else:
                        inputs[i, j] = T.cast(0, "bfloat16")
                for word_index in T.serial(T.ceildiv(BN * BK // 8, 128)):
                    index = word_index * 128 + tid
                    row = index // (BK // 8)
                    col = index % (BK // 8) * 8
                    if row < BN:
                        if bx * BN + row < N:
                            global_col = part * span + block * BK + col
                            packed = B[bx * BN + row, global_col // 8]
                            scale = C[bx * BN + row, global_col // 64].astype("float32")
                            bias = D[bx * BN + row, global_col // 64].astype("float32")
                            for offset in T.unroll(8, explicit=True):
                                code = (packed >> T.uint32(offset * 4)) & T.uint32(15)
                                weights[row, col + offset] = code.astype("float32") * scale + bias
                        else:
                            for offset in T.unroll(8, explicit=True):
                                weights[row, col + offset] = T.cast(0, "bfloat16")
                T.sync_threads()
                T.gemm(inputs[:, :BK], weights[:, :BK], accum, transpose_B=True)
                T.sync_threads()
            T.sync_threads()
            # Fragment stores are opaque to automatic barrier insertion. Result
            # scratch may alias input tiles; all groups must finish reads first.
            T.copy(accum, result)
            T.sync_threads()
            for i, j in T.Parallel(BM, BN):
                if by * BM + i < M and bx * BN + j < N:
                    E[
                        part,
                        output_at(by * BM + i, bx * BN + j)
                        if PARTS == 1
                        else (by * BM + i) * N + bx * BN + j,
                    ] = result[i, j].astype("bfloat16")

    return main


def finish(M, widths: tuple[int, ...], PARTS, output_dtype: DType = DType.BF16):
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


def embedding(N, K, ROWS=1, cpu=False):
    @T.macro
    def load(A, B, C, D, E, row, k):
        if k < K:
            code = (A[D[row], k // 8] >> ((k % 8) * 4)) & T.uint32(15)
            E[row, k] = code.astype("float32") * B[D[row], k // 64].astype("float32") + C[
                D[row], k // 64
            ].astype("float32")

    @T.prim_func
    def main(
        A: T.Tensor((N, K // 8), "uint32"),
        B: T.Tensor((N, K // 64), "bfloat16"),
        C: T.Tensor((N, K // 64), "bfloat16"),
        D: T.Tensor((ROWS,), "int32"),
        E: T.Tensor((ROWS, K), "bfloat16"),
    ):
        if cpu:
            for row, k in T.Parallel(ROWS, K):
                load(A, B, C, D, E, row, k)
        else:
            with T.Kernel(T.ceildiv(K, 128), ROWS, threads=128) as (block, row):
                k = block * 128 + T.get_thread_binding()
                load(A, B, C, D, E, row, k)

    return main


def parameter(size: int, dtype: DType, transform: WeightTransform, cpu: bool):
    @T.macro
    def convert(A, B, i):
        value = A[i].astype("float32")
        B[i] = -T.exp(value) if transform == WeightTransform.NEGATIVE_EXP else value

    @T.prim_func
    def main(A: T.Tensor((size,), dtype.value), B: T.Tensor((size,), "float32")):
        if cpu:
            for i in T.Parallel(size):
                convert(A, B, i)
        else:
            with T.Kernel(T.ceildiv(size, 128), threads=128) as block:
                i = block * 128 + T.get_thread_binding()
                if i < size:
                    convert(A, B, i)

    return main


def portable(M, widths: tuple[int, ...], K, cpu: bool, output_dtype: DType):
    N = sum(widths)
    output_at = output_index(M, widths)

    @T.macro
    def dot(A, B, C, D, E, row, col):
        total = T.alloc_local((1,), "float32")
        total[0] = 0
        for k in T.serial(K):
            code = (B[col, k // 8] >> ((k % 8) * 4)) & T.uint32(15)
            weight = (
                code.astype("float32") * C[col, k // 64].astype("float32")
                + D[col, k // 64].astype("float32")
            ).astype("bfloat16")
            total[0] += A[row, k].astype("float32") * weight.astype("float32")
        E[output_at(row, col)] = total[0].astype("bfloat16")

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((N, K // 8), "uint32"),
        C: T.Tensor((N, K // 64), "bfloat16"),
        D: T.Tensor((N, K // 64), "bfloat16"),
        E: T.Tensor((M * N,), output_dtype.value),
    ):
        if cpu:
            for row, col in T.Parallel(M, N):
                dot(A, B, C, D, E, row, col)
        else:
            with T.Kernel(T.ceildiv(N, 128), M, threads=128) as (block, row):
                col = block * 128 + T.get_thread_binding()
                if col < N:
                    dot(A, B, C, D, E, row, col)

    return main


def gated_vector(M, N, K):
    from magnitude_engine.numerics.native_bf16 import native_sigmoid

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((2 * N, K // 8), "uint32"),
        C: T.Tensor((2 * N, K // 64), "bfloat16"),
        D: T.Tensor((2 * N, K // 64), "bfloat16"),
        E: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(N, M, threads=32) as (row, token):
            lane = T.get_thread_binding()
            partial = T.alloc_local((2,), "float32")
            dot = T.alloc_local((2,), "float32")
            bias_sum = T.alloc_local((1,), "float32")
            partial[0] = 0
            partial[1] = 0
            for i in T.serial(T.ceildiv(K, 512)):
                base = (i * 32 + lane) * 16
                dot[0] = 0
                dot[1] = 0
                bias_sum[0] = 0
                for j in T.unroll(4, explicit=True):
                    column = base + j * 4
                    if column < K:
                        x0 = A[token, column].astype("float32")
                        x1 = A[token, column + 1].astype("float32")
                        x2 = A[token, column + 2].astype("float32")
                        x3 = A[token, column + 3].astype("float32")
                        xs = (x0 + x1).astype("bfloat16").astype("float32")
                        xs2 = (xs + x2).astype("bfloat16").astype("float32")
                        bias_sum[0] += (xs2 + x3).astype("bfloat16").astype("float32")
                        for branch in T.unroll(2, explicit=True):
                            word = (
                                B[branch * N + row, column // 8] >> ((column % 8) * 4)
                            ) & T.uint32(65535)
                            dot[branch] += (
                                x0 * (word & T.uint32(15)).astype("float32")
                                + x1 * ((word >> 4) & T.uint32(15)).astype("float32")
                                + x2 * ((word >> 8) & T.uint32(15)).astype("float32")
                                + x3 * ((word >> 12) & T.uint32(15)).astype("float32")
                            )
                if base < K:
                    for branch in T.unroll(2, explicit=True):
                        partial[branch] += dot[branch] * C[branch * N + row, base // 64].astype(
                            "float32"
                        ) + bias_sum[0] * D[branch * N + row, base // 64].astype("float32")
            gate = T.warp_reduce_sum(partial[0]).astype("bfloat16").astype("float32")
            up = T.warp_reduce_sum(partial[1]).astype("bfloat16").astype("float32")
            if lane == 0:
                activated = (
                    (gate * native_sigmoid(gate).astype("float32"))
                    .astype("bfloat16")
                    .astype("float32")
                )
                E[token, row] = activated * up

    return main
