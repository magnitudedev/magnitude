"""Three-stage matrix attention over bounded physical segments.

The score and value contractions keep their accumulators in matrix fragments.
Only softmax scans scores; it does not repeatedly rescale a shared output tile.
Each segment produces normalized output and mergeable max/sum statistics.
"""

import math

import tilelang.language as T

from magnitude_engine.platform.execution import DType


def scores(
    rows,
    heads,
    kv_heads,
    width,
    capacity,
    segments,
    length,
    dtype: DType = DType.F32,
    score_dtype: DType = DType.F32,
):
    head_tile = heads // kv_heads if rows < 8 else 1
    groups, queries = heads // head_tile, rows * head_tile
    native = score_dtype == DType.BF16
    bm, bn, bk = (64 if native and rows >= 256 else 8 if queries < 32 else 32), 64, 32
    pad = 8 if native else 0

    @T.macro
    def contract(a, b, accum, Q, K, Run, segment, head_group, bx, by, guarded):
        for block in T.serial(T.ceildiv(width, bk)):
            for i, j in T.Parallel(bm, bk):
                a[i, j] = 0
                if by * bm + i < queries and block * bk + j < width:
                    query = by * bm + i
                    a[i, j] = Q[
                        query // head_tile,
                        head_group * head_tile + query % head_tile,
                        block * bk + j,
                    ].astype("float32") * (1 / math.sqrt(width) if native else 1)
            for i, j in T.Parallel(bn, bk):
                b[i, j] = 0
                if (not guarded or bx * bn + i < Run[segment, 2]) and block * bk + j < width:
                    b[i, j] = K[
                        Run[segment, 0] + bx * bn + i,
                        head_group // (groups // kv_heads),
                        block * bk + j,
                    ]
            T.sync_threads()
            T.gemm(a[:, :bk], b[:, :bk], accum, transpose_B=True)
            T.sync_threads()

    @T.prim_func
    def main(
        Q: T.Tensor((rows, heads, width), dtype.value),
        K: T.Tensor((capacity, kv_heads, width), dtype.value),
        Positions: T.Tensor((rows,), "int32"),
        Run: T.Tensor((segments, 3), "int32"),
        Scores: T.Tensor((segments, heads, rows, length), score_dtype.value),
    ):
        with T.Kernel(
            T.ceildiv(length, bn), T.ceildiv(queries, bm), segments * groups, threads=128
        ) as (bx, by, z):
            segment, head_group = z // groups, z % groups
            if bx * bn < Run[segment, 2]:
                a = T.alloc_shared((bm, bk + pad), dtype.value)
                b = T.alloc_shared((bn, bk + pad), dtype.value)
                accum = T.alloc_fragment((bm, bn), "float32")
                result = T.alloc_shared((bm, bn), "float32")
                T.clear(accum)
                if (bx + 1) * bn <= Run[segment, 2]:
                    contract(a, b, accum, Q, K, Run, segment, head_group, bx, by, False)
                else:
                    contract(a, b, accum, Q, K, Run, segment, head_group, bx, by, True)
                T.copy(accum, result)
                T.sync_threads()
                for i, j in T.Parallel(bm, bn):
                    query, token = by * bm + i, bx * bn + j
                    row = query // head_tile
                    head = head_group * head_tile + query % head_tile
                    if query < queries and token < length:
                        Scores[segment, head, row, token] = T.if_then_else(
                            token < Run[segment, 2] and Run[segment, 1] + token <= Positions[row],
                            result[i, j] * (1 if native else 1 / math.sqrt(width)),
                            -3.402823466e38,
                        )

    return main


def normalize(rows, heads, segments, length):
    threads = 256

    @T.prim_func
    def main(
        Scores: T.Tensor((segments, heads, rows, length), "float32"),
        Positions: T.Tensor((rows,), "int32"),
        Run: T.Tensor((segments, 3), "int32"),
        Statistics: T.Tensor((segments, rows, heads, 2), "float32"),
    ):
        with T.Kernel(rows, heads, segments, threads=threads) as (row, head, segment):
            lane = T.get_thread_binding()
            local = T.alloc_local((1,), "float32")
            scratch = T.alloc_shared((threads,), "float32")
            count = T.max(0, T.min(Run[segment, 2], Positions[row] - Run[segment, 1] + 1))
            local[0] = -3.402823466e38
            for chunk in T.serial(T.ceildiv(count, threads)):
                token = chunk * threads + lane
                if token < count:
                    local[0] = T.max(local[0], Scores[segment, head, row, token])
            scratch[lane] = local[0]
            T.sync_threads()
            for step in T.unroll(8, explicit=True):
                if lane < (threads >> (step + 1)):
                    scratch[lane] = T.max(scratch[lane], scratch[lane + (threads >> (step + 1))])
                T.sync_threads()
            maximum = scratch[0]
            T.sync_threads()
            local[0] = 0
            for chunk in T.serial(T.ceildiv(length, threads)):
                token = chunk * threads + lane
                if token < length:
                    value = T.if_then_else(
                        token < count, T.exp(Scores[segment, head, row, token] - maximum), 0
                    )
                    Scores[segment, head, row, token] = value
                    local[0] += value
            scratch[lane] = local[0]
            T.sync_threads()
            for step in T.unroll(8, explicit=True):
                if lane < (threads >> (step + 1)):
                    scratch[lane] += scratch[lane + (threads >> (step + 1))]
                T.sync_threads()
            denominator = scratch[0]
            for chunk in T.serial(T.ceildiv(length, threads)):
                token = chunk * threads + lane
                if token < length:
                    Scores[segment, head, row, token] = T.if_then_else(
                        denominator > 0, Scores[segment, head, row, token] / denominator, 0
                    )
            if lane == 0:
                Statistics[segment, row, head, 0] = maximum
                Statistics[segment, row, head, 1] = denominator

    return main


def values(
    rows,
    heads,
    kv_heads,
    width,
    capacity,
    segments,
    length,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
    score_dtype: DType = DType.F32,
):
    head_tile = heads // kv_heads if rows < 8 else 1
    groups, queries = heads // head_tile, rows * head_tile
    native = score_dtype == DType.BF16
    bm, bn, bk = (64 if native and rows >= 256 else 8 if queries < 32 else 32), 64, 32
    pad = 8 if native else 0

    @T.macro
    def tile(a, b, accum, Scores, V, Run, segment, head_group, bx, by, block, guarded):
        for i, j in T.Parallel(bm, bk):
            a[i, j] = 0
            if by * bm + i < queries and (not guarded or block * bk + j < Run[segment, 2]):
                query = by * bm + i
                a[i, j] = Scores[
                    segment,
                    head_group * head_tile + query % head_tile,
                    query // head_tile,
                    block * bk + j,
                ]
        for i, j in T.Parallel(bk, bn):
            b[i, j] = 0
            if (not guarded or block * bk + i < Run[segment, 2]) and bx * bn + j < width:
                b[i, j] = V[
                    Run[segment, 0] + block * bk + i,
                    head_group // (groups // kv_heads),
                    bx * bn + j,
                ]
        T.sync_threads()
        T.gemm(a[:, :bk], b[:, :bn], accum)
        T.sync_threads()

    @T.prim_func
    def main(
        Scores: T.Tensor((segments, heads, rows, length), score_dtype.value),
        V: T.Tensor((capacity, kv_heads, width), dtype.value),
        Run: T.Tensor((segments, 3), "int32"),
        Output: T.Tensor((segments, rows, heads, width), output_dtype.value),
    ):
        with T.Kernel(
            T.ceildiv(width, bn), T.ceildiv(queries, bm), segments * groups, threads=128
        ) as (
            bx,
            by,
            z,
        ):
            segment, head_group = z // groups, z % groups
            a = T.alloc_shared((bm, bk + pad), dtype.value)
            b = T.alloc_shared((bk, bn + pad), dtype.value)
            accum = T.alloc_fragment((bm, bn), "float32")
            result = T.alloc_shared((bm, bn), "float32")
            T.clear(accum)
            for block in T.serial(Run[segment, 2] // bk):
                tile(a, b, accum, Scores, V, Run, segment, head_group, bx, by, block, False)
            if Run[segment, 2] % bk != 0:
                tile(
                    a,
                    b,
                    accum,
                    Scores,
                    V,
                    Run,
                    segment,
                    head_group,
                    bx,
                    by,
                    Run[segment, 2] // bk,
                    True,
                )
            T.copy(accum, result)
            T.sync_threads()
            for i, j in T.Parallel(bm, bn):
                if by * bm + i < queries and bx * bn + j < width:
                    query = by * bm + i
                    Output[
                        segment,
                        query // head_tile,
                        head_group * head_tile + query % head_tile,
                        bx * bn + j,
                    ] = result[i, j]

    return main


def normalize_bf16(rows, heads, segments, length):
    """POC looped softmax; scores become BF16 probabilities in the same storage."""

    @T.prim_func
    def main(
        Scores: T.Tensor((segments, heads, rows, length), "bfloat16"),
        Positions: T.Tensor((rows,), "int32"),
        Run: T.Tensor((segments, 3), "int32"),
        Statistics: T.Tensor((segments, rows, heads, 2), "float32"),
    ):
        with T.Kernel(rows, heads, segments, threads=1024) as (row, head, segment):
            tid = T.get_thread_binding()
            peaks = T.alloc_shared((32,), "float32")
            sums = T.alloc_shared((32,), "float32")
            vals = T.alloc_local((4,), "float32")
            peak = T.alloc_local((1,), "float32")
            total = T.alloc_local((1,), "float32")
            previous = T.alloc_local((1,), "float32")
            count = T.max(0, T.min(Run[segment, 2], Positions[row] - Run[segment, 1] + 1))
            peak[0] = -3.4028234663852886e38
            total[0] = 0
            for block in T.serial(T.ceildiv(count, 4096)):
                offset = block * 4096 + tid * 4
                for j in T.unroll(4, explicit=True):
                    vals[j] = T.call_pure_extern("float32", "as_type<float>", T.uint32(0xFF800000))
                    if offset + j < count:
                        vals[j] = Scores[segment, head, row, offset + j].astype("float32")
                previous[0] = peak[0]
                for j in T.unroll(4, explicit=True):
                    peak[0] = T.max(peak[0], vals[j])
                total[0] *= T.call_pure_extern("float32", "metal::fast::exp", previous[0] - peak[0])
                for j in T.unroll(4, explicit=True):
                    total[0] += T.call_pure_extern("float32", "metal::fast::exp", vals[j] - peak[0])
            previous[0] = peak[0]
            peak[0] = T.call_extern("float32", "simd_max", peak[0])
            total[0] *= T.call_pure_extern("float32", "metal::fast::exp", previous[0] - peak[0])
            total[0] = T.call_extern("float32", "simd_sum", total[0])
            previous[0] = peak[0]
            if tid % 32 == 0:
                peaks[tid // 32] = peak[0]
            T.sync_threads()
            peak[0] = T.call_extern("float32", "simd_max", peaks[tid % 32])
            total[0] *= T.call_pure_extern("float32", "metal::fast::exp", previous[0] - peak[0])
            if tid % 32 == 0:
                sums[tid // 32] = total[0]
            T.sync_threads()
            total[0] = T.call_extern("float32", "simd_sum", sums[tid % 32])
            if tid == 0:
                Statistics[segment, row, head, 0] = peak[0]
                Statistics[segment, row, head, 1] = total[0]
            for block in T.serial(T.ceildiv(Run[segment, 2], 4096)):
                for j in T.unroll(4, explicit=True):
                    index = block * 4096 + tid * 4 + j
                    if index < Run[segment, 2]:
                        Scores[segment, head, row, index] = T.if_then_else(
                            index < count and total[0] > 0,
                            T.call_pure_extern(
                                "float32",
                                "metal::fast::exp",
                                Scores[segment, head, row, index].astype("float32") - peak[0],
                            )
                            / total[0],
                            T.float32(0),
                        )

    return main
