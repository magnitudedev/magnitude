"""Register online softmax with KV loads shared across grouped query heads.

One SIMD scans a history span and owns every query head attached to its KV head.
Its K/V values are loaded once and reused from registers. Two SIMDs cover separate
history spans, then merge locally. Physical range ownership remains external.
"""

import tilelang.language as T

from magnitude_engine.platform.execution import DType


def run_attention(
    rows,
    query_heads,
    kv_heads,
    width,
    capacity,
    *,
    segments,
    segment_capacity,
    partitions,
    dtype: DType = DType.BF16,
    output_dtype: DType = DType.F32,
):
    if width != 256 or query_heads % kv_heads or dtype != DType.BF16:
        raise ValueError("online Metal decode requires BF16 256-coordinate heads")
    if min(rows, query_heads, kv_heads, capacity, segments, segment_capacity, partitions) <= 0:
        raise ValueError("invalid online attention geometry")
    if segment_capacity > capacity:
        raise ValueError("visible segment exceeds physical capacity")
    HG = query_heads // kv_heads
    NC = min(2, max(1, 16384 // (HG * 258 * 4)))
    if HG * 258 * 4 > 16384:
        raise ValueError("query grouping exceeds bounded online scratch")
    span = (segment_capacity + partitions - 1) // partitions
    subspan = (span + NC - 1) // NC

    @T.prim_func
    def main(
        A: T.Tensor((rows, query_heads, 256), "bfloat16"),
        B: T.Tensor((capacity, kv_heads, 256), "bfloat16"),
        C: T.Tensor((capacity, kv_heads, 256), "bfloat16"),
        D: T.Tensor((rows,), "int32"),
        Run: T.Tensor((segments, 3), "int32"),
        E: T.Tensor((segments * partitions, rows, query_heads, 256), output_dtype.value),
        Stats: T.Tensor((segments * partitions, rows, query_heads, 2), "float32"),
    ):
        with T.Kernel(kv_heads, segments * partitions, rows, threads=32 * NC) as (head, block, row):
            tid = T.get_thread_binding()
            lane = tid % 32
            chunk = tid // 32
            query = T.alloc_local((HG, 8), "float32")
            acc = T.alloc_local((HG, 8), "float32")
            key = T.alloc_local((8,), "float32")
            value = T.alloc_local((8,), "float32")
            peak = T.alloc_local((HG,), "float32")
            total = T.alloc_local((HG,), "float32")
            dot = T.alloc_local((1,), "float32")
            scratch = T.alloc_shared((NC, HG, 258), "float32")
            for h in T.unroll(HG, explicit=True):
                peak[h] = T.call_pure_extern("float32", "as_type<float>", T.uint32(0xFF800000))
                total[h] = 0
                for i in T.unroll(8, explicit=True):
                    query[h, i] = A[row, head * HG + h, lane * 8 + i].astype("float32")
                    acc[h, i] = 0
            segment = block // partitions
            first = block % partitions * span
            count = T.max(
                0, T.min(T.min(Run[segment, 2], segment_capacity), D[row] + 1 - Run[segment, 1])
            )
            begin = first + chunk * subspan
            end = T.min(count, T.min(first + span, begin + subspan))
            physical_start = T.if_then_else(segments == 1, 0, Run[segment, 0])
            for token in T.serial(begin, T.max(begin, end)):
                physical = physical_start + token
                for i in T.unroll(8, explicit=True):
                    key[i] = B[physical, head, lane * 8 + i].astype("float32")
                    value[i] = C[physical, head, lane * 8 + i].astype("float32")
                for h in T.unroll(HG, explicit=True):
                    dot[0] = 0
                    for i in T.unroll(8, explicit=True):
                        dot[0] += query[h, i] * key[i]
                    score = T.call_extern("float32", "simd_sum", dot[0]) * T.float32(1 / 16)
                    maximum = T.max(peak[h], score)
                    previous = T.call_pure_extern("float32", "metal::fast::exp", peak[h] - maximum)
                    weight = T.call_pure_extern("float32", "metal::fast::exp", score - maximum)
                    total[h] = total[h] * previous + weight
                    for i in T.unroll(8, explicit=True):
                        acc[h, i] = acc[h, i] * previous + weight * value[i]
                    peak[h] = maximum
            for h in T.unroll(HG, explicit=True):
                for i in T.unroll(8, explicit=True):
                    scratch[chunk, h, lane * 8 + i] = acc[h, i]
                if lane == 0:
                    scratch[chunk, h, 256] = peak[h]
                    scratch[chunk, h, 257] = total[h]
            T.sync_threads()
            if chunk == 0:
                for h in T.unroll(HG, explicit=True):
                    peak[h] = T.call_pure_extern("float32", "as_type<float>", T.uint32(0xFF800000))
                    total[h] = 0
                    for c in T.serial(NC):
                        peak[h] = T.max(peak[h], scratch[c, h, 256])
                    for i in T.unroll(8, explicit=True):
                        acc[h, i] = 0
                    for c in T.serial(NC):
                        weight = T.if_then_else(
                            scratch[c, h, 257] > 0,
                            T.call_pure_extern(
                                "float32", "metal::fast::exp", scratch[c, h, 256] - peak[h]
                            ),
                            T.float32(0),
                        )
                        total[h] += weight * scratch[c, h, 257]
                        for i in T.unroll(8, explicit=True):
                            acc[h, i] += weight * scratch[c, h, lane * 8 + i]
                    for i in T.unroll(8, explicit=True):
                        E[block, row, head * HG + h, lane * 8 + i] = T.if_then_else(
                            total[h] > 0, acc[h, i] / total[h], T.float32(0)
                        )
                    if lane == 0:
                        Stats[block, row, head * HG + h, 0] = peak[h]
                        Stats[block, row, head * HG + h, 1] = total[h]

    return main
