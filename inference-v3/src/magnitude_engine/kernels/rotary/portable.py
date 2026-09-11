"""Qwen query/key normalization and three-axis partial rotary preparation."""

import math
import struct

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.launch import serial
from magnitude_engine.kernels.precision import Precision, Rounding
from magnitude_engine.platform.execution import DType


def prepare_attention(
    rows: int,
    query_heads: int,
    kv_heads: int,
    width: int,
    rotary_width: int,
    base: float,
    sections: tuple[int, int, int, int],
    epsilon: float,
    *,
    capability: Capability,
    precision: Precision,
    threads: int = 128,
    dtype: DType = DType.F32,
):
    if min(rows, query_heads, kv_heads, width, rotary_width, threads) <= 0:
        raise ValueError("invalid attention preparation geometry")
    if rotary_width % 2 or rotary_width > width or threads & (threads - 1):
        raise ValueError("invalid rotary/reduction width")
    if not math.isfinite(base) or base <= 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("invalid rotary/normalization constants")
    if len(sections) != 4 or any(n < 0 for n in sections) or sum(sections) * 2 != rotary_width:
        raise ValueError("invalid rotary axis sections")
    cpu = serial(capability)
    native_rounding = precision.rounding == Rounding.NATIVE_BF16
    half = rotary_width // 2
    # These are geometry-dependent compile-time constants, rounded once to the
    # model's FP32 frequency precision. No activation or weight data is computed
    # on the host, and every position-dependent operation remains in TileLang.
    frequencies = tuple(
        struct.unpack("I", struct.pack("f", 1 / base ** (i / half)))[0] for i in range(half)
    )

    def frequency(index):
        # Metal's current code generator prints float literals with only seven
        # significant digits. Select integer bit patterns before reinterpretation
        # so frequency error cannot grow into a long-context phase error.
        value = T.uint32(frequencies[0])
        for i, constant in enumerate(frequencies[1:], 1):
            value = T.if_then_else(index == i, T.uint32(constant), value)
        return T.reinterpret(value, "float32")

    @T.macro
    def raw(QueryGate, Keys, row, head, d):
        return T.if_then_else(
            head < query_heads,
            QueryGate[row, head * 2 * width + d],
            Keys[row, (head - query_heads) * width + d],
        ).astype("float32")

    @T.macro
    def norm_weight(QueryNorm, KeyNorm, head, d):
        return T.if_then_else(head < query_heads, QueryNorm[d], KeyNorm[d])

    @T.macro
    def normalize(QueryGate, Keys, QueryNorm, KeyNorm, row, head, d, inverse):
        scaled = raw(QueryGate, Keys, row, head, d) * inverse
        if native_rounding:
            return (
                (
                    scaled.astype("bfloat16").astype("float32")
                    * norm_weight(QueryNorm, KeyNorm, head, d)
                )
                .astype("bfloat16")
                .astype("float32")
            )
        return scaled * norm_weight(QueryNorm, KeyNorm, head, d)

    @T.macro
    def finish(
        QueryGate,
        Keys,
        QueryNorm,
        KeyNorm,
        Coordinates,
        QueryOut,
        KeyOut,
        Gate,
        row,
        head,
        d,
        inverse,
    ):
        normalized = normalize(QueryGate, Keys, QueryNorm, KeyNorm, row, head, d, inverse)
        value = T.alloc_local((1,), "float32")
        value[0] = normalized
        if d < rotary_width:
            index = d % half
            axis = T.if_then_else(
                index % 3 == 1 and index < sections[1] * 3,
                1,
                T.if_then_else(index % 3 == 2 and index < sections[2] * 3, 2, 0),
            )
            angle = Coordinates[row, axis].astype("float32") * frequency(index)
            pair = (d + half) % rotary_width
            paired = normalize(QueryGate, Keys, QueryNorm, KeyNorm, row, head, pair, inverse)
            signed = T.if_then_else(d < half, -paired, paired)
            value[0] = normalized * T.cos(angle) + signed * T.sin(angle)
        if head < query_heads:
            QueryOut[row, head, d] = value[0]
            Gate[row, head, d] = QueryGate[row, head * 2 * width + width + d]
        else:
            KeyOut[row, head - query_heads, d] = value[0]

    @T.prim_func
    def main(
        QueryGate: T.Tensor((rows, query_heads * 2 * width), dtype.value),
        Keys: T.Tensor((rows, kv_heads * width), dtype.value),
        QueryNorm: T.Tensor((width,), "float32"),
        KeyNorm: T.Tensor((width,), "float32"),
        Coordinates: T.Tensor((rows, 3), "int32"),
        QueryOut: T.Tensor((rows, query_heads, width), dtype.value),
        KeyOut: T.Tensor((rows, kv_heads, width), dtype.value),
        Gate: T.Tensor((rows, query_heads, width), dtype.value),
    ):
        if cpu:
            for row, head in T.Parallel(rows, query_heads + kv_heads):
                squares = T.alloc_local((1,), "float32")
                squares[0] = 0
                for d in T.serial(width):
                    x = raw(QueryGate, Keys, row, head, d)
                    squares[0] += x * x
                inverse = T.rsqrt(squares[0] / width + epsilon)
                for d in T.serial(width):
                    finish(
                        QueryGate,
                        Keys,
                        QueryNorm,
                        KeyNorm,
                        Coordinates,
                        QueryOut,
                        KeyOut,
                        Gate,
                        row,
                        head,
                        d,
                        inverse,
                    )
        else:
            with T.Kernel(query_heads + kv_heads, rows, threads=threads) as (head, row):
                lane = T.get_thread_binding(0)
                squares = T.alloc_local((1,), "float32")
                shared = T.alloc_shared((threads,), "float32")
                squares[0] = 0
                for chunk in T.serial(T.ceildiv(width, threads)):
                    d = chunk * threads + lane
                    if d < width:
                        x = raw(QueryGate, Keys, row, head, d)
                        squares[0] += x * x
                shared[lane] = squares[0]
                T.sync_threads()
                for step in T.unroll(int(math.log2(threads))):
                    if lane < (threads >> (step + 1)):
                        shared[lane] += shared[lane + (threads >> (step + 1))]
                    T.sync_threads()
                inverse = T.rsqrt(shared[0] / width + epsilon)
                for chunk in T.serial(T.ceildiv(width, threads)):
                    d = chunk * threads + lane
                    if d < width:
                        finish(
                            QueryGate,
                            Keys,
                            QueryNorm,
                            KeyNorm,
                            Coordinates,
                            QueryOut,
                            KeyOut,
                            Gate,
                            row,
                            head,
                            d,
                            inverse,
                        )

    return main
