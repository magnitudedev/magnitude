"""Score and reduce one vocabulary tile per group."""

import math

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.launch import serial
from magnitude_engine.kernels.random import philox


@T.macro
def better(score, index, candidate, candidate_index):
    return candidate > score or (candidate == score and candidate_index < index)


def shared_reduction(threads):
    @T.macro
    def reduce(Scores, Indices, Invalid, lane):
        for step in T.unroll(int(math.log2(threads))):
            distance = threads >> (step + 1)
            if lane < distance:
                if better(
                    Scores[lane], Indices[lane], Scores[lane + distance], Indices[lane + distance]
                ):
                    Scores[lane] = Scores[lane + distance]
                    Indices[lane] = Indices[lane + distance]
                Invalid[lane] |= Invalid[lane + distance]
            T.sync_threads()

    return reduce


def select_tiles(
    rows: int,
    vocabulary: int,
    *,
    capability: Capability,
    tile: int = 1024,
    threads: int = 128,
):
    if min(rows, vocabulary, tile, threads) <= 0 or threads & (threads - 1):
        raise ValueError("invalid selection geometry")
    if vocabulary > 0x7FFFFFFF:
        raise ValueError("vocabulary exceeds token ID range")
    tiles = math.ceil(vocabulary / tile)
    cpu = serial(capability)
    reduce = shared_reduction(threads)

    @T.macro
    def consume(Logits, Draws, row, token, Score, Index, Invalid):
        raw = Logits[row, token]
        negative_infinity = T.reinterpret(T.uint32(0xFF800000), "float32")
        bits = T.reinterpret(raw, "uint32")
        if (bits & T.uint32(0x7FFFFFFF)) > T.uint32(0x7F800000) or bits == T.uint32(0x7F800000):
            Invalid[0] = 1
        elif raw > negative_infinity:
            value = T.alloc_local((1,), "float32")
            value[0] = raw
            if Draws[row, 0] == 1:
                word, _, _, _ = philox(
                    token.astype("uint32"),
                    Draws[row, 3],
                    Draws[row, 4],
                    Draws[row, 5],
                    Draws[row, 1],
                    Draws[row, 2],
                )
                # 2**23 equally spaced midpoints are exactly representable in
                # FP32 and strictly inside (0, 1), including the endpoint cases.
                uniform = ((word >> 9).astype("float32") + T.float32(0.5)) * T.float32(2**-23)
                value[0] -= T.log(-T.log(uniform))
            if better(Score[0], Index[0], value[0], token):
                Score[0], Index[0] = value[0], token

    @T.prim_func
    def main(
        Logits: T.Tensor((rows, vocabulary), "float32"),
        Draws: T.Tensor((rows, 6), "uint32"),
        Values: T.Tensor((rows, tiles), "float32"),
        Indices: T.Tensor((rows, tiles), "int32"),
        Invalid: T.Tensor((rows, tiles), "int32"),
    ):
        if cpu:
            for row, block in T.Parallel(rows, tiles):
                score = T.alloc_local((1,), "float32")
                index = T.alloc_local((1,), "int32")
                invalid = T.alloc_local((1,), "int32")
                score[0] = T.reinterpret(T.uint32(0xFF800000), "float32")
                index[0], invalid[0] = 0x7FFFFFFF, 0
                for col in T.serial(tile):
                    token = block * tile + col
                    if token < vocabulary:
                        consume(Logits, Draws, row, token, score, index, invalid)
                Values[row, block], Indices[row, block], Invalid[row, block] = (
                    score[0],
                    index[0],
                    invalid[0],
                )
        else:
            with T.Kernel(tiles, rows, threads=threads) as (block, row):
                lane = T.get_thread_binding(0)
                score = T.alloc_local((1,), "float32")
                index = T.alloc_local((1,), "int32")
                invalid = T.alloc_local((1,), "int32")
                scores = T.alloc_shared((threads,), "float32")
                indices = T.alloc_shared((threads,), "int32")
                flags = T.alloc_shared((threads,), "int32")
                score[0] = T.reinterpret(T.uint32(0xFF800000), "float32")
                index[0], invalid[0] = 0x7FFFFFFF, 0
                for col in T.serial(T.ceildiv(tile, threads)):
                    token = block * tile + col * threads + lane
                    if token < vocabulary and col * threads + lane < tile:
                        consume(Logits, Draws, row, token, score, index, invalid)
                scores[lane], indices[lane], flags[lane] = score[0], index[0], invalid[0]
                T.sync_threads()
                reduce(scores, indices, flags, lane)
                if lane == 0:
                    Values[row, block], Indices[row, block], Invalid[row, block] = (
                        scores[0],
                        indices[0],
                        flags[0],
                    )

    return main
