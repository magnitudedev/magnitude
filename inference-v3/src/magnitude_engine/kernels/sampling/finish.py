"""Reduce the per-tile winners into one selected token."""

import math

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.launch import serial


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


def finish_selection(rows: int, tiles: int, *, capability: Capability, threads: int = 128):
    if min(rows, tiles, threads) <= 0 or threads & (threads - 1):
        raise ValueError("invalid selection reduction geometry")
    cpu = serial(capability)
    reduce = shared_reduction(threads)

    @T.macro
    def finish(Output, row, index, invalid):
        # Output columns are token ID and status: success / empty / invalid.
        status = T.if_then_else(invalid != 0, 2, T.if_then_else(index == 0x7FFFFFFF, 1, 0))
        Output[row, 0] = T.if_then_else(status == 0, index, -1)
        Output[row, 1] = status

    @T.prim_func
    def main(
        Values: T.Tensor((rows, tiles), "float32"),
        Indices: T.Tensor((rows, tiles), "int32"),
        Invalid: T.Tensor((rows, tiles), "int32"),
        Output: T.Tensor((rows, 2), "int32"),
    ):
        if cpu:
            for row in T.Parallel(rows):
                score = T.alloc_local((1,), "float32")
                index = T.alloc_local((1,), "int32")
                invalid = T.alloc_local((1,), "int32")
                score[0] = T.reinterpret(T.uint32(0xFF800000), "float32")
                index[0], invalid[0] = 0x7FFFFFFF, 0
                for block in T.serial(tiles):
                    if better(score[0], index[0], Values[row, block], Indices[row, block]):
                        score[0], index[0] = Values[row, block], Indices[row, block]
                    invalid[0] |= Invalid[row, block]
                finish(Output, row, index[0], invalid[0])
        else:
            with T.Kernel(rows, threads=threads) as row:
                lane = T.get_thread_binding(0)
                score = T.alloc_local((1,), "float32")
                index = T.alloc_local((1,), "int32")
                invalid = T.alloc_local((1,), "int32")
                scores = T.alloc_shared((threads,), "float32")
                indices = T.alloc_shared((threads,), "int32")
                flags = T.alloc_shared((threads,), "int32")
                score[0] = T.reinterpret(T.uint32(0xFF800000), "float32")
                index[0], invalid[0] = 0x7FFFFFFF, 0
                for chunk in T.serial(T.ceildiv(tiles, threads)):
                    block = chunk * threads + lane
                    if block < tiles:
                        if better(score[0], index[0], Values[row, block], Indices[row, block]):
                            score[0], index[0] = Values[row, block], Indices[row, block]
                        invalid[0] |= Invalid[row, block]
                scores[lane], indices[lane], flags[lane] = score[0], index[0], invalid[0]
                T.sync_threads()
                reduce(scores, indices, flags, lane)
                if lane == 0:
                    finish(Output, row, indices[0], flags[0])

    return main
