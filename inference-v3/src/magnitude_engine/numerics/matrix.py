"""Tiled encoded contraction for prefill; no full-size dequantized weights."""

import tilelang.language as T

from magnitude_engine.artifacts.gguf import Encoding
from magnitude_engine.numerics.encoded_groups import interpretation
from magnitude_engine.numerics.encoded_layout import EncodedLayout, storage_bytes
from magnitude_engine.platform.execution import DType


def projection(
    rows: int,
    outputs: int,
    inputs: int,
    encoding: Encoding,
    *,
    row_tile: int = 64,
    output_tile: int = 64,
    reduction_tile: int = 32,
    threads: int = 128,
    partitions: int = 1,
    dtype: DType = DType.F32,
    output_dtype: DType = DType.F32,
    layout: EncodedLayout = EncodedLayout.GGUF,
):
    if min(rows, outputs, inputs, row_tile, output_tile, reduction_tile, threads, partitions) <= 0:
        raise ValueError("matrix projection geometry must be positive")
    if inputs % encoding.block_elements or any(
        n % 8 for n in (row_tile, output_tile, reduction_tile)
    ):
        raise ValueError("invalid encoded matrix tile")
    parameters, payload = interpretation(encoding, layout, outputs * inputs)
    size = storage_bytes(outputs * inputs, encoding, layout) // 2
    blocks_per_partition = (inputs + reduction_tile * partitions - 1) // (
        reduction_tile * partitions
    )

    @T.prim_func
    def main(
        A: T.Tensor((rows, inputs), dtype.value),
        B: T.Tensor((size,), "uint16"),
        C: T.Tensor((partitions * rows, outputs), output_dtype.value),
    ):
        with T.Kernel(
            T.ceildiv(outputs, output_tile), T.ceildiv(rows, row_tile), partitions, threads=threads
        ) as (bx, by, part):
            x = T.alloc_shared((row_tile, reduction_tile), dtype.value)
            w = T.alloc_shared((output_tile, reduction_tile), dtype.value)
            result = T.alloc_shared((row_tile, output_tile), "float32")
            accum = T.alloc_fragment((row_tile, output_tile), "float32")
            element = T.alloc_local((1,), "float32")
            T.clear(accum)
            for step in T.serial(blocks_per_partition):
                block = part * blocks_per_partition + step
                for i, k in T.Parallel(row_tile, reduction_tile):
                    if by * row_tile + i < rows and block * reduction_tile + k < inputs:
                        x[i, k] = A[by * row_tile + i, block * reduction_tile + k]
                    else:
                        x[i, k] = 0
                # A worker consumes an aligned eight-coordinate group, sharing
                # its scale/header interpretation across those payload elements.
                # The decoded tile is then reused across all query rows by GEMM.
                for packet in T.serial(T.ceildiv(output_tile * reduction_tile, threads * 8)):
                    index = (packet * threads + T.get_thread_binding()) * 8
                    j, k0 = index // reduction_tile, index % reduction_tile
                    if j < output_tile:
                        if bx * output_tile + j < outputs and block * reduction_tile + k0 < inputs:
                            first = (bx * output_tile + j) * inputs + block * reduction_tile + k0
                            scale, bias = parameters(B, first)
                            for offset in T.unroll(8, explicit=True):
                                # Publish once. A clear followed by an overwrite
                                # makes the compiler insert a barrier per element.
                                element[0] = 0
                                if block * reduction_tile + k0 + offset < inputs:
                                    element[0] = scale * payload(B, first + offset) + bias
                                w[j, k0 + offset] = element[0]
                        else:
                            for offset in T.unroll(8, explicit=True):
                                w[j, k0 + offset] = 0
                T.sync_threads()
                T.gemm(x, w, accum, transpose_B=True)
                T.sync_threads()
            T.sync_threads()
            T.copy(accum, result)
            T.sync_threads()
            for i, j in T.Parallel(row_tile, output_tile):
                if by * row_tile + i < rows and bx * output_tile + j < outputs:
                    C[part * rows + by * row_tile + i, bx * output_tile + j] = result[i, j]

    return main


def merge_partitions(rows: int, outputs: int, partitions: int, output_dtype: DType = DType.F32):
    @T.prim_func
    def main(
        A: T.Tensor((partitions * rows, outputs), "float32"),
        B: T.Tensor((rows, outputs), output_dtype.value),
    ):
        with T.Kernel(T.ceildiv(rows * outputs, 128), threads=128) as block:
            index = block * 128 + T.get_thread_binding()
            total = T.alloc_local((1,), "float32")
            total[0] = 0
            if index < rows * outputs:
                for part in T.serial(partitions):
                    total[0] += A[part * rows + index // outputs, index % outputs]
                B[index // outputs, index % outputs] = total[0]

    return main
