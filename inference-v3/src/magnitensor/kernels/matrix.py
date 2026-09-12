"""Capability-selected dense matrix schedules authored directly in TileLang."""

from __future__ import annotations

from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..compiler.tuning import TuningKey
from ..representations import Dense
from ..tensor.graph import Graph


@T.macro
def _fragment_matrix(
    source,
    weight,
    bias,
    result,
    operation,
    m,
    n,
    k_size,
    dtype,
    output_dtype,
    threads,
    bm,
    bn,
    bk,
    has_bias,
):
    with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
        left = T.alloc_shared((bm, bk), dtype)
        right = T.alloc_shared((bn, bk), dtype)
        accum = T.alloc_fragment((bm, bn), "float32")
        T.clear(accum)
        for block in T.serial(T.ceildiv(k_size, bk)):
            for i, k in T.Parallel(bm, bk):
                left[i, k] = T.if_then_else(
                    by * bm + i < m and block * bk + k < k_size,
                    source[by * bm + i, block * bk + k],
                    0,
                )
            for j, k in T.Parallel(bn, bk):
                if operation == "linear":
                    value = weight[bx * bn + j, block * bk + k]
                else:
                    value = weight[block * bk + k, bx * bn + j]
                right[j, k] = T.if_then_else(bx * bn + j < n and block * bk + k < k_size, value, 0)
            T.sync_threads()
            T.gemm(left, right, accum, transpose_B=True)
            T.sync_threads()
        for i, j in T.Parallel(bm, bn):
            if by * bm + i < m and bx * bn + j < n:
                value = accum[i, j]
                if has_bias:
                    value += T.cast(bias[bx * bn + j], "float32")
                result[by * bm + i, bx * bn + j] = T.cast(value, output_dtype)


class _TiledMatrixEmitter:
    def __init__(
        self,
        operation: str,
        m: int,
        n: int,
        k: int,
        dtype: str,
        output: str,
        threads: int,
        tile: tuple[int, int, int],
        bias: bool,
    ):
        self.operation = operation
        self.m, self.n, self.k = m, n, k
        self.dtype, self.output = dtype, output
        self.threads, self.tile, self.bias = threads, tile, bias

    def __call__(self, operands: tuple[Any, ...]) -> None:
        source, weight = operands[:2]
        bias = operands[2] if self.bias else source
        result = operands[3] if self.bias else operands[2]
        _fragment_matrix(
            source,
            weight,
            bias,
            result,
            self.operation,
            self.m,
            self.n,
            self.k,
            self.dtype,
            self.output,
            self.threads,
            *self.tile,
            self.bias,
        )


class DenseMatrixRule:
    name = "fragment-tiled-dense-matrix"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation not in {"matmul", "linear"}:
            return ()
        specs = tuple(graph.values[value].spec for value in node.inputs)
        left, right = specs[:2]
        if (
            not left.static
            or not right.static
            or right.representation not in (None, Dense(right.dtype))
            or left.rank != 2
            or right.rank != 2
        ):
            return ()
        instruction = next(
            (
                item
                for item in context.capabilities.matrix_instructions
                if item.input_dtype == left.dtype
            ),
            None,
        )
        if instruction is None:
            return ()
        output = graph.values[node.outputs[0]].spec
        n = cast(int, output.shape[-1])
        k = cast(int, left.shape[-1])
        m = left.elements // k
        bm, bn, bk = instruction.m * 4, instruction.n * 4, instruction.k * 2
        shared = (bm * bk + bn * bk) * left.dtype.itemsize
        if shared > context.capabilities.shared_memory_bytes:
            bm, bn, bk = instruction.m * 2, instruction.n * 2, instruction.k
        threads = min(
            context.capabilities.threads_per_group, context.capabilities.subgroup_width * 4
        )
        inputs = tuple(node.inputs)
        outputs = tuple(node.outputs)
        geometry = (m, n, k, bm, bn, bk, threads)
        key = TuningKey(
            node.operation,
            geometry,
            ("dense",),
            context.precision,
            context.capabilities.fingerprint,
            context.compiler_identity,
        )
        return (
            Candidate(
                f"{node.operation}.fragment-tiled@{root}",
                frozenset({root}),
                inputs,
                outputs,
                _TiledMatrixEmitter(
                    node.operation,
                    m,
                    n,
                    k,
                    left.dtype.value,
                    output.dtype.value,
                    threads,
                    (bm, bn, bk),
                    len(inputs) == 3,
                ),
                5e-7 + (2 * m * n * k) / 5e12,
                tuning_key=key,
                priority=10,
            ),
        )
