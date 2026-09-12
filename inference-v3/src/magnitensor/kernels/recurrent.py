"""Sequence-specialized recurrent preparation schedules."""

from __future__ import annotations

import math
from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..tensor.graph import Graph
from ..tensor.types import TensorSpec


@T.macro
def _channel_parallel_prepare(
    projected,
    convolution,
    previous,
    offsets,
    alpha,
    beta_input,
    rate,
    bias,
    query,
    key,
    value,
    beta,
    decay,
    following,
    batch,
    rows,
    key_heads,
    value_heads,
    width,
    history,
    epsilon,
    query_gain,
    threads,
    dtype,
):
    heads = 2 * key_heads + value_heads
    with T.Kernel(heads, batch, threads=threads) as (head, sequence):
        lane = T.get_thread_binding(0)
        squares = T.alloc_shared((threads,), "float32")
        convolved = T.alloc_local((1,), "float32")
        packed = head * width + lane
        count = offsets[sequence + 1] - offsets[sequence]
        for step in T.serial(rows):
            if step < count:
                row = offsets[sequence] + step
                convolved[0] = 0.0
                if lane < width:
                    for time in T.serial(history):
                        source_step = step - history + time
                        convolved[0] += (
                            T.if_then_else(
                                source_step < 0,
                                previous[sequence, packed, source_step + history],
                                projected[offsets[sequence] + source_step, packed],
                            )
                            * convolution[packed, time]
                        )
                    convolved[0] += projected[row, packed] * convolution[packed, history]
                    convolved[0] *= T.sigmoid(convolved[0])
                squares[lane] = T.if_then_else(
                    lane < width and head < 2 * key_heads,
                    convolved[0] * convolved[0],
                    0.0,
                )
                T.sync_threads()
                for reduction in T.unroll(int(math.log2(threads))):
                    distance = threads >> (reduction + 1)
                    if lane < distance:
                        squares[lane] += squares[lane + distance]
                    T.sync_threads()
                if lane < width:
                    if head < key_heads:
                        query[row, head, lane] = T.cast(
                            convolved[0] * T.rsqrt(squares[0] + epsilon) * query_gain,
                            dtype,
                        )
                    elif head < 2 * key_heads:
                        key[row, head - key_heads, lane] = T.cast(
                            convolved[0] * T.rsqrt(squares[0] + epsilon), dtype
                        )
                    else:
                        value[row, head - 2 * key_heads, lane] = T.cast(convolved[0], dtype)
                if lane == 0 and head >= 2 * key_heads:
                    value_head = head - 2 * key_heads
                    beta[row, value_head] = T.cast(
                        T.sigmoid(beta_input[row, value_head]), dtype
                    )
                    shifted = T.cast(alpha[row, value_head], "float32") + T.cast(
                        bias[value_head], "float32"
                    )
                    softplus = T.max(shifted, 0.0) + T.log(1 + T.exp(-T.abs(shifted)))
                    decay[row, value_head] = T.exp(
                        T.cast(rate[value_head], "float32") * softplus
                    )
                T.sync_threads()
        if lane < width:
            for time in T.serial(history):
                source_step = count - history + time
                following[sequence, packed, time] = T.if_then_else(
                    source_step < 0,
                    previous[sequence, packed, source_step + history],
                    projected[offsets[sequence] + source_step, packed],
                )


class _RecurrentPrepareEmitter:
    def __init__(self, specs: tuple[TensorSpec, ...], attrs, threads: int):
        self.specs, self.attrs, self.threads = specs, attrs, threads

    def __call__(self, operands: tuple[Any, ...]) -> None:
        batch, _, history = cast(tuple[int, int, int], self.specs[2].shape)
        rows = cast(int, self.specs[0].shape[0])
        attrs = self.attrs
        _channel_parallel_prepare(
            operands[0],
            operands[1],
            operands[2],
            operands[7],
            operands[3],
            operands[4],
            operands[5],
            operands[6],
            operands[8],
            operands[9],
            operands[10],
            operands[11],
            operands[12],
            operands[13],
            batch,
            rows,
            attrs["key_heads"],
            attrs["value_heads"],
            attrs["width"],
            history,
            attrs["epsilon"],
            1.0 / math.sqrt(attrs["width"]),
            self.threads,
            self.specs[0].dtype.value,
        )


class RecurrentPrepareRule:
    name = "channel-parallel-recurrent-prepare"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if (
            node.operation != "recurrent_prepare"
            or "shared" not in context.capabilities.memory_scopes
        ):
            return ()
        specs = tuple(graph.values[value].spec for value in (*node.inputs, *node.outputs))
        if any(not spec.static for spec in specs):
            return ()
        width = node.attributes["width"]
        threads = 1 << (width - 1).bit_length()
        if threads > context.capabilities.threads_per_group:
            return ()
        moved = sum(spec.storage_nbytes for spec in specs)
        return (
            Candidate(
                f"recurrent_prepare.channel-parallel@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                _RecurrentPrepareEmitter(specs, node.attributes, threads),
                4e-7 + moved / 200e9,
                priority=30,
            ),
        )
