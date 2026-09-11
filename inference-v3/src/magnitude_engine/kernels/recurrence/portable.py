"""Gated delta recurrence over explicit previous and next state operands.

This baseline uses the exact sequential equations with one final state write.
It avoids host submission and state materialization per token; matrix-based
chunk realizations can replace it beneath the same operation contract.
"""

import math

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.launch import serial
from magnitude_engine.kernels.semantics import HeadMapping
from magnitude_engine.platform.execution import DType


def delta_sequence(
    batch: int,
    steps: int,
    key_heads: int,
    value_heads: int,
    key_width: int,
    value_width: int,
    mapping: HeadMapping,
    *,
    capability: Capability,
    lanes: int = 32,
    output_tile: int = 4,
    dtype: DType = DType.F32,
):
    """Scan uniformly packed sequences, retaining private state across their steps.

    This baseline uses the exact sequential equations with one final state write.
    It avoids host submission/state materialization per token. Matrix-based chunk
    realizations can replace this scan beneath the same operation contract.
    """
    if min(batch, steps, key_heads, value_heads, key_width, value_width, lanes, output_tile) <= 0:
        raise ValueError("delta geometry must be positive")
    if value_heads % key_heads or lanes & (lanes - 1) or not isinstance(mapping, HeadMapping):
        raise ValueError("invalid delta head mapping or reduction tile")
    cpu = serial(capability)

    @T.macro
    def key_head(h):
        return h % key_heads if mapping == HeadMapping.TILED else h // (value_heads // key_heads)

    @T.prim_func
    def main(
        Q: T.Tensor((batch * steps, key_heads, key_width), dtype.value),
        K: T.Tensor((batch * steps, key_heads, key_width), dtype.value),
        V: T.Tensor((batch * steps, value_heads, value_width), dtype.value),
        Decay: T.Tensor((batch * steps, value_heads), "float32"),
        Beta: T.Tensor((batch * steps, value_heads), dtype.value),
        Previous: T.Tensor((batch, value_heads, value_width, key_width), "float32"),
        Next: T.Tensor((batch, value_heads, value_width, key_width), "float32"),
        Output: T.Tensor((batch * steps, value_heads, value_width), dtype.value),
    ):
        if cpu:
            for sequence, h, channel in T.Parallel(batch, value_heads, value_width):
                state = T.alloc_local((key_width,), "float32")
                remembered = T.alloc_local((1,), "float32")
                answer = T.alloc_local((1,), "float32")
                for k in T.serial(key_width):
                    state[k] = Previous[sequence, h, channel, k]
                for step in T.serial(steps):
                    row = sequence * steps + step
                    remembered[0] = 0
                    answer[0] = 0
                    for k in T.serial(key_width):
                        state[k] *= Decay[row, h]
                        remembered[0] += state[k] * K[row, key_head(h), k]
                    residual = (V[row, h, channel] - remembered[0]) * Beta[row, h]
                    for k in T.serial(key_width):
                        state[k] += residual * K[row, key_head(h), k]
                        answer[0] += state[k] * Q[row, key_head(h), k]
                    Output[row, h, channel] = answer[0]
                for k in T.serial(key_width):
                    Next[sequence, h, channel, k] = state[k]
        else:
            with T.Kernel(
                T.ceildiv(value_width, output_tile), value_heads, batch, threads=lanes * output_tile
            ) as (block, h, sequence):
                tid = T.get_thread_binding(0)
                lane = tid % lanes
                slot = tid // lanes
                channel = block * output_tile + slot
                state = T.alloc_local((T.ceildiv(key_width, lanes),), "float32")
                partial = T.alloc_local((1,), "float32")
                residual = T.alloc_local((1,), "float32")
                sums = T.alloc_shared((output_tile, lanes), "float32")
                for chunk in T.serial(T.ceildiv(key_width, lanes)):
                    k = chunk * lanes + lane
                    if channel < value_width and k < key_width:
                        state[chunk] = Previous[sequence, h, channel, k]
                for step_index in T.serial(steps):
                    row = sequence * steps + step_index
                    partial[0] = 0
                    for chunk in T.serial(T.ceildiv(key_width, lanes)):
                        k = chunk * lanes + lane
                        if channel < value_width and k < key_width:
                            state[chunk] *= Decay[row, h]
                            partial[0] += state[chunk] * K[row, key_head(h), k]
                    sums[slot, lane] = partial[0]
                    T.sync_threads()
                    for reduction in T.unroll(int(math.log2(lanes))):
                        if lane < (lanes >> (reduction + 1)):
                            sums[slot, lane] += sums[slot, lane + (lanes >> (reduction + 1))]
                        T.sync_threads()
                    residual[0] = 0
                    if channel < value_width:
                        residual[0] = (V[row, h, channel] - sums[slot, 0]) * Beta[row, h]
                    T.sync_threads()
                    partial[0] = 0
                    for chunk in T.serial(T.ceildiv(key_width, lanes)):
                        k = chunk * lanes + lane
                        if channel < value_width and k < key_width:
                            state[chunk] += residual[0] * K[row, key_head(h), k]
                            partial[0] += state[chunk] * Q[row, key_head(h), k]
                    sums[slot, lane] = partial[0]
                    T.sync_threads()
                    for reduction in T.unroll(int(math.log2(lanes))):
                        if lane < (lanes >> (reduction + 1)):
                            sums[slot, lane] += sums[slot, lane + (lanes >> (reduction + 1))]
                        T.sync_threads()
                    if lane == 0 and channel < value_width:
                        Output[row, h, channel] = sums[slot, 0]
                    T.sync_threads()
                for chunk in T.serial(T.ceildiv(key_width, lanes)):
                    k = chunk * lanes + lane
                    if channel < value_width and k < key_width:
                        Next[sequence, h, channel, k] = state[chunk]

    return main
