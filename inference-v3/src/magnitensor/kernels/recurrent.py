"""Sequence-specialized recurrent preparation schedules."""

from __future__ import annotations

import math
from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..tensor.graph import Graph
from ..tensor.types import TensorSpec
from .matrix import _matrix_instruction, _packed_matrix, _packed_vector
from .packed import packet_format


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


@T.macro
def _register_delta_recurrence(
    query, key, value, decay, beta, previous, offsets, output, next_state,
    batch, rows, key_heads, value_heads, key_width, value_width, mapping,
    lanes, output_tile, dtype,
):
    """Keep each state row in registers for the complete sequence span."""
    with T.Kernel(
        T.ceildiv(value_width, output_tile), value_heads, batch,
        threads=lanes * output_tile,
    ) as (block, head, sequence):
        thread = T.get_thread_binding()
        lane = thread % lanes
        slot = thread // lanes
        channel = block * output_tile + slot
        state = T.alloc_local((T.ceildiv(key_width, lanes),), "float32")
        sums = T.alloc_shared((output_tile, lanes), "float32")
        key_head = T.if_then_else(
            mapping == "tiled", head % key_heads, head // (value_heads // key_heads)
        )
        for chunk in T.serial(T.ceildiv(key_width, lanes)):
            reduction = chunk * lanes + lane
            if channel < value_width and reduction < key_width:
                state[chunk] = previous[sequence, head, channel, reduction]
        count = offsets[sequence + 1] - offsets[sequence]
        for step in T.serial(rows):
            if step < count:
                row = offsets[sequence] + step
                dot = T.alloc_local((1,), "float32")
                dot[0] = 0.0
                for chunk in T.serial(T.ceildiv(key_width, lanes)):
                    reduction = chunk * lanes + lane
                    if channel < value_width and reduction < key_width:
                        state[chunk] *= decay[row, head]
                        dot[0] += state[chunk] * key[row, key_head, reduction]
                sums[slot, lane] = dot[0]
                T.sync_threads()
                for reduction_step in T.unroll(int(math.log2(lanes))):
                    distance = lanes >> (reduction_step + 1)
                    if lane < distance:
                        sums[slot, lane] += sums[slot, lane + distance]
                    T.sync_threads()
                residual = T.alloc_local((1,), "float32")
                residual[0] = T.if_then_else(
                    channel < value_width,
                    (T.cast(value[row, head, channel], "float32") - sums[slot, 0])
                    * T.cast(beta[row, head], "float32"),
                    0.0,
                )
                dot[0] = 0.0
                for chunk in T.serial(T.ceildiv(key_width, lanes)):
                    reduction = chunk * lanes + lane
                    if channel < value_width and reduction < key_width:
                        state[chunk] += residual[0] * key[row, key_head, reduction]
                        dot[0] += state[chunk] * query[row, key_head, reduction]
                sums[slot, lane] = dot[0]
                T.sync_threads()
                for reduction_step in T.unroll(int(math.log2(lanes))):
                    distance = lanes >> (reduction_step + 1)
                    if lane < distance:
                        sums[slot, lane] += sums[slot, lane + distance]
                    T.sync_threads()
                if lane == 0 and channel < value_width:
                    output[row, head, channel] = T.cast(sums[slot, 0], dtype)
                T.sync_threads()
        for chunk in T.serial(T.ceildiv(key_width, lanes)):
            reduction = chunk * lanes + lane
            if channel < value_width and reduction < key_width:
                next_state[sequence, head, channel, reduction] = state[chunk]


class _GatedDeltaEmitter:
    def __init__(self, specs, mapping, lanes, output_tile):
        self.specs, self.mapping = specs, mapping
        self.lanes, self.output_tile = lanes, output_tile

    def __call__(self, operands: tuple[Any, ...]) -> None:
        batch, value_heads, value_width, key_width = cast(
            tuple[int, int, int, int], self.specs[5].shape
        )
        rows, key_heads, _ = cast(tuple[int, int, int], self.specs[0].shape)
        _register_delta_recurrence(
            *operands[:7], operands[7], operands[8],
            batch, rows, key_heads, value_heads, key_width, value_width,
            self.mapping, self.lanes, self.output_tile, self.specs[7].dtype.value,
        )


class GatedDeltaRule:
    name = "register-gated-delta"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation != "gated_delta_recurrence" or "shared" not in context.capabilities.memory_scopes:
            return ()
        specs = tuple(graph.values[value].spec for value in (*node.inputs, *node.outputs))
        if any(not spec.static for spec in specs):
            return ()
        lanes = min(32, context.capabilities.subgroup_width)
        output_tile = min(4, context.capabilities.threads_per_group // lanes)
        if lanes < 2 or lanes & (lanes - 1) or output_tile < 1:
            return ()
        moved = sum(spec.storage_nbytes for spec in specs)
        return (Candidate(
            f"gated_delta.register-state@{root}", frozenset({root}), node.inputs, node.outputs,
            _GatedDeltaEmitter(specs, node.attributes["mapping"], lanes, output_tile),
            5e-7 + moved / 200e9, aliases=((node.outputs[1], node.inputs[5]),),
            priority=60,
        ),)


@T.macro
def _recurrent_norm_gate(mixed, norm, gate, activation, rows, heads, width, epsilon, dtype, threads):
    with T.Kernel(heads, rows, threads=threads) as (head, row):
        lane = T.get_thread_binding()
        squares = T.alloc_shared((threads,), "float32")
        value = T.alloc_local((1,), "float32")
        value[0] = T.if_then_else(
            lane < width, T.cast(mixed[row, head, lane], "float32"), 0.0
        )
        squares[lane] = value[0] * value[0]
        T.sync_threads()
        for step in T.unroll(int(math.log2(threads))):
            distance = threads >> (step + 1)
            if lane < distance:
                squares[lane] += squares[lane + distance]
            T.sync_threads()
        if lane < width:
            channel = head * width + lane
            normalized = value[0] * T.rsqrt(squares[0] / width + epsilon) * T.cast(
                norm[lane], "float32"
            )
            gate_value = T.cast(gate[row, channel], "float32")
            activation[row, channel] = T.cast(
                normalized * gate_value * T.sigmoid(gate_value), dtype
            )


def _recurrent_output_region(graph: Graph, root: int):
    if root + 4 >= len(graph.nodes):
        return None
    norm, reshape, silu, multiply, linear = graph.nodes[root : root + 5]
    if (
        norm.operation != "rms_norm" or len(norm.inputs) != 2
        or reshape.operation != "reshape" or reshape.inputs != norm.outputs
        or silu.operation != "silu"
        or multiply.operation != "multiply" or reshape.outputs[0] not in multiply.inputs
        or silu.outputs[0] not in multiply.inputs
        or linear.operation != "linear" or linear.inputs[0] != multiply.outputs[0]
    ):
        return None
    gate = silu.inputs[0]
    inputs = (norm.inputs[0], norm.inputs[1], gate, linear.inputs[1])
    outputs = linear.outputs
    specs = tuple(graph.values[value].spec for value in (*inputs, *outputs))
    if any(not spec.static for spec in specs) or packet_format(specs[3]) is None:
        return None
    return frozenset(range(root, root + 5)), inputs, outputs, specs, norm.attributes["epsilon"]


class _RecurrentOutputEmitter:
    def __init__(self, specs, epsilon, mode, tile):
        self.specs, self.epsilon, self.mode, self.tile = specs, epsilon, mode, tile

    def __call__(self, operands: tuple[Any, ...]) -> None:
        mixed, norm, gate, weight, output, activation = operands
        rows, heads, width = cast(tuple[int, int, int], self.specs[0].shape)
        channels = heads * width
        threads = 1 << (width - 1).bit_length()
        _recurrent_norm_gate(
            mixed, norm, gate, activation, rows, heads, width, self.epsilon,
            self.specs[0].dtype.value, threads,
        )
        outputs = cast(int, self.specs[3].shape[0])
        if self.mode == "decode":
            _packed_vector(
                activation, weight, activation, output, self.specs[3], rows, outputs,
                channels, self.specs[4].dtype.value, False,
            )
        else:
            threads, bm, bn, bk = self.tile
            _packed_matrix(
                activation, weight, activation, output, self.specs[3], rows, outputs,
                channels, self.specs[0].dtype.value, self.specs[4].dtype.value,
                threads, bm, bn, bk, False,
            )


class RecurrentOutputRule:
    name = "recurrent-output"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        region = _recurrent_output_region(graph, root)
        if region is None:
            return ()
        nodes, inputs, outputs, specs, epsilon = region
        rows, heads, width = cast(tuple[int, int, int], specs[0].shape)
        channels = heads * width
        packet = packet_format(specs[3])
        assert packet is not None
        if channels % packet.tile:
            return ()
        tile = None
        if context.mode == "prefill":
            instruction = _matrix_instruction(context, specs[0].dtype)
            if instruction is None:
                return ()
            bm, bn, bk = instruction.m * 4, instruction.n * 4, instruction.k * 2
            bk = ((bk + packet.packet - 1) // packet.packet) * packet.packet
            threads = min(context.capabilities.threads_per_group, context.capabilities.subgroup_width * 4)
            if (bm * bk + bn * bk) * specs[0].dtype.itemsize > context.capabilities.shared_memory_bytes:
                return ()
            tile = (threads, bm, bn, bk)
        elif rows > 8 or context.capabilities.subgroup_width != 32:
            return ()
        activation = TensorSpec((rows, channels), specs[0].dtype)
        moved = sum(spec.storage_nbytes for spec in specs)
        return (Candidate(
            f"recurrent.output-{context.mode}@{root}:{max(nodes)}", nodes, inputs, outputs,
            _RecurrentOutputEmitter(specs, epsilon, context.mode, tile),
            4e-7 + moved / 4e12, workspace=(activation,), kernel_count=2, priority=90,
        ),)


__all__ = ["GatedDeltaRule", "RecurrentOutputRule", "RecurrentPrepareRule"]
