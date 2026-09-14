"""Sequence-specialized recurrent preparation schedules."""

from __future__ import annotations

import math
from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import BoundOperation, LoweringContext
from ..tensor.graph import Graph
from ..tensor.types import TensorSpec
from .matrix import (
    _packed_matrix,
    _packed_vector,
    _packed_vector_geometry,
    _packet_matrix_instruction,
    _packet_reduction_width,
)
from .normalization import _reduction_threads
from .packed import affine_shared_bytes, packet_format


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
    subgroup_width,
):
    heads = 2 * key_heads + value_heads
    with T.Kernel(heads, rows, threads=threads) as (head, row):
        lane = T.get_thread_binding(0)
        warp_sums = T.alloc_shared((T.ceildiv(threads, subgroup_width),), "float32")
        convolved = T.alloc_local((1,), "float32")
        square_sum = T.alloc_local((1,), "float32")
        sequence = T.alloc_local((1,), "int32")
        sequence[0] = 0
        for candidate in T.serial(batch):
            if offsets[candidate] <= row and row < offsets[candidate + 1]:
                sequence[0] = candidate
        step = row - offsets[sequence[0]]
        packed = head * width + lane
        count = offsets[sequence[0] + 1] - offsets[sequence[0]]
        convolved[0] = 0.0
        if lane < width:
            for time in T.serial(history):
                source_step = step - history + time
                convolved[0] += (
                    T.if_then_else(
                        source_step < 0,
                        previous[sequence[0], packed, source_step + history],
                        projected[offsets[sequence[0]] + source_step, packed],
                    )
                    * convolution[packed, time]
                )
            convolved[0] += projected[row, packed] * convolution[packed, history]
            convolved[0] *= T.sigmoid(convolved[0])
        if head < 2 * key_heads:
            square_sum[0] = T.if_then_else(
                lane < width,
                convolved[0] * convolved[0],
                0.0,
            )
            reduced = T.warp_reduce_sum(square_sum[0])
            if lane % subgroup_width == 0:
                warp_sums[lane // subgroup_width] = reduced
            T.sync_threads()
            square_sum[0] = 0.0
            for warp in T.unroll(T.ceildiv(threads, subgroup_width), explicit=True):
                square_sum[0] += warp_sums[warp]
            if lane < width:
                if head < key_heads:
                    query[row, head, lane] = T.cast(
                        convolved[0] * T.rsqrt(square_sum[0] + epsilon) * query_gain,
                        dtype,
                    )
                else:
                    key[row, head - key_heads, lane] = T.cast(
                        convolved[0] * T.rsqrt(square_sum[0] + epsilon), dtype
                    )
        else:
            value_head = head - 2 * key_heads
            if lane < width:
                value[row, value_head, lane] = T.cast(convolved[0], dtype)
            if lane == 0:
                beta[row, value_head] = T.cast(T.sigmoid(beta_input[row, value_head]), dtype)
                shifted = T.cast(alpha[row, value_head], "float32") + T.cast(
                    bias[value_head], "float32"
                )
                softplus = T.max(shifted, 0.0) + T.log(1 + T.exp(-T.abs(shifted)))
                decay[row, value_head] = T.exp(T.cast(rate[value_head], "float32") * softplus)
        if step == 0 and lane < width:
            for time in T.serial(history):
                source_step = count - history + time
                following[sequence[0], packed, time] = T.if_then_else(
                    source_step < 0,
                    previous[sequence[0], packed, source_step + history],
                    projected[offsets[sequence[0]] + source_step, packed],
                )


class _RecurrentPrepareEmitter:
    def __init__(self, specs: tuple[TensorSpec, ...], attrs, threads: int, subgroup_width: int):
        self.specs, self.attrs, self.threads = specs, attrs, threads
        self.subgroup_width = subgroup_width

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
            self.subgroup_width,
        )


class RecurrentPrepareRule:
    name = "channel-parallel-recurrent-prepare"

    def build(self, graph: Graph, root: int, context: LoweringContext):
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
        threads = _reduction_threads(width, context)
        if threads is None:
            return ()
        moved = sum(spec.storage_nbytes for spec in specs)
        return (
            BoundOperation(
                f"recurrent_prepare.channel-parallel@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                _RecurrentPrepareEmitter(
                    specs, node.attributes, threads, context.capabilities.subgroup_width
                ),
            ),
        )


@T.macro
def _register_delta_recurrence(
    query,
    key,
    value,
    decay,
    beta,
    previous,
    offsets,
    output,
    next_state,
    batch,
    key_heads,
    value_heads,
    key_width,
    value_width,
    mapping,
    lanes,
    output_tile,
    dtype,
):
    """Keep each state row in registers for the complete sequence span."""
    with T.Kernel(
        T.ceildiv(value_width, output_tile),
        value_heads,
        batch,
        threads=lanes * output_tile,
    ) as (block, head, sequence):
        thread = T.get_thread_binding()
        lane = thread % lanes
        slot = thread // lanes
        channel = block * output_tile + slot
        state = T.alloc_local((T.ceildiv(key_width, lanes),), "float32")
        key_values = T.alloc_local((T.ceildiv(key_width, lanes),), "float32")
        query_values = T.alloc_local((T.ceildiv(key_width, lanes),), "float32")
        partial = T.alloc_local((1,), "float32")
        residual = T.alloc_local((1,), "float32")
        key_head = T.if_then_else(
            mapping == "tiled", head % key_heads, head // (value_heads // key_heads)
        )
        for chunk in T.unroll(T.ceildiv(key_width, lanes), explicit=True):
            reduction = (
                lane * T.ceildiv(key_width, lanes) + chunk
                if dtype == "bfloat16"
                else chunk * lanes + lane
            )
            state[chunk] = 0.0
            if channel < value_width and reduction < key_width:
                state[chunk] = previous[sequence, head, channel, reduction]
        # An absolute bounded row domain lets lowering prove each input access
        # is valid while retaining runtime sequence lengths and packed offsets.
        for row in T.serial(
            T.max(0, offsets[sequence]), T.min(query.shape[0], offsets[sequence + 1])
        ):
            partial[0] = 0.0
            for chunk in T.unroll(T.ceildiv(key_width, lanes), explicit=True):
                reduction = (
                    lane * T.ceildiv(key_width, lanes) + chunk
                    if dtype == "bfloat16"
                    else chunk * lanes + lane
                )
                key_values[chunk] = 0.0
                query_values[chunk] = 0.0
                if channel < value_width and reduction < key_width:
                    key_values[chunk] = T.cast(key[row, key_head, reduction], "float32")
                    query_values[chunk] = T.cast(query[row, key_head, reduction], "float32")
                    state[chunk] *= decay[row, head]
                    partial[0] += state[chunk] * key_values[chunk]
            remembered = T.warp_reduce_sum(partial[0])
            residual[0] = T.if_then_else(
                channel < value_width,
                (T.cast(value[row, head, channel], "float32") - remembered)
                * T.cast(beta[row, head], "float32"),
                0.0,
            )
            partial[0] = 0.0
            for chunk in T.unroll(T.ceildiv(key_width, lanes), explicit=True):
                reduction = (
                    lane * T.ceildiv(key_width, lanes) + chunk
                    if dtype == "bfloat16"
                    else chunk * lanes + lane
                )
                if channel < value_width and reduction < key_width:
                    state[chunk] += residual[0] * key_values[chunk]
                    partial[0] += state[chunk] * query_values[chunk]
            answer = T.warp_reduce_sum(partial[0])
            if lane == 0 and channel < value_width:
                output[row, head, channel] = T.cast(answer, dtype)
        for chunk in T.unroll(T.ceildiv(key_width, lanes), explicit=True):
            reduction = (
                lane * T.ceildiv(key_width, lanes) + chunk
                if dtype == "bfloat16"
                else chunk * lanes + lane
            )
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
        _, key_heads, _ = cast(tuple[int, int, int], self.specs[0].shape)
        _register_delta_recurrence(
            *operands[:7],
            operands[7],
            operands[8],
            batch,
            key_heads,
            value_heads,
            key_width,
            value_width,
            self.mapping,
            self.lanes,
            self.output_tile,
            self.specs[7].dtype.value,
        )


class GatedDeltaRule:
    name = "register-gated-delta"

    def build(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if (
            node.operation != "gated_delta_recurrence"
            or "shared" not in context.capabilities.memory_scopes
        ):
            return ()
        specs = tuple(graph.values[value].spec for value in (*node.inputs, *node.outputs))
        if any(not spec.static for spec in specs):
            return ()
        lanes = context.capabilities.subgroup_width
        output_tile = min(4, context.capabilities.threads_per_group // lanes)
        if lanes < 2 or lanes & (lanes - 1) or output_tile < 1:
            return ()
        return (
            BoundOperation(
                f"gated_delta.register-state@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                _GatedDeltaEmitter(specs, node.attributes["mapping"], lanes, output_tile),
            ),
        )


@T.macro
def _recurrent_norm_gate(
    mixed,
    norm,
    gate,
    activation,
    rows,
    heads,
    width,
    epsilon,
    dtype,
    threads,
    subgroup_width,
):
    with T.Kernel(heads, rows, threads=threads) as (head, row):
        lane = T.get_thread_binding()
        warp_sums = T.alloc_shared((T.ceildiv(threads, subgroup_width),), "float32")
        value = T.alloc_local((1,), "float32")
        value[0] = T.if_then_else(lane < width, T.cast(mixed[row, head, lane], "float32"), 0.0)
        reduced = T.warp_reduce_sum(value[0] * value[0])
        if lane % subgroup_width == 0:
            warp_sums[lane // subgroup_width] = reduced
        T.sync_threads()
        square_sum = T.alloc_local((1,), "float32")
        square_sum[0] = 0.0
        for warp in T.unroll(T.ceildiv(threads, subgroup_width), explicit=True):
            square_sum[0] += warp_sums[warp]
        if lane < width:
            channel = head * width + lane
            normalized = T.cast(
                value[0] * T.rsqrt(square_sum[0] / width + epsilon) * T.cast(norm[lane], "float32"),
                dtype,
            )
            gate_value = T.cast(gate[row, channel], "float32")
            activated_gate = T.cast(gate_value * T.sigmoid(gate_value), dtype)
            activation[row, channel] = T.cast(
                T.cast(normalized, "float32") * T.cast(activated_gate, "float32"), dtype
            )


def _recurrent_output_region(graph: Graph, root: int):
    if root + 4 >= len(graph.nodes):
        return None
    norm, reshape, silu, multiply, linear = graph.nodes[root : root + 5]
    if (
        norm.operation != "rms_norm"
        or len(norm.inputs) != 2
        or graph.values[norm.outputs[0]].spec.dtype != graph.values[norm.inputs[0]].spec.dtype
        or reshape.operation != "reshape"
        or reshape.inputs != norm.outputs
        or silu.operation != "silu"
        or multiply.operation != "multiply"
        or reshape.outputs[0] not in multiply.inputs
        or silu.outputs[0] not in multiply.inputs
        or linear.operation != "linear"
        or linear.inputs[0] != multiply.outputs[0]
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
    def __init__(self, specs, epsilon, mode, tile, vector, norm_threads, subgroup_width):
        self.specs, self.epsilon, self.mode = specs, epsilon, mode
        self.tile, self.vector = tile, vector
        self.norm_threads, self.subgroup_width = norm_threads, subgroup_width

    def __call__(self, operands: tuple[Any, ...]) -> None:
        mixed, norm, gate, weight, output, activation = operands
        rows, heads, width = cast(tuple[int, int, int], self.specs[0].shape)
        channels = heads * width
        _recurrent_norm_gate(
            mixed,
            norm,
            gate,
            activation,
            rows,
            heads,
            width,
            self.epsilon,
            self.specs[0].dtype.value,
            self.norm_threads,
            self.subgroup_width,
        )
        outputs = cast(int, self.specs[3].shape[0])
        if self.mode == "decode":
            threads, outputs_per_subgroup = self.vector
            _packed_vector(
                activation,
                weight,
                activation,
                output,
                self.specs[3],
                rows,
                outputs,
                channels,
                self.specs[4].dtype.value,
                False,
                threads,
                outputs_per_subgroup,
            )
        else:
            threads, bm, bn, bk, instruction = self.tile
            _packed_matrix(
                activation,
                weight,
                activation,
                output,
                self.specs[3],
                rows,
                outputs,
                channels,
                instruction,
                self.specs[4].dtype.value,
                threads,
                bm,
                bn,
                bk,
                False,
            )


class RecurrentOutputRule:
    name = "recurrent-output"

    def build(self, graph: Graph, root: int, context: LoweringContext):
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
        norm_threads = _reduction_threads(width, context)
        if norm_threads is None:
            return ()
        tile = None
        vector = None
        if context.mode == "prefill":
            instruction = _packet_matrix_instruction(context, specs[0].dtype)
            if instruction is None:
                return ()
            if rows >= 256 and min(cast(int, specs[3].shape[0]), channels) >= 512:
                bm, bn, bk = 32, 64, 32
            else:
                bm, bn, bk = (
                    instruction.m * 4,
                    instruction.n * 4,
                    instruction.k * 2,
                )
            bk = _packet_reduction_width(specs[3])
            if bk % instruction.k:
                return ()
            threads = min(
                context.capabilities.threads_per_group,
                context.capabilities.subgroup_width * 4,
                bm // instruction.m * context.capabilities.subgroup_width,
            )
            if affine_shared_bytes(bm, bn, bk, specs[0].dtype, specs[3]) > context.capabilities.shared_memory_bytes:
                return ()
            tile = (threads, bm, bn, bk, instruction)
        else:
            vector = _packed_vector_geometry(specs[3], context)
            if rows > 8 or vector is None:
                return ()
        activation = TensorSpec((rows, channels), specs[0].dtype)
        moved = sum(spec.storage_nbytes for spec in specs)
        return (
            BoundOperation(
                f"recurrent.output-{context.mode}@{root}:{max(nodes)}",
                nodes,
                inputs,
                outputs,
                _RecurrentOutputEmitter(
                    specs,
                    epsilon,
                    context.mode,
                    tile,
                    vector,
                    norm_threads,
                    context.capabilities.subgroup_width,
                ),
                workspace=(activation,),
                kernel_count=2,
            ),
        )


__all__ = ["GatedDeltaRule", "RecurrentOutputRule", "RecurrentPrepareRule"]
