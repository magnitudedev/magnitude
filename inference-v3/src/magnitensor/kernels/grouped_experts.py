"""Grouped expert prefill pipeline selected from explicit target capabilities."""

from __future__ import annotations

from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..representations import Dense
from ..tensor.graph import Graph
from ..tensor.types import DType, TensorSpec
from .quantization import represented_load


def _aligned_capacity(rows: int, selected: int, experts: int, tile: int) -> int:
    routes = rows * selected
    return ((routes + experts * (tile - 1) + tile - 1) // tile) * tile


@T.macro
def _group_count(
    routes, order, inverse, block_experts, counts, rows, selected, experts, capacity, tile
):
    count = rows * selected
    blocks = capacity // tile
    with T.Kernel(1, threads=256):
        lane = T.get_thread_binding(0)
        if lane < experts:
            counts[lane] = 0
        for chunk in T.serial(T.ceildiv(capacity, 256)):
            position = chunk * 256 + lane
            if position < capacity:
                order[position] = -1
        for chunk in T.serial(T.ceildiv(count, 256)):
            position = chunk * 256 + lane
            if position < count:
                inverse[position] = -1
        for chunk in T.serial(T.ceildiv(blocks, 256)):
            block = chunk * 256 + lane
            if block < blocks:
                block_experts[block] = -1
        T.sync_threads()
        for chunk in T.serial(T.ceildiv(count, 256)):
            route = chunk * 256 + lane
            if route < count:
                expert = routes[route // selected, route % selected]
                if 0 <= expert and expert < experts:
                    T.atomic_add(counts[expert], 1)


@T.macro
def _group_prefix(counts, cursors, block_experts, experts, tile):
    with T.Kernel(1, threads=1):
        cursor = T.alloc_local((1,), "int32")
        cursor[0] = 0
        for expert in T.serial(experts):
            cursors[expert] = cursor[0]
            for block in T.serial(T.ceildiv(counts[expert], tile)):
                block_experts[cursor[0] // tile + block] = expert
            cursor[0] += T.ceildiv(counts[expert], tile) * tile


@T.macro
def _group_scatter(routes, order, inverse, cursors, rows, selected, experts):
    count = rows * selected
    with T.Kernel(1, threads=256):
        lane = T.get_thread_binding(0)
        for chunk in T.serial(T.ceildiv(count, 256)):
            route = chunk * 256 + lane
            if route < count:
                expert = routes[route // selected, route % selected]
                if 0 <= expert and expert < experts:
                    position = T.atomic_add(cursors[expert], 1, return_prev=True)
                    order[position] = route
                    inverse[route] = position


def _weight(storage, spec: TensorSpec, expert, output, reduction):
    experts, rows, columns = cast(tuple[int, int, int], spec.shape)
    del experts
    if spec.representation is None or isinstance(spec.representation, Dense):
        return T.cast(storage[expert, output, reduction], spec.dtype.value)
    index = (expert * rows + output) * columns + reduction
    return T.cast(represented_load(storage, spec, index), spec.dtype.value)


@T.macro
def _grouped_projection(
    source,
    order,
    block_experts,
    weight,
    output,
    weight_spec,
    blocks,
    source_width,
    output_width,
    selected,
    source_grouped,
    bm,
    bn,
    bk,
    threads,
):
    with T.Kernel(T.ceildiv(output_width, bn), blocks, threads=threads) as (bx, by):
        x = T.alloc_shared((bm, bk), weight_spec.dtype.value)
        w = T.alloc_shared((bn, bk), weight_spec.dtype.value)
        accum = T.alloc_fragment((bm, bn), "float32")
        T.clear(accum)
        expert = block_experts[by]
        for reduction_block in T.serial(T.ceildiv(source_width, bk)):
            for i, k in T.Parallel(bm, bk):
                route = order[by * bm + i]
                reduction = reduction_block * bk + k
                x[i, k] = T.if_then_else(
                    route >= 0 and reduction < source_width,
                    source[
                        T.if_then_else(source_grouped, by * bm + i, route // selected),
                        reduction,
                    ],
                    0,
                )
            for j, k in T.Parallel(bn, bk):
                channel = bx * bn + j
                reduction = reduction_block * bk + k
                if expert >= 0 and channel < output_width and reduction < source_width:
                    w[j, k] = _weight(weight, weight_spec, expert, channel, reduction)
                else:
                    w[j, k] = 0
            T.sync_threads()
            T.gemm(x, w, accum, transpose_B=True)
            T.sync_threads()
        for i, j in T.Parallel(bm, bn):
            route = order[by * bm + i]
            channel = bx * bn + j
            if route >= 0 and channel < output_width:
                output[by * bm + i, channel] = T.cast(
                    accum[i, j], weight_spec.dtype.value
                )


@T.macro
def _activate(gate, up, output, capacity, width, dtype, threads):
    with T.Kernel(T.ceildiv(capacity * width, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < capacity * width:
                row, column = flat // width, flat % width
                value = T.cast(gate[row, column], "float32")
                output[row, column] = T.cast(
                    value * T.sigmoid(value) * T.cast(up[row, column], "float32"),
                    dtype,
                )


@T.macro
def _unpermute(projected, inverse, scores, output, rows, selected, width, dtype, threads):
    with T.Kernel(T.ceildiv(rows * width, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < rows * width:
                row, column = flat // width, flat % width
                total = T.alloc_local((1,), "float32")
                total[0] = 0.0
                for rank in T.serial(selected):
                    source = inverse[row * selected + rank]
                    if source >= 0:
                        total[0] += T.cast(projected[source, column], "float32") * T.cast(
                            scores[row, rank], "float32"
                        )
                output[row, column] = T.cast(total[0], dtype)


class _GroupedExpertsEmitter:
    def __init__(self, specs, capacity, blocks, tile):
        self.specs = specs
        self.capacity, self.blocks, self.tile = capacity, blocks, tile

    def __call__(self, operands: tuple[Any, ...]) -> None:
        hidden, routes, scores, gate, up, down, output = operands[:7]
        (
            order,
            inverse,
            block_experts,
            counts,
            cursors,
            gate_result,
            up_result,
            activation,
            projected,
        ) = operands[7:]
        rows, width = cast(tuple[int, int], self.specs[0].shape)
        selected = cast(int, self.specs[1].shape[1])
        experts, intermediate, _ = cast(tuple[int, int, int], self.specs[3].shape)
        instruction = self.tile
        bm, bn, bk, threads = instruction
        _group_count(
            routes,
            order,
            inverse,
            block_experts,
            counts,
            rows,
            selected,
            experts,
            self.capacity,
            bm,
        )
        _group_prefix(counts, cursors, block_experts, experts, bm)
        _group_scatter(routes, order, inverse, cursors, rows, selected, experts)
        _grouped_projection(
            hidden,
            order,
            block_experts,
            gate,
            gate_result,
            self.specs[3],
            self.blocks,
            width,
            intermediate,
            selected,
            False,
            bm,
            bn,
            bk,
            threads,
        )
        _grouped_projection(
            hidden,
            order,
            block_experts,
            up,
            up_result,
            self.specs[4],
            self.blocks,
            width,
            intermediate,
            selected,
            False,
            bm,
            bn,
            bk,
            threads,
        )
        _activate(
            gate_result,
            up_result,
            activation,
            self.capacity,
            intermediate,
            self.specs[0].dtype.value,
            threads,
        )
        _grouped_projection(
            activation,
            order,
            block_experts,
            down,
            projected,
            self.specs[5],
            self.blocks,
            intermediate,
            width,
            selected,
            True,
            bm,
            bn,
            bk,
            threads,
        )
        _unpermute(
            projected,
            inverse,
            scores,
            output,
            rows,
            selected,
            width,
            self.specs[6].dtype.value,
            threads,
        )


class GroupedExpertsRule:
    name = "grouped-prefill-experts"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if (
            node.operation != "routed_experts"
            or node.attributes["activation"] != "silu"
            or context.mode == "decode"
            or DType.I32 not in context.capabilities.atomics
            or "shared" not in context.capabilities.memory_scopes
        ):
            return ()
        specs = tuple(graph.values[value].spec for value in (*node.inputs, *node.outputs))
        if any(not spec.static for spec in specs):
            return ()
        hidden, routes, scores, gate, up, down, output = specs
        rows, width = cast(tuple[int, int], hidden.shape)
        selected = cast(int, routes.shape[1])
        experts, intermediate, input_width = cast(tuple[int, int, int], gate.shape)
        instruction = next(
            (
                value
                for value in context.capabilities.matrix_instructions
                if value.input_dtype == hidden.dtype
            ),
            None,
        )
        if (
            instruction is None
            or rows * selected < experts
            or input_width != width
            or up.shape != gate.shape
            or down.shape != (experts, width, intermediate)
            or scores.shape != routes.shape
            or output.shape != hidden.shape
            or experts > 256
        ):
            return ()
        bm = instruction.m * 2
        bn = instruction.n * 2
        bk = instruction.k
        threads = min(
            context.capabilities.threads_per_group,
            context.capabilities.subgroup_width * 4,
        )
        shared = (bm * bk + bn * bk) * hidden.dtype.itemsize
        if shared > context.capabilities.shared_memory_bytes:
            return ()
        capacity = _aligned_capacity(rows, selected, experts, bm)
        blocks = capacity // bm
        workspace = (
            TensorSpec((capacity,), DType.I32),
            TensorSpec((rows * selected,), DType.I32),
            TensorSpec((blocks,), DType.I32),
            TensorSpec((experts,), DType.I32),
            TensorSpec((experts,), DType.I32),
            TensorSpec((capacity, intermediate), hidden.dtype),
            TensorSpec((capacity, intermediate), hidden.dtype),
            TensorSpec((capacity, intermediate), hidden.dtype),
            TensorSpec((capacity, width), hidden.dtype),
        )
        if sum(value.storage_nbytes for value in workspace) > context.workspace_limit:
            return ()
        operations = 6 * rows * selected * width * intermediate
        return (
            Candidate(
                f"routed_experts.grouped@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                _GroupedExpertsEmitter(
                    specs, capacity, blocks, (bm, bn, bk, threads)
                ),
                1e-7 + operations / 5e12,
                workspace=workspace,
                kernel_count=8,
                priority=40,
            ),
        )
