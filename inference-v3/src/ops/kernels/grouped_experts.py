"""Grouped expert prefill pipeline selected from explicit target capabilities."""

from __future__ import annotations

from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import BoundOperation, LoweringContext
from ..tensor.graph import Graph
from ..tensor.types import DType, TensorSpec
from .experts import _routed_shared_region
from .matrix import _packet_matrix_instruction, _packet_reduction_width
from .packed import (affine_gemm, affine_has_bias, affine_shared_bytes, affine_storage,
                     load_matrix_tile, packet_format, prepare_decoded_packets, publish_decoded_packets)
from .publication import publish, residual_epilogue


def _aligned_capacity(rows: int, selected: int, experts: int, tile: int) -> int:
    routes = rows * selected
    return ((routes + experts * (tile - 1) + tile - 1) // tile) * tile


def _continuous_contraction(context, dtype, instruction):
    return (dtype == DType.BF16 and instruction.input_dtype == DType.BF16
            and (instruction.m, instruction.n, instruction.k) == (8, 8, 8)
            and "gemm.shared_instruction_tiles" in context.capabilities.features)


@T.macro
def _group_routes(
    routes,
    order,
    inverse,
    block_metadata,
    block_count,
    rows,
    selected,
    experts,
    capacity,
    tile,
):
    """Build the complete expert permutation in one workgroup launch."""
    count = rows * selected
    blocks = capacity // tile
    with T.Kernel(1, threads=256):
        lane = T.get_thread_binding(0)
        counts = T.alloc_shared((experts,), "int32")
        cursors = T.alloc_shared((experts,), "int32")
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
        for chunk in T.serial(T.ceildiv(blocks * 2, 256)):
            index = chunk * 256 + lane
            if index < blocks * 2:
                block_metadata[index // 2, index % 2] = -1
        T.sync_threads()
        for chunk in T.serial(T.ceildiv(count, 256)):
            route = chunk * 256 + lane
            if route < count:
                expert = routes[route // selected, route % selected]
                if 0 <= expert and expert < experts:
                    T.atomic_add(counts[expert], 1)
        T.sync_threads()
        if lane == 0:
            cursor = T.alloc_local((1,), "int32")
            cursor[0] = 0
            for expert in T.serial(experts):
                cursors[expert] = cursor[0]
                for block in T.serial(T.ceildiv(counts[expert], tile)):
                    block_index = cursor[0] // tile + block
                    remainder = counts[expert] - block * tile
                    block_metadata[block_index, 0] = expert
                    block_metadata[block_index, 1] = T.min(
                        tile,
                        remainder,
                    )
                cursor[0] += T.ceildiv(counts[expert], tile) * tile
            block_count[0] = cursor[0] // tile
        T.sync_threads()
        for chunk in T.serial(T.ceildiv(count, 256)):
            route = chunk * 256 + lane
            if route < count:
                expert = routes[route // selected, route % selected]
                if 0 <= expert and expert < experts:
                    position = T.atomic_add(cursors[expert], 1, return_prev=True)
                    order[position] = route
                    inverse[route] = position


@T.macro
def _prepare_affine_rows(source, order, gathered, statistics, rows, capacity, width,
                         selected, bm, bk, threads, routed=True):
    """Resolve irregular rows and affine sums once, before output-channel tiling."""
    with T.Kernel(T.ceildiv(width, bk), T.ceildiv(capacity, bm), threads=threads) as (bx, by):
        values = T.alloc_fragment((bm, bk), "float32")
        totals = T.alloc_fragment((bm,), "float32") if statistics is not None else None
        for i, k in T.Parallel(bm, bk):
            row, column = by * bm + i, bx * bk + k
            route = order[row] if routed else row
            source_row = route // selected if routed else row
            values[i, k] = T.if_then_else(row < capacity and route >= 0 and source_row < rows and column < width,
                                         T.cast(source[source_row, column], "float32"), 0)
            if routed and row < capacity and column < width:
                gathered[row, column] = T.cast(values[i, k], source.dtype)
        if statistics is not None:
            T.reduce_sum(values, totals, dim=1)
            for i in T.Parallel(bm):
                if by * bm + i < capacity:
                    statistics[by * bm + i, bx] = totals[i]


@T.macro
def _expert_gated_tile(
    source,
    order,
    gate,
    up,
    output,
    gate_spec,
    up_spec,
    expert,
    valid_m,
    block,
    output_block,
    source_width,
    output_width,
    selected,
    routed,
    source_grouped,
    bm,
    bn,
    bk,
    threads,
    storage,
    statistics=None,
):
    """A complete tile contraction with the route kind fixed before reduction."""
    dtype = source.dtype
    contraction, gate_activation = storage
    x, paired_tile, coefficients, paired_accum, partial, sum_values, sums = contraction
    input_tile = T.alloc_fragment((bm, bk), dtype) if coefficients is None else x
    owners = threads // 2
    packet_width = min(16, max(8, bn * bk // owners))
    packets = T.alloc_local((T.ceildiv(bn * bk, owners * packet_width), packet_width), "float32") if coefficients is None else None
    T.clear(paired_accum)
    for reduction_block in T.serial(T.ceildiv(source_width, bk)):
        for i, k in T.Parallel(bm, bk):
            reduction = reduction_block * bk + k
            if routed and not source_grouped:
                route = order[block * bm + i]
                source_row = route // selected
                input_tile[i, k] = T.if_then_else(
                    i < valid_m and route >= 0 and reduction < source_width,
                    source[source_row, reduction],
                    0,
                )
            else:
                input_tile[i, k] = T.if_then_else(
                    i < valid_m and reduction < source_width,
                    source[block * bm + i, reduction],
                    0,
                )
        if coefficients is None:
            prepare_decoded_packets(packets, gate, gate_spec,
                                    expert * output_width + output_block * bn, reduction_block * bk,
                                    (expert + 1) * output_width, source_width, bn, bk, owners)
            prepare_decoded_packets(packets, up, up_spec,
                                    expert * output_width + output_block * bn, reduction_block * bk,
                                    (expert + 1) * output_width, source_width, bn, bk, owners, owners)
            T.sync_threads()
            T.copy(input_tile, x)
            publish_decoded_packets(packets, paired_tile, bn, bk, owners, 0, 2, 0)
            publish_decoded_packets(packets, paired_tile, bn, bk, owners, owners, 2, 1)
        else:
            load_matrix_tile(paired_tile, coefficients, gate, gate_spec,
                             expert * output_width + output_block * bn, reduction_block * bk,
                             (expert + 1) * output_width, source_width, bn, bk, threads, 2, 0)
            load_matrix_tile(paired_tile, coefficients, up, up_spec,
                             expert * output_width + output_block * bn, reduction_block * bk,
                             (expert + 1) * output_width, source_width, bn, bk, threads, 2, 1)
        if statistics is not None:
            for i in T.Parallel(bm):
                sums[i] = statistics[block * bm + i, reduction_block]
        affine_gemm(contraction, bm, 2 * bn, bk, valid_m, affine_has_bias(gate_spec, up_spec),
                    sums_ready=statistics is not None)
    for i, j in T.Parallel(bm, bn):
        gate_value = T.cast(T.cast(paired_accum[i, 2 * j], dtype), "float32")
        if routed:
            gate_activation[i, j] = gate_value * T.sigmoid(gate_value)
        else:
            # The shared branch is an explicit linear -> SiLU -> multiply
            # graph, unlike the internal arithmetic of routed_experts.
            gate_activation[i, j] = T.cast(gate_value * T.sigmoid(gate_value), dtype)
    for i, j in T.Parallel(bm, bn):
        channel = output_block * bn + j
        if i < valid_m and channel < output_width:
            up_value = T.cast(T.cast(paired_accum[i, 2 * j + 1], dtype), "float32")
            output[block * bm + i, channel] = T.cast(
                gate_activation[i, j] * up_value,
                dtype,
            )


@T.macro
def _expert_down_tile(
    source,
    order,
    weight,
    output,
    weight_spec,
    expert,
    valid_m,
    block,
    output_block,
    source_width,
    output_width,
    selected,
    routed,
    source_grouped,
    bm,
    bn,
    bk,
    threads,
    storage,
):
    x, w, coefficients, accum, partial, sum_values, sums = storage
    input_tile = T.alloc_fragment((bm, bk), source.dtype) if coefficients is None else x
    packet_width = min(16, max(8, bn * bk // threads))
    packets = T.alloc_local((T.ceildiv(bn * bk, threads * packet_width), packet_width), "float32") if coefficients is None else None
    T.clear(accum)
    for reduction_block in T.serial(T.ceildiv(source_width, bk)):
        for i, k in T.Parallel(bm, bk):
            reduction = reduction_block * bk + k
            if routed and not source_grouped:
                route = order[block * bm + i]
                source_row = route // selected
                input_tile[i, k] = T.if_then_else(
                    i < valid_m and route >= 0 and reduction < source_width,
                    source[source_row, reduction],
                    0,
                )
            else:
                input_tile[i, k] = T.if_then_else(
                    i < valid_m and reduction < source_width,
                    source[block * bm + i, reduction],
                    0,
                )
        if coefficients is None:
            prepare_decoded_packets(packets, weight, weight_spec,
                                    expert * output_width + output_block * bn, reduction_block * bk,
                                    (expert + 1) * output_width, source_width, bn, bk, threads)
            T.sync_threads()
            T.copy(input_tile, x)
            publish_decoded_packets(packets, w, bn, bk, threads)
        else:
            load_matrix_tile(w, coefficients, weight, weight_spec,
                             expert * output_width + output_block * bn, reduction_block * bk,
                             (expert + 1) * output_width, source_width, bn, bk, threads)
        affine_gemm(storage, bm, bn, bk, valid_m, affine_has_bias(weight_spec))
    for i, j in T.Parallel(bm, bn):
        channel = output_block * bn + j
        if i < valid_m and channel < output_width:
            output[block * bm + i, channel] = T.cast(accum[i, j], output.dtype)


@T.macro
def _grouped_gated_projection(
    source,
    order,
    block_metadata,
    gate,
    up,
    output,
    gate_spec,
    up_spec,
    blocks,
    source_width,
    output_width,
    selected,
    source_grouped,
    bm,
    bn,
    bk,
    threads,
    arithmetic_dtype,
    routed=True,
    rows=0,
    statistics=None,
    decoded=False,
):
    with T.Kernel(T.ceildiv(output_width, bn), blocks, threads=threads) as (bx, by):
        # One homogeneous tile: route kind and encoding are compile-time inputs.
        storage = (
            affine_storage(bm, 2 * bn, bk, arithmetic_dtype, decoded),
            T.alloc_fragment((bm, bn), "float32"),
        )
        expert = block_metadata[by, 0] if routed else 0
        valid_m = block_metadata[by, 1] if routed else (bm if rows % bm == 0 else T.min(bm, rows - by * bm))
        if expert >= 0:
            _expert_gated_tile(
                source,
                order,
                gate,
                up,
                output,
                gate_spec,
                up_spec,
                expert,
                valid_m,
                by,
                bx,
                source_width,
                output_width,
                selected,
                routed,
                source_grouped,
                bm,
                bn,
                bk,
                threads,
                storage,
                statistics,
            )


@T.macro
def _grouped_projection(
    source,
    order,
    block_metadata,
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
    arithmetic_dtype,
    routed=True,
    rows=0,
    decoded=False,
):
    with T.Kernel(T.ceildiv(output_width, bn), blocks, threads=threads) as (bx, by):
        # One homogeneous tile: route kind and encoding are compile-time inputs.
        storage = affine_storage(bm, bn, bk, arithmetic_dtype, decoded)
        expert = block_metadata[by, 0] if routed else 0
        valid_m = block_metadata[by, 1] if routed else (bm if rows % bm == 0 else T.min(bm, rows - by * bm))
        if expert >= 0:
            _expert_down_tile(
                source,
                order,
                weight,
                output,
                weight_spec,
                expert,
                valid_m,
                by,
                bx,
                source_width,
                output_width,
                selected,
                routed,
                source_grouped,
                bm,
                bn,
                bk,
                threads,
                storage,
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


@T.macro
def _unpermute_shared(
    expert_projected,
    shared_projected,
    inverse,
    scores,
    hidden,
    shared_router,
    output,
    rows,
    selected,
    width,
    dtype,
    threads,
    residual=None,
):
    """Combine routed and gated shared outputs without another launch."""
    with T.Kernel(rows, threads=threads) as row:
        lane = T.get_thread_binding()
        router_partial = T.alloc_local((1,), "float32")
        router_partial[0] = 0.0
        for chunk in T.serial(T.ceildiv(width, threads)):
            column = chunk * threads + lane
            if column < width:
                router_partial[0] += T.cast(hidden[row, column], "float32") * T.cast(
                    shared_router[column], "float32"
                )
        coefficient = T.cast(T.sigmoid(T.warp_reduce_sum(router_partial[0])), dtype)
        for chunk in T.serial(T.ceildiv(width, threads)):
            column = chunk * threads + lane
            if column < width:
                total = T.alloc_local((1,), "float32")
                total[0] = 0.0
                for rank in T.serial(selected):
                    source = inverse[row * selected + rank]
                    if source >= 0:
                        total[0] += T.cast(expert_projected[source, column], "float32") * T.cast(
                            scores[row, rank], "float32"
                        )
                routed_value = T.cast(total[0], dtype)
                shared_value = T.cast(
                    T.cast(coefficient, "float32")
                    * T.cast(shared_projected[row, column], "float32"),
                    dtype,
                )
                publish(output, residual, row, column,
                        T.cast(routed_value, "float32") + T.cast(shared_value, "float32"), dtype)


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
            block_count,
            activation,
            projected,
            gathered,
            *statistics,
        ) = operands[7:]
        statistics = statistics[0] if statistics else None
        rows, width = cast(tuple[int, int], self.specs[0].shape)
        selected = cast(int, self.specs[1].shape[1])
        experts, intermediate, _ = cast(tuple[int, int, int], self.specs[3].shape)
        instruction = self.tile
        bm, bn, bk, threads, arithmetic_dtype, decoded = instruction
        _group_routes(
            routes,
            order,
            inverse,
            block_experts,
            block_count,
            rows,
            selected,
            experts,
            self.capacity,
            bm,
        )
        _prepare_affine_rows(hidden, order, gathered, statistics, rows, self.capacity,
                             width, selected, bm, bk, threads)
        _grouped_gated_projection(
            gathered,
            order,
            block_experts,
            gate,
            up,
            activation,
            self.specs[3],
            self.specs[4],
            self.blocks,
            width,
            intermediate,
            selected,
            True,
            bm,
            bn,
            bk,
            threads,
            arithmetic_dtype,
            statistics=statistics,
            decoded=decoded,
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
            32 if decoded else _packet_reduction_width(self.specs[5]),
            threads,
            arithmetic_dtype,
            decoded=decoded,
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


class _GroupedSharedExpertsEmitter:
    def __init__(self, specs, capacity, expert_blocks, shared_blocks, tile, final_threads, residual=False):
        self.specs = specs
        self.capacity = capacity
        self.expert_blocks = expert_blocks
        self.shared_blocks = shared_blocks
        self.tile = tile
        self.final_threads = final_threads
        self.residual = residual

    def __call__(self, operands: tuple[Any, ...]) -> None:
        residual = operands[10] if self.residual else None
        if self.residual:
            operands = (*operands[:10], *operands[11:])
        (
            hidden,
            routes,
            scores,
            expert_gate,
            expert_up,
            expert_down,
            shared_gate,
            shared_up,
            shared_down,
            shared_router,
            output,
            order,
            inverse,
            block_experts,
            block_count,
            expert_activation,
            shared_activation,
            expert_projected,
            shared_projected,
            *preparation,
        ) = operands
        preparation = iter(preparation)
        gathered = next(preparation)
        decoded = self.tile[5]
        expert_statistics = next(preparation) if not decoded and affine_has_bias(self.specs[3], self.specs[4]) else None
        shared_statistics = next(preparation) if not decoded and affine_has_bias(self.specs[6], self.specs[7]) else None
        rows, width = cast(tuple[int, int], self.specs[0].shape)
        selected = cast(int, self.specs[1].shape[1])
        experts, expert_width, _ = cast(tuple[int, int, int], self.specs[3].shape)
        shared_width = cast(int, self.specs[6].shape[0])
        bm, bn, bk, threads, arithmetic_dtype, decoded = self.tile
        _group_routes(
            routes,
            order,
            inverse,
            block_experts,
            block_count,
            rows,
            selected,
            experts,
            self.capacity,
            bm,
        )
        _prepare_affine_rows(hidden, order, gathered, expert_statistics, rows, self.capacity,
                             width, selected, bm, bk, threads)
        if shared_statistics is not None:
            _prepare_affine_rows(hidden, order, hidden, shared_statistics, rows, rows, width,
                                 selected, bm, _packet_reduction_width(self.specs[6], self.specs[7]),
                                 threads, routed=False)
        # Static, homogeneous grids. No persistent worker owns both packet
        # interpretations or constrains a branch to another branch's group size.
        _grouped_gated_projection(
            gathered, order, block_experts, expert_gate, expert_up, expert_activation,
            self.specs[3], self.specs[4], self.expert_blocks, width, expert_width,
            selected, True, bm, bn, bk, threads, arithmetic_dtype, statistics=expert_statistics, decoded=decoded,
        )
        _grouped_gated_projection(
            hidden, order, block_experts, shared_gate, shared_up, shared_activation,
            self.specs[6], self.specs[7], self.shared_blocks, width, shared_width,
            selected, False, bm, bn, 32 if decoded else _packet_reduction_width(self.specs[6], self.specs[7]),
            threads, arithmetic_dtype, routed=False, rows=rows, statistics=shared_statistics, decoded=decoded,
        )
        _grouped_projection(
            expert_activation, order, block_experts, expert_down, expert_projected,
            self.specs[5], self.expert_blocks, expert_width, width, selected, True,
            bm, bn, 32 if decoded else _packet_reduction_width(self.specs[5]), threads, arithmetic_dtype, decoded=decoded,
        )
        _grouped_projection(
            shared_activation, order, block_experts, shared_down, shared_projected,
            self.specs[8], self.shared_blocks, shared_width, width, selected, True,
            bm, bn, 32 if decoded else _packet_reduction_width(self.specs[8]), threads, arithmetic_dtype,
            routed=False, rows=rows, decoded=decoded,
        )
        _unpermute_shared(
            expert_projected,
            shared_projected,
            inverse,
            scores,
            hidden,
            shared_router,
            output,
            rows,
            selected,
            width,
            self.specs[10].dtype.value,
            self.final_threads,
            residual,
        )


class GroupedExpertsRule:
    name = "grouped-prefill-experts"

    def build(self, graph: Graph, root: int, context: LoweringContext, *, shared: bool = False,
              residual: bool = False):
        if shared and context.mode != "decode":
            combined = _grouped_shared_operation(graph, root, context, residual=residual)
            if combined is not None:
                return (combined,)
        if residual:
            return ()
        node = graph.nodes[root]
        if (
            node.operation != "routed_experts"
            or node.attributes["activation"] != "silu"
            or context.mode == "decode"
            or DType.I32 not in context.capabilities.atomics
            or "shared" not in context.capabilities.memory_scopes
            or "gemm.runtime_valid_m" not in context.capabilities.features
        ):
            return ()
        specs = tuple(graph.values[value].spec for value in (*node.inputs, *node.outputs))
        if any(not spec.static for spec in specs):
            return ()
        hidden, routes, scores, gate, up, down, output = specs
        rows, width = cast(tuple[int, int], hidden.shape)
        selected = cast(int, routes.shape[1])
        experts, intermediate, input_width = cast(tuple[int, int, int], gate.shape)
        gate_packet, up_packet, down_packet = (packet_format(spec) for spec in (gate, up, down))
        instruction = _packet_matrix_instruction(context, hidden.dtype)
        if (
            instruction is None
            or rows * selected < experts
            or input_width != width
            or up.shape != gate.shape
            or down.shape != (experts, width, intermediate)
            or scores.shape != routes.shape
            or output.shape != hidden.shape
            or experts > 256
            or gate_packet is None
            or up_packet is None
            or down_packet is None
            or gate_packet != up_packet
            or width % gate_packet.tile
            or intermediate % down_packet.tile
        ):
            return ()
        if rows >= 256:
            bm, bn = 32, 32
        else:
            bm = instruction.m * 2
            bn = instruction.n * 2
        decoded = _continuous_contraction(context, hidden.dtype, instruction)
        bk = 32 if decoded else _packet_reduction_width(gate, up)
        down_bk = 32 if decoded else _packet_reduction_width(down)
        if bk % instruction.k or down_bk % instruction.k:
            return ()
        threads = min(
            context.capabilities.threads_per_group,
            context.capabilities.subgroup_width * 4,
            bm // instruction.m * context.capabilities.subgroup_width,
        )
        shared = max(
            affine_shared_bytes(bm, 2 * bn, bk, instruction.input_dtype, decoded=decoded),
            affine_shared_bytes(bm, bn, down_bk, instruction.input_dtype, decoded=decoded),
        )
        if shared > context.capabilities.shared_memory_bytes:
            return ()
        capacity = _aligned_capacity(rows, selected, experts, bm)
        blocks = capacity // bm
        workspace = (
            TensorSpec((capacity,), DType.I32),
            TensorSpec((rows * selected,), DType.I32),
            TensorSpec((blocks, 2), DType.I32),
            TensorSpec((1,), DType.I32),
            TensorSpec((capacity, intermediate), hidden.dtype),
            TensorSpec((capacity, width), hidden.dtype),
            TensorSpec((capacity, width), hidden.dtype),
        )
        if not decoded and affine_has_bias(gate, up):
            workspace += (TensorSpec((capacity, width // bk), DType.F32),)
        if sum(value.storage_nbytes for value in workspace) > context.workspace_limit:
            return ()
        return (
            BoundOperation(
                f"routed_experts.grouped@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                _GroupedExpertsEmitter(
                    specs, capacity, blocks, (bm, bn, bk, threads, instruction.input_dtype.value, decoded)
                ),
                workspace=workspace,
                kernel_count=5,
            ),
        )


def _grouped_shared_operation(
    graph: Graph, root: int, context: LoweringContext, *, residual: bool = False,
) -> BoundOperation | None:
    region = _routed_shared_region(graph, root)
    if region is None:
        return None
    nodes, inputs, outputs, specs = region
    if residual:
        epilogue = residual_epilogue(graph, outputs[0])
        if epilogue is None:
            return None
        nodes |= epilogue.nodes
        inputs += (epilogue.residual,)
        outputs = (epilogue.output,)
    if (
        DType.I32 not in context.capabilities.atomics
        or "shared" not in context.capabilities.memory_scopes
        or "gemm.runtime_valid_m" not in context.capabilities.features
    ):
        return None
    hidden, routes, scores = specs[:3]
    expert_gate, expert_up, expert_down = specs[3:6]
    shared_gate, shared_up, shared_down = specs[6:9]
    output = specs[10]
    rows, width = cast(tuple[int, int], hidden.shape)
    selected = cast(int, routes.shape[1])
    experts, expert_width, input_width = cast(tuple[int, int, int], expert_gate.shape)
    shared_width = cast(int, shared_gate.shape[0])
    packets = tuple(
        packet_format(spec)
        for spec in (
            expert_gate,
            expert_up,
            expert_down,
            shared_gate,
            shared_up,
            shared_down,
        )
    )
    instruction = _packet_matrix_instruction(context, hidden.dtype)
    if (
        instruction is None
        or any(packet is None for packet in packets)
        or packets[0] != packets[1]
        or packets[3] != packets[4]
        or rows * selected < experts
        or experts > 256
        or input_width != width
        or expert_up.shape != expert_gate.shape
        or expert_down.shape != (experts, width, expert_width)
        or shared_up.shape != shared_gate.shape
        or shared_gate.shape[1] != width
        or shared_down.shape != (width, shared_width)
        or scores.shape != routes.shape
        or output.shape != hidden.shape
    ):
        return None
    concrete_packets = cast(tuple[Any, ...], packets)
    if (
        width % concrete_packets[0].tile
        or width % concrete_packets[3].tile
        or expert_width % concrete_packets[2].tile
        or shared_width % concrete_packets[5].tile
    ):
        return None
    if rows >= 256:
        bm, bn = 32, 32
    else:
        bm = instruction.m * 2
        bn = instruction.n * 2
    decoded = _continuous_contraction(context, hidden.dtype, instruction)
    bk = 32 if decoded else _packet_reduction_width(expert_gate, expert_up)
    shared_bk = 32 if decoded else _packet_reduction_width(shared_gate, shared_up)
    down_bk = 32 if decoded else _packet_reduction_width(expert_down)
    shared_down_bk = 32 if decoded else _packet_reduction_width(shared_down)
    if any(reduction % instruction.k for reduction in (bk, shared_bk, down_bk, shared_down_bk)):
        return None
    threads = min(
        context.capabilities.threads_per_group,
        context.capabilities.subgroup_width * 4,
        bm // instruction.m * context.capabilities.subgroup_width,
    )
    shared_bytes = max(
        affine_shared_bytes(bm, 2 * bn, bk, instruction.input_dtype, decoded=decoded),
        affine_shared_bytes(bm, 2 * bn, shared_bk, instruction.input_dtype, decoded=decoded),
        affine_shared_bytes(bm, bn, down_bk, instruction.input_dtype, decoded=decoded),
        affine_shared_bytes(bm, bn, shared_down_bk, instruction.input_dtype, decoded=decoded),
    )
    if shared_bytes > context.capabilities.shared_memory_bytes:
        return None
    capacity = _aligned_capacity(rows, selected, experts, bm)
    expert_blocks = capacity // bm
    shared_blocks = (rows + bm - 1) // bm
    workspace = (
        TensorSpec((capacity,), DType.I32),
        TensorSpec((rows * selected,), DType.I32),
        TensorSpec((expert_blocks, 2), DType.I32),
        TensorSpec((1,), DType.I32),
        TensorSpec((capacity, expert_width), hidden.dtype),
        TensorSpec((rows, shared_width), hidden.dtype),
        TensorSpec((capacity, width), hidden.dtype),
        TensorSpec((rows, width), hidden.dtype),
        TensorSpec((capacity, width), hidden.dtype),
    )
    if not decoded and affine_has_bias(expert_gate, expert_up):
        workspace += (TensorSpec((capacity, width // bk), DType.F32),)
    if not decoded and affine_has_bias(shared_gate, shared_up):
        workspace += (TensorSpec((rows, width // shared_bk), DType.F32),)
    if sum(value.storage_nbytes for value in workspace) > context.workspace_limit:
        return None
    return BoundOperation(
        f"routed_experts.grouped@{root}:{max(nodes)}",
        nodes,
        inputs,
        outputs,
        _GroupedSharedExpertsEmitter(
            specs,
            capacity,
            expert_blocks,
            shared_blocks,
            (bm, bn, bk, threads, instruction.input_dtype.value, decoded),
            context.capabilities.subgroup_width,
            residual,
        ),
        workspace=workspace,
        kernel_count=7 + int(not decoded and affine_has_bias(shared_gate, shared_up)),
    )
