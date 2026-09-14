"""Normalization regions that preserve residual values without extra launches."""

from __future__ import annotations

from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..representations import Dense
from ..tensor.graph import Graph
from ..tensor.types import DType, TensorSpec, dense_strides


def _row_coordinates(row, channel, shape: tuple[int, ...]):
    leading = shape[:-1]
    return tuple(
        (row // stride) % extent
        for stride, extent in zip(dense_strides(leading), leading, strict=True)
    ) + (channel,)


def _reduction_threads(width: int, context: LoweringContext) -> int | None:
    subgroup = context.capabilities.subgroup_width
    groups = min(
        (width + subgroup - 1) // subgroup,
        context.capabilities.threads_per_group // subgroup,
    )
    return None if groups < 1 else groups * subgroup


@T.macro
def _residual_rms(
    left,
    right,
    weight,
    residual,
    normalized,
    rows,
    width,
    epsilon,
    threads,
    subgroup_width,
    dtype,
    output_dtype,
):
    with T.Kernel(rows, threads=threads) as row:
        lane = T.get_thread_binding(0)
        warp_sums = T.alloc_shared((T.ceildiv(threads, subgroup_width),), "float32")
        partial = T.alloc_local((1,), "float32")
        partial[0] = 0.0
        for channel in T.serial(T.ceildiv(width, threads)):
            index = channel * threads + lane
            if index < width:
                # The add's dtype boundary is observable by both consumers:
                # the retained residual and the following normalization.
                value = T.cast(
                    T.cast(
                        T.cast(left[row, index], "float32") + T.cast(right[row, index], "float32"),
                        dtype,
                    ),
                    "float32",
                )
                residual[row, index] = T.cast(value, dtype)
                partial[0] += value * value
        reduced = T.warp_reduce_sum(partial[0])
        if lane % subgroup_width == 0:
            warp_sums[lane // subgroup_width] = reduced
        T.sync_threads()
        partial[0] = 0.0
        for warp in T.unroll(T.ceildiv(threads, subgroup_width), explicit=True):
            partial[0] += warp_sums[warp]
        inverse = T.rsqrt(partial[0] / width + epsilon)
        for channel in T.serial(T.ceildiv(width, threads)):
            index = channel * threads + lane
            if index < width:
                normalized[row, index] = T.cast(
                    T.cast(residual[row, index], "float32")
                    * inverse
                    * T.cast(weight[index], "float32"),
                    output_dtype,
                )


class _ResidualRMSEmitter:
    def __init__(
        self,
        specs: tuple[TensorSpec, ...],
        epsilon: float,
        threads: int,
        subgroup_width: int,
        residual_dtype: DType,
        output_dtype: DType,
    ):
        self.specs = specs
        self.epsilon = epsilon
        self.threads = threads
        self.subgroup_width = subgroup_width
        self.residual_dtype, self.output_dtype = residual_dtype, output_dtype

    def __call__(self, operands: tuple[Any, ...]) -> None:
        rows = self.specs[0].elements // cast(int, self.specs[0].shape[-1])
        width = cast(int, self.specs[0].shape[-1])
        _residual_rms(
            operands[0],
            operands[1],
            operands[2],
            operands[3],
            operands[4],
            rows,
            width,
            self.epsilon,
            self.threads,
            self.subgroup_width,
            self.residual_dtype.value,
            self.output_dtype.value,
        )


class ResidualRMSRule:
    name = "residual-rms"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        add = graph.nodes[root]
        if add.operation != "add" or len(add.outputs) != 1:
            return ()
        residual = add.outputs[0]
        consumers = tuple(graph.users[residual])
        norm_ids = tuple(
            node_id for node_id in consumers if graph.nodes[node_id].operation == "rms_norm"
        )
        if len(norm_ids) != 1:
            return ()
        norm = graph.nodes[norm_ids[0]]
        if len(norm.inputs) != 2 or norm.inputs[0] != residual:
            return ()
        region = {add.id, norm.id}
        add_inputs = list(add.inputs)
        residual_dtype = graph.values[residual].spec.dtype
        for index, value_id in enumerate(add_inputs):
            value = graph.values[value_id]
            if (
                value.producer is None
                or value_id in graph.outputs
                or graph.users[value_id] != (add.id,)
            ):
                continue
            producer = graph.nodes[value.producer]
            if producer.operation != "cast" or residual_dtype != DType.F32:
                continue
            source = graph.values[producer.inputs[0]].spec
            if source.dtype.floating and source.dtype.itemsize <= residual_dtype.itemsize:
                region.add(producer.id)
                add_inputs[index] = producer.inputs[0]
        nodes = frozenset(region)
        if nodes != frozenset(range(min(nodes), norm.id + 1)):
            return ()
        specs = tuple(graph.values[value].spec for value in (*add_inputs, norm.inputs[1]))
        if (
            any(not spec.static for spec in specs)
            or len({specs[0].shape, specs[1].shape}) != 1
            or specs[0].rank != 2
            or any(
                spec.representation is not None and not isinstance(spec.representation, Dense)
                for spec in specs
            )
            or "shared" not in context.capabilities.memory_scopes
        ):
            return ()
        threads = _reduction_threads(cast(int, specs[0].shape[-1]), context)
        if threads is None:
            return ()
        inputs = (*add_inputs, norm.inputs[1])
        outputs = tuple(
            value
            for value in (residual, *norm.outputs)
            if value in graph.outputs or any(user not in nodes for user in graph.users[value])
        )
        # The residual is normally consumed by the following skip connection; keep
        # it explicit even when the traced function ends at the normalization.
        if residual not in outputs:
            return ()
        moved = sum(graph.values[value].spec.storage_nbytes for value in (*inputs, *outputs))
        return (
            Candidate(
                f"residual_rms.fused@{root}:{norm.id}",
                nodes,
                inputs,
                outputs,
                _ResidualRMSEmitter(
                    specs,
                    norm.attributes["epsilon"],
                    threads,
                    context.capabilities.subgroup_width,
                    residual_dtype,
                    graph.values[norm.outputs[0]].spec.dtype,
                ),
                4e-7 + moved / 180e9,
                priority=40,
            ),
        )


@T.macro
def _rms(
    source, weight, output, shape, rows, width, epsilon, threads, subgroup_width, dtype, weighted
):
    with T.Kernel(rows, threads=threads) as row:
        lane = T.get_thread_binding()
        warp_sums = T.alloc_shared((T.ceildiv(threads, subgroup_width),), "float32")
        partial = T.alloc_local((1,), "float32")
        partial[0] = 0.0
        for chunk in T.serial(T.ceildiv(width, threads)):
            channel = chunk * threads + lane
            if channel < width:
                coordinates = _row_coordinates(row, channel, shape)
                value = T.cast(source[coordinates], "float32")
                partial[0] += value * value
        reduced = T.warp_reduce_sum(partial[0])
        if lane % subgroup_width == 0:
            warp_sums[lane // subgroup_width] = reduced
        T.sync_threads()
        partial[0] = 0.0
        for warp in T.unroll(T.ceildiv(threads, subgroup_width), explicit=True):
            partial[0] += warp_sums[warp]
        inverse = T.rsqrt(partial[0] / width + epsilon)
        for chunk in T.serial(T.ceildiv(width, threads)):
            channel = chunk * threads + lane
            if channel < width:
                if weighted:
                    coordinates = _row_coordinates(row, channel, shape)
                    output[coordinates] = T.cast(
                        T.cast(source[coordinates], "float32")
                        * inverse
                        * T.cast(weight[channel], "float32"),
                        dtype,
                    )
                else:
                    coordinates = _row_coordinates(row, channel, shape)
                    output[coordinates] = T.cast(
                        T.cast(source[coordinates], "float32") * inverse,
                        dtype,
                    )


class _RMSEmitter:
    def __init__(self, spec, epsilon, threads, subgroup_width, weighted, output_dtype):
        self.spec, self.epsilon = spec, epsilon
        self.threads, self.subgroup_width, self.weighted = threads, subgroup_width, weighted
        self.output_dtype = output_dtype

    def __call__(self, operands: tuple[Any, ...]) -> None:
        rows = self.spec.elements // cast(int, self.spec.shape[-1])
        width = cast(int, self.spec.shape[-1])
        weight = operands[1] if self.weighted else operands[0]
        output = operands[2] if self.weighted else operands[1]
        _rms(
            operands[0],
            weight,
            output,
            self.spec.shape,
            rows,
            width,
            self.epsilon,
            self.threads,
            self.subgroup_width,
            self.output_dtype.value,
            self.weighted,
        )


class RMSRule:
    name = "parallel-rms"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation != "rms_norm" or "shared" not in context.capabilities.memory_scopes:
            return ()
        specs = tuple(graph.values[value].spec for value in (*node.inputs, *node.outputs))
        source = specs[0]
        if any(not spec.static for spec in specs) or source.rank < 2:
            return ()
        width = cast(int, source.shape[-1])
        threads = _reduction_threads(width, context)
        if threads is None:
            return ()
        moved = sum(spec.storage_nbytes for spec in specs)
        return (
            Candidate(
                f"rms_norm.parallel@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                _RMSEmitter(
                    source,
                    node.attributes["epsilon"],
                    threads,
                    context.capabilities.subgroup_width,
                    len(node.inputs) == 2,
                    specs[-1].dtype,
                ),
                3e-7 + moved / 180e9,
                priority=40,
            ),
        )


@T.macro
def _row_dot(source, weight, output, rows, width, threads, dtype):
    with T.Kernel(rows, threads=threads) as row:
        lane = T.get_thread_binding()
        partial = T.alloc_local((1,), "float32")
        partial[0] = 0.0
        for chunk in T.serial(T.ceildiv(width, threads)):
            channel = chunk * threads + lane
            if channel < width:
                partial[0] += T.cast(source[row, channel], "float32") * T.cast(
                    weight[channel], "float32"
                )
        total = T.warp_reduce_sum(partial[0])
        if lane == 0:
            output[row, 0] = T.cast(total, dtype)


class _RowDotEmitter:
    def __init__(self, rows, width, threads, dtype):
        self.args = rows, width, threads, dtype

    def __call__(self, operands: tuple[Any, ...]) -> None:
        _row_dot(operands[0], operands[1], operands[2], *self.args)


class RowDotRule:
    name = "subgroup-row-dot"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation != "row_dot":
            return ()
        source = graph.values[node.inputs[0]].spec
        if not source.static or source.rank != 2:
            return ()
        rows, width = cast(tuple[int, int], source.shape)
        output = graph.values[node.outputs[0]].spec
        return (
            Candidate(
                f"row_dot.subgroup@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                _RowDotEmitter(
                    rows, width, context.capabilities.subgroup_width, output.dtype.value
                ),
                2e-7 + source.storage_nbytes / 200e9,
                priority=40,
            ),
        )


__all__ = ["RMSRule", "ResidualRMSRule", "RowDotRule"]
