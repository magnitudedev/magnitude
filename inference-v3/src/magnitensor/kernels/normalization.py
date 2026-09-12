"""Normalization regions that preserve residual values without extra launches."""

from __future__ import annotations

from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..representations import Dense
from ..tensor.graph import Graph
from ..tensor.types import TensorSpec


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
    dtype,
):
    with T.Kernel(rows, threads=threads) as row:
        lane = T.get_thread_binding(0)
        squares = T.alloc_shared((threads,), "float32")
        partial = T.alloc_local((1,), "float32")
        partial[0] = 0.0
        for channel in T.serial(T.ceildiv(width, threads)):
            index = channel * threads + lane
            if index < width:
                value = T.cast(left[row, index], "float32") + T.cast(right[row, index], "float32")
                residual[row, index] = T.cast(value, dtype)
                partial[0] += value * value
        squares[lane] = partial[0]
        T.sync_threads()
        for reduction in T.unroll(threads.bit_length() - 1):
            distance = threads >> (reduction + 1)
            if lane < distance:
                squares[lane] += squares[lane + distance]
            T.sync_threads()
        inverse = T.rsqrt(squares[0] / width + epsilon)
        for channel in T.serial(T.ceildiv(width, threads)):
            index = channel * threads + lane
            if index < width:
                normalized[row, index] = T.cast(
                    T.cast(residual[row, index], "float32")
                    * inverse
                    * T.cast(weight[index], "float32"),
                    dtype,
                )


class _ResidualRMSEmitter:
    def __init__(self, specs: tuple[TensorSpec, ...], epsilon: float, threads: int):
        self.specs = specs
        self.epsilon = epsilon
        self.threads = threads

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
            self.specs[0].dtype.value,
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
        specs = tuple(graph.values[value].spec for value in (*add.inputs, norm.inputs[1]))
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
        threads = min(
            1 << (cast(int, specs[0].shape[-1]) - 1).bit_length(),
            context.capabilities.threads_per_group,
        )
        nodes = frozenset((root, norm.id))
        inputs = (*add.inputs, norm.inputs[1])
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
                _ResidualRMSEmitter(specs, norm.attributes["epsilon"], threads),
                4e-7 + moved / 180e9,
                priority=40,
            ),
        )
