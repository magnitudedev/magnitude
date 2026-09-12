"""Generic pointwise fusion using Python-composed TileLang expressions."""

from __future__ import annotations

from typing import Any

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..representations import Dense
from ..tensor.graph import Graph
from .portable import _indices, _load

_SUPPORTED = {
    "scalar",
    "add",
    "subtract",
    "multiply",
    "divide",
    "cast",
    "exp",
    "sigmoid",
    "silu",
    "tanh",
}


@T.macro
def _fused_pointwise(output, expression, output_spec, threads):
    with T.Kernel(T.ceildiv(output_spec.elements, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < output_spec.elements:
                output[_indices(flat, output_spec.shape)] = expression(flat)


class _PointwiseEmitter:
    def __init__(
        self,
        graph: Graph,
        nodes: frozenset[int],
        inputs: tuple[int, ...],
        output: int,
        threads: int,
    ):
        self.graph = graph
        self.nodes = nodes
        self.inputs = inputs
        self.output = output
        self.threads = threads

    def __call__(self, operands: tuple[Any, ...]) -> None:
        input_buffers = dict(zip(self.inputs, operands[: len(self.inputs)], strict=True))
        output = operands[-1]
        output_spec = self.graph.values[self.output].spec

        def expression(flat):
            values = {
                value: _load(buffer, self.graph.values[value].spec, flat, output_spec)
                for value, buffer in input_buffers.items()
            }
            for node_id in sorted(self.nodes):
                node = self.graph.nodes[node_id]
                args = tuple(values[value] for value in node.inputs)
                if node.operation == "scalar":
                    result = node.attributes["value"]
                elif node.operation == "add":
                    result = args[0] + args[1]
                elif node.operation == "subtract":
                    result = args[0] - args[1]
                elif node.operation == "multiply":
                    result = args[0] * args[1]
                elif node.operation == "divide":
                    result = args[0] / args[1]
                elif node.operation == "cast":
                    result = T.cast(args[0], node.attributes["dtype"].value)
                elif node.operation == "exp":
                    result = T.exp(args[0])
                elif node.operation == "sigmoid":
                    result = T.sigmoid(args[0])
                elif node.operation == "silu":
                    result = args[0] * T.sigmoid(args[0])
                elif node.operation == "tanh":
                    result = T.tanh(args[0])
                else:
                    raise AssertionError(node.operation)
                values[node.outputs[0]] = result
            return values[self.output]

        _fused_pointwise(output, expression, output_spec, self.threads)


class PointwiseFusionRule:
    name = "generic-pointwise-fusion"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        if graph.nodes[root].operation not in _SUPPORTED:
            return ()
        region: set[int] = set()
        results = []
        for node_id in range(root, min(len(graph.nodes), root + 16)):
            node = graph.nodes[node_id]
            if node.operation not in _SUPPORTED or node.effects.reads or node.effects.writes:
                break
            if region and not any(graph.values[value].producer in region for value in node.inputs):
                break
            region.add(node_id)
            if len(region) < 2:
                continue
            nodes = frozenset(region)
            inputs = tuple(
                dict.fromkeys(
                    value
                    for current in sorted(nodes)
                    for value in graph.nodes[current].inputs
                    if graph.values[value].producer not in nodes
                )
            )
            outputs = tuple(
                value
                for current in sorted(nodes)
                for value in graph.nodes[current].outputs
                if value in graph.outputs or any(user not in nodes for user in graph.users[value])
            )
            if len(outputs) != 1:
                continue
            output_spec = graph.values[outputs[0]].spec
            if not output_spec.shape:
                continue
            if any(
                spec.representation is not None and not isinstance(spec.representation, Dense)
                for spec in (graph.values[value].spec for value in (*inputs, *outputs))
            ):
                continue
            moved = sum(graph.values[value].spec.storage_nbytes for value in (*inputs, *outputs))
            results.append(
                Candidate(
                    f"pointwise.fused@{root}:{node_id}",
                    nodes,
                    inputs,
                    outputs,
                    _PointwiseEmitter(
                        graph,
                        nodes,
                        inputs,
                        outputs[0],
                        min(256, context.capabilities.threads_per_group),
                    ),
                    1e-6 + moved / 100e9,
                    priority=len(nodes),
                )
            )
        return tuple(results)
