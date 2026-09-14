"""Generic pointwise fusion using Python-composed TileLang expressions."""

from __future__ import annotations

from typing import Any

import tilelang.language as T

from ..compiler.lowering import BoundOperation
from ..representations import Dense
from ..tensor.graph import Graph
from .portable import _indices, _load

_SUPPORTED = {
    "scalar",
    "add",
    "subtract",
    "multiply",
    "divide",
    "less",
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

    def specialization_key(self):
        values = {value: index for index, value in enumerate(self.inputs)}
        nodes = []
        for node_id in sorted(self.nodes):
            node = self.graph.node(node_id)
            inputs = tuple(values[value] for value in node.inputs)
            for value in node.outputs:
                values[value] = len(values)
            nodes.append((node.operation, inputs, tuple(values[value] for value in node.outputs),
                          dict(node.attributes)))
        return (tuple(nodes), tuple(self.graph.value(value).spec for value in self.inputs),
                self.graph.value(self.output).spec, self.threads)

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
                elif node.operation == "less":
                    result = args[0] < args[1]
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
                # Fusion preserves every declared intermediate rounding boundary.
                dtype = self.graph.value(node.outputs[0]).spec.dtype.value
                # Preserve the authored DAG as scalar temporaries, not one
                # recursively substituted expression. Nested comparisons otherwise
                # make canonical simplification repeatedly revisit the entire
                # prefix. These are thread-local scalars, not global publications.
                temporary = T.alloc_var(dtype, init=T.cast(result, dtype))
                values[node.outputs[0]] = temporary[0]
            return values[self.output]

        _fused_pointwise(output, expression, output_spec, self.threads)


def pointwise(context):
    """Fuse the explicitly assigned formula boundary, with no region search."""
    graph, nodes = context.graph, context.nodes
    if not nodes or any(graph.node(node).operation not in _SUPPORTED or
                        graph.node(node).effects.reads or graph.node(node).effects.writes
                        for node in nodes):
        raise ValueError("pointwise body requires a nonempty pure elementwise formula")
    inputs = tuple(value.id for value in context.inputs)
    outputs = tuple(value.id for value in context.outputs)
    if len(outputs) != 1:
        raise ValueError("pointwise body requires one observable output")
    output_spec = graph.value(outputs[0]).spec
    if not output_spec.shape:
        raise ValueError("pointwise body requires a tensor output")
    if any(value.spec.representation is not None and not isinstance(value.spec.representation, Dense)
           for value in (*context.inputs, *context.outputs)):
        raise ValueError("pointwise body requires dense physical ports")
    emitter = _PointwiseEmitter(graph, nodes, inputs, outputs[0],
                               min(256, context.lowering.capabilities.threads_per_group))
    return (BoundOperation(f"pointwise.fused@{min(nodes)}:{max(nodes)}", nodes, inputs, outputs, emitter),)
