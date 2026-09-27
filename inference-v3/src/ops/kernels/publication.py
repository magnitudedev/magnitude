"""Absorb a declared residual publication without changing numerical boundaries."""

from dataclasses import dataclass

import tilelang.language as T

from ..tensor.graph import Graph
from ..tensor.types import DType


@dataclass(frozen=True, slots=True)
class ResidualEpilogue:
    nodes: frozenset[int]
    residual: int
    output: int


def residual_epilogue(graph: Graph, value: int) -> ResidualEpilogue | None:
    """An unobserved result, optional FP32 widening, then a same-shape FP32 add.

    The caller owns the enclosing formula. Child measurements expose the original
    result and therefore cannot silently absorb the parent's publication.
    """
    source = graph.value(value).spec
    if not source.dtype.floating:
        return None
    nodes = set()
    while True:
        if value in graph.outputs or len(graph.users[value]) != 1:
            return None
        node = graph.node(graph.users[value][0])
        if node.operation == "cast" and not nodes:
            following = graph.value(node.outputs[0]).spec
            if following.dtype != DType.F32 or following.shape != source.shape:
                return None
            nodes.add(node.id)
            value = node.outputs[0]
            continue
        if node.operation != "add" or len(node.inputs) != 2:
            return None
        residual = node.inputs[1] if node.inputs[0] == value else node.inputs[0]
        output = graph.value(node.outputs[0]).spec
        skip = graph.value(residual).spec
        if (output.dtype != DType.F32 or skip.dtype != DType.F32 or
                output.shape != source.shape or skip.shape != source.shape):
            return None
        return ResidualEpilogue(frozenset((*nodes, node.id)), residual, node.outputs[0])


@T.macro
def publish(result, residual, row, column, value, dtype):
    # The original projection/expert formula publishes in dtype BEFORE the
    # residual widening. Fusing storage must not erase that numerical rounding.
    rounded = T.cast(value, dtype)
    if residual is None:
        result[row, column] = rounded
    else:
        result[row, column] = T.cast(residual[row, column], "float32") + T.cast(rounded, "float32")
