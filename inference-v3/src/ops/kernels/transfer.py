"""Portable bounded storage assembly; all writes use TileLang execution."""

from dataclasses import dataclass
import tilelang.language as T

from ..compiler.lowering import BoundOperation


@T.macro
def _byte_copy(source, target, extent, capacity, threads):
    with T.Kernel(T.ceildiv(capacity, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            index = block * threads + lane
            if index < capacity and index < extent[1]:
                target[extent[0] + index] = source[index]


@dataclass(frozen=True, slots=True)
class ByteCopyEmitter:
    capacity: int
    threads: int

    def __call__(self, operands):
        _byte_copy(operands[0], operands[1], operands[2], self.capacity, self.threads)


class ByteCopyRule:
    name = "bounded-byte-transfer"

    def build(self, graph, root, context):
        node = graph.node(root)
        if node.operation != "byte_copy":
            return ()
        return (BoundOperation(f"byte_copy@{root}", frozenset({root}), node.inputs, node.outputs,
                          ByteCopyEmitter(graph.value(node.inputs[0]).spec.elements,
                                          min(256, context.compiler_target.threads_per_group)),
                          aliases=((node.outputs[0], node.inputs[1]),)),)
