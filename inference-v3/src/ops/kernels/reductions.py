"""Portable standalone reduction/scan bodies for independently measured formulas."""

import math
from dataclasses import dataclass

import tilelang.language as T

from ..compiler.lowering import BoundOperation
from ..tensor.types import DType, TensorSpec
from .portable import _indices


def _coordinates(row, column, shape, axis, inner):
    return (*_indices(row // inner, shape[:axis]), column,
            *_indices(row % inner, shape[axis + 1:]))


@T.macro
def _softmax_first(source, output, statistics, shape, axis, inner, rows, width, tile, parts, threads):
    with T.Kernel(rows, parts, threads=threads) as (row, part):
        values = T.alloc_fragment((tile,), "float32")
        maximum = T.alloc_fragment((1,), "float32")
        total = T.alloc_fragment((1,), "float32")
        for lane in T.Parallel(tile):
            column = part * tile + lane
            values[lane] = T.if_then_else(column < width,
                T.cast(source[_coordinates(row, column, shape, axis, inner)], "float32"), -math.inf)
        T.reduce_max(values, maximum, dim=0)
        for lane in T.Parallel(tile):
            values[lane] = T.if_then_else(part * tile + lane < width, T.exp(values[lane] - T.if_then_else(maximum[0] == -math.inf, 0.0, maximum[0])), 0.0)
        T.reduce_sum(values, total, dim=0)
        if parts == 1:
            for lane in T.Parallel(tile):
                if lane < width:
                    output[_coordinates(row, lane, shape, axis, inner)] = values[lane] / total[0]
        else:
            if T.get_thread_binding() == 0:
                statistics[row, part, 0] = maximum[0]
                statistics[row, part, 1] = total[0]


@T.macro
def _softmax_finish(source, statistics, output, shape, axis, inner, rows, width, tile, parts, reduction, threads):
    with T.Kernel(rows, parts, threads=threads) as (row, part):
        maxima = T.alloc_fragment((reduction,), "float32")
        weights = T.alloc_fragment((reduction,), "float32")
        maximum = T.alloc_fragment((1,), "float32")
        total = T.alloc_fragment((1,), "float32")
        for lane in T.Parallel(reduction):
            maxima[lane] = T.if_then_else(lane < parts, statistics[row, lane, 0], -math.inf)
        T.reduce_max(maxima, maximum, dim=0)
        for lane in T.Parallel(reduction):
            weights[lane] = T.if_then_else(lane < parts,
                statistics[row, lane, 1] * T.exp(maxima[lane] - maximum[0]), 0.0)
        T.reduce_sum(weights, total, dim=0)
        for lane in T.Parallel(tile):
            column = part * tile + lane
            if column < width:
                coordinates = _coordinates(row, column, shape, axis, inner)
                output[coordinates] = T.exp(T.cast(source[coordinates], "float32") - maximum[0]) / total[0]


@dataclass(frozen=True, slots=True)
class _SoftmaxEmitter:
    spec: TensorSpec
    axis: int
    tile: int
    threads: int

    def __call__(self, operands):
        source, output = operands[:2]
        width = self.spec.shape[self.axis]
        rows = self.spec.elements // width
        inner = math.prod(self.spec.shape[self.axis + 1:])
        parts = math.ceil(width / self.tile)
        statistics = operands[2] if parts > 1 else output
        _softmax_first(source, output, statistics, self.spec.shape, self.axis, inner,
                       rows, width, self.tile, parts, self.threads)
        if parts > 1:
            _softmax_finish(source, statistics, output, self.spec.shape, self.axis, inner,
                            rows, width, self.tile, parts, 1 << (parts - 1).bit_length(), self.threads)


class SoftmaxRule:
    def build(self, graph, root, context):
        node = graph.node(root)
        spec = graph.value(node.inputs[0]).spec
        axis = node.attributes["axis"] % spec.rank
        width = spec.shape[axis]
        if width <= 0:
            raise ValueError("softmax reduction must contain at least one element")
        tile = min(4096, 1 << (width - 1).bit_length())
        parts = math.ceil(width / tile)
        workspace = (TensorSpec((spec.elements // width, parts, 2), DType.F32),) if parts > 1 else ()
        return (BoundOperation(f"softmax.tiled@{root}", frozenset({root}), node.inputs, node.outputs,
                               _SoftmaxEmitter(spec, axis, tile, min(256, context.compiler_target.threads_per_group)),
                               workspace=workspace, kernel_count=2 if parts > 1 else 1),)


@T.macro
def _affine_scan(values, state, decay, output, updated, rows, width, state_rank, decay_rank, threads):
    with T.Kernel(T.ceildiv(width, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            channel = block * threads + lane
            if channel < width:
                carried = T.alloc_local((1,), "float32")
                carried[0] = T.cast(state[0, channel] if state_rank == 2 else state[channel], "float32")
                for row in T.serial(rows):
                    factor = (T.cast(decay[row, channel] if decay_rank == 2 else decay[channel], "float32")
                              if decay_rank else 0.0)
                    carried[0] = carried[0] * factor + T.cast(values[row, channel], "float32")
                    output[row, channel] = carried[0]
                if state_rank == 2:
                    updated[0, channel] = carried[0]
                else:
                    updated[channel] = carried[0]


@dataclass(frozen=True, slots=True)
class _AffineScanEmitter:
    shape: tuple[int, ...]
    state_rank: int
    decay_rank: int
    threads: int

    def __call__(self, operands):
        count = 3 if self.decay_rank else 2
        _affine_scan(operands[0], operands[1], operands[2] if self.decay_rank else operands[0],
                     operands[count], operands[count + 1], *self.shape, self.state_rank,
                     self.decay_rank, self.threads)


class AffineScanRule:
    def build(self, graph, root, context):
        node = graph.node(root)
        values, state = (graph.value(value).spec for value in node.inputs[:2])
        decay_rank = graph.value(node.inputs[2]).spec.rank if len(node.inputs) == 3 else 0
        return (BoundOperation(f"delta.affine-scan@{root}", frozenset({root}), node.inputs, node.outputs,
                               _AffineScanEmitter(values.shape, state.rank, decay_rank,
                                                  min(256, context.compiler_target.threads_per_group)),
                               aliases=((node.outputs[1], node.inputs[1]),)),)
