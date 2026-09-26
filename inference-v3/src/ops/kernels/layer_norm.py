"""Two-pass centered normalization with row-local FP32 reductions."""

import tilelang.language as T

from ..compiler.lowering import BoundOperation
from .normalization import _reduction_threads, _row_coordinates


@T.macro
def _layer_norm(
    source, weight, bias, output, shape, rows, width, epsilon, threads, subgroup, dtype
):
    with T.Kernel(rows, threads=threads) as row:
        lane = T.get_thread_binding()
        sums = T.alloc_shared((T.ceildiv(threads, subgroup),), "float32")
        partial = T.alloc_local((1,), "float32")
        mean = T.alloc_local((1,), "float32")
        partial[0] = 0.0
        for chunk in T.serial(T.ceildiv(width, threads)):
            channel = chunk * threads + lane
            if channel < width:
                partial[0] += T.cast(source[_row_coordinates(row, channel, shape)], "float32")
        reduced = T.warp_reduce_sum(partial[0])
        if lane % subgroup == 0:
            sums[lane // subgroup] = reduced
        T.sync_threads()
        partial[0] = 0.0
        for group in T.unroll(T.ceildiv(threads, subgroup), explicit=True):
            partial[0] += sums[group]
        mean[0] = partial[0] / width
        # Everyone must finish reading sums before the variance overwrites it.
        T.sync_threads()
        partial[0] = 0.0
        for chunk in T.serial(T.ceildiv(width, threads)):
            channel = chunk * threads + lane
            if channel < width:
                value = T.cast(source[_row_coordinates(row, channel, shape)], "float32") - mean[0]
                partial[0] += value * value
        reduced_variance = T.warp_reduce_sum(partial[0])
        if lane % subgroup == 0:
            sums[lane // subgroup] = reduced_variance
        T.sync_threads()
        partial[0] = 0.0
        for group in T.unroll(T.ceildiv(threads, subgroup), explicit=True):
            partial[0] += sums[group]
        inverse = T.rsqrt(partial[0] / width + epsilon)
        for chunk in T.serial(T.ceildiv(width, threads)):
            channel = chunk * threads + lane
            if channel < width:
                coordinates = _row_coordinates(row, channel, shape)
                output[coordinates] = T.cast(
                    (T.cast(source[coordinates], "float32") - mean[0])
                    * inverse
                    * T.cast(weight[channel], "float32")
                    + T.cast(bias[channel], "float32"),
                    dtype,
                )


class _Emitter:
    def __init__(self, shape, epsilon, threads, subgroup, dtype):
        self.shape, self.epsilon = shape, epsilon
        self.threads, self.subgroup, self.dtype = threads, subgroup, dtype

    def __call__(self, operands):
        import math

        _layer_norm(
            *operands,
            self.shape,
            math.prod(self.shape[:-1]),
            self.shape[-1],
            self.epsilon,
            self.threads,
            self.subgroup,
            self.dtype.value,
        )


class LayerNormRule:
    def build(self, graph, root, context):
        node = graph.nodes[root]
        specs = tuple(graph.values[value].spec for value in (*node.inputs, *node.outputs))
        if node.operation != "layer_norm" or any(not spec.static for spec in specs):
            return ()
        source = specs[0]
        threads = _reduction_threads(source.shape[-1], context)
        if threads is None or context.compiler_target.shared_memory_bytes < (
            threads // context.compiler_target.subgroup_width * 4
        ):
            return ()
        return (
            BoundOperation(
                f"layer_norm.centered@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                _Emitter(
                    source.shape,
                    node.attributes["epsilon"],
                    threads,
                    context.compiler_target.subgroup_width,
                    specs[-1].dtype,
                ),
            ),
        )
