"""An ordinary authored operation, not a second benchmark execution path."""

import tilelang.language as T

from .helpers import tile_width


def square(context):
    length = context.inputs[0].spec.shape[0]
    width = tile_width()

    @T.macro
    def kernel(source, output):
        with T.Kernel(T.ceildiv(length, width), threads=width) as block:
            for lane in T.Parallel(width):
                index = block * width + lane
                if index < length:
                    output[index] = source[index] * source[index]

    return context.kernel(kernel)
