"""Rewrite a container's K-quant blocks into the shared planar-affine layout.

The container is read through the ordinary group interpretation; the planes are
written at the offsets every planar-affine consumer reads from.
"""

import tilelang.language as T

from magnitude_engine.kernels.projection.decode import interpretation
from magnitude_engine.kernels.projection.planar_affine.layout import check
from magnitude_engine.weights.representation import (
    Blocked,
    Encoding,
    PlanarAffine,
    plane_offsets,
)


def pack(blocks: int, elements: int, encoding: Encoding, representation: PlanarAffine):
    """Place a bounded source chunk into final planes at a runtime element offset."""
    if blocks <= 0 or elements < blocks * 256 or elements % 256:
        raise ValueError("packing requires complete K superblocks")
    check(representation)
    group_elements = representation.group
    high_bits = representation.high_bits
    zero_point = representation.zero_point
    groups = blocks * 256 // group_elements
    offsets = plane_offsets(representation, elements)
    assert offsets.low == 0  # the low plane is the origin of the allocation
    words = offsets.words
    parameters, code = interpretation(Blocked(encoding))

    @T.prim_func
    def main(
        A: T.Tensor((blocks * (encoding.block_bytes // 2),), "uint16"),
        B: T.Tensor((words,), "uint32"),
        Offset: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(T.ceildiv(groups, 128), threads=128) as block:
            group = block * 128 + T.get_thread_binding()
            low = T.alloc_local((1,), "uint32")
            high = T.alloc_local((1,), "uint32")
            if group < groups:
                first = Offset[0] + group * group_elements
                scale, bias = parameters(A, group * group_elements)
                B[offsets.scales + first // group_elements] = T.reinterpret(scale, "uint32")
                if zero_point == 0:
                    B[offsets.biases + first // group_elements] = T.reinterpret(bias, "uint32")
                high[0] = 0
                for packet in T.unroll(group_elements // 8, explicit=True):
                    low[0] = 0
                    for j in T.unroll(8, explicit=True):
                        value = (
                            code(A, group * group_elements + packet * 8 + j) + zero_point
                        ).astype("uint32")
                        low[0] |= (value & T.uint32(15)) << T.uint32(j * 4)
                        if high_bits:
                            high[0] |= (value >> 4) << T.uint32((packet * 8 + j) * high_bits)
                    B[first // 8 + packet] = low[0]
                if high_bits:
                    B[offsets.high + first // group_elements] = high[0]

    return main
