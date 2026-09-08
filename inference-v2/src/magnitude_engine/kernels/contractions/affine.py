"""Canonical affine arithmetic with reusable inline or materialized input preparation."""

import mlx.core as mx

from .. import kernel
from ..core.graph import Tensor
from ..core.metal import (
    ArgumentType,
    BlockedRows,
    ColumnStart,
    Dispatch,
    FragmentFold,
    Lane,
    MetalType,
    ReadOnly,
    RowIndices,
)
from ..core.plan import Launch, Source
from .tiles import ENCODED, packing, row_tile

AFFINE = Source("contractions/affine.metal", (ENCODED, Source("core/fragments.metal")))
PREPARE = Source("contractions/affine_input.metal", (Source("contractions/encoded.metal"),))


def whole(tensor):
    return ReadOnly(tensor[(slice(None),) * tensor.ndim])


@kernel(source=PREPARE)
def prepare_input(x, *, bits, pack):
    if bits not in (4, 8) or pack not in (32 // bits, 64 // bits):
        raise ValueError("unsupported affine preparation packing")
    if x.shape[-1] % pack:
        raise ValueError("input width must contain whole affine packs")
    rows, width = x.size // x.shape[-1], x.shape[-1]
    return Dispatch(
        {"x": x},
        {
            "prepared": Tensor(x.shape, mx.float32),
            "sums": Tensor((rows, width // pack), mx.float32),
        },
        Launch((rows * width // pack, 1, 1), (256, 1, 1)),
        (("T", x.dtype), ("M", rows), ("K", width), ("BITS", bits), ("PACK", pack)),
    )


@kernel(source=AFFINE, function="magnitude_affine_fold")
def affine(x, w, s, b, sums, *, bits, group_size, prepared=False):
    """Fixed lane traversal, FP32 accumulation and one native output conversion."""
    k, n = x.shape[-1], w.shape[-2]
    native = s.dtype
    if (
        w.ndim != 2
        or w.dtype != mx.uint32
        or w.shape[-1] * 32 != k * bits
        or s.value.tensor != b.value.tensor
        or s.shape != (n, k // group_size)
        or native not in (mx.float16, mx.bfloat16, mx.float32)
        or x.dtype != (mx.float32 if prepared else native)
        or packing(k, n, bits, group_size) is None
    ):
        raise ValueError("unsupported canonical affine geometry or encoding")
    pack = packing(k, n, bits, group_size)
    assert pack is not None
    count = x.size // k
    if prepared and (sums.shape != (count, k // pack) or sums.dtype != mx.float32):
        raise ValueError("prepared sums disagree with the affine input geometry")
    output = Tensor((*x.shape[:-1], n), native)
    rows = row_tile(count, 4)
    source = whole(x)
    return FragmentFold(
        output,
        BlockedRows(count, n, rows),
        dict(x=source, sums=whole(sums), rows=RowIndices(), first=ColumnStart(), lane=Lane()),
        (native, bits, pack, k, rows, prepared, ArgumentType(source), ArgumentType(whole(sums))),
        MetalType(
            "AffineStep",
            (native, bits, pack, k, n, group_size, rows),
            (whole(w), whole(s), whole(b)),
        ),
        pack,
    )
