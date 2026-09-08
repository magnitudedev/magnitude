"""Selected dots share route grouping/input traversal; route sums retain slot order."""

from .. import kernel
from ..core.graph import Tensor
from ..core.metal import (
    ArgumentType,
    BlockedRows,
    ColumnStart,
    FragmentFold,
    Lane,
    MetalType,
    OrderedReduction,
    RowIndices,
)
from ..core.plan import Source
from .affine import AFFINE, whole
from .tiles import row_tile

SELECTED = Source("contractions/selected.metal", (AFFINE,))
ROUTE_DRIVER = Source(
    "reductions/ordered.metal",
    (Source("contractions/route.metal", (Source("core/fragments.metal"),)),),
)


@kernel(source=SELECTED, function="magnitude_selected")
def selected_affine(
    x, ids, order, w, s, bias, sw, ss, sb, *, bits, group_size, slots, shared, per_slot=False
):
    count, n, k = ids.size, w.shape[-2], x.shape[-1]
    pack = 64 // bits
    if (
        k % (32 * pack)
        or group_size % pack
        or n % 8
        or count % slots
        or x.size // k != count // (1 if per_slot else slots)
        or order.size != count
        or w.shape[-1] * 32 != k * bits
        or s.value.tensor != bias.value.tensor
        or s.shape != (*w.shape[:-1], k // group_size)
    ):
        raise ValueError("selected affine operands disagree with the route geometry")
    banks = w.shape[0]
    reuse = count // slots > 1 and count >= 2 * (banks + shared)
    rows = row_tile(count, 4) if reuse else 1
    output = Tensor((count // slots, slots, n), x.dtype)
    return FragmentFold(
        output,
        BlockedRows(count, n, rows, order.value),
        dict(x=whole(x), ids=whole(ids), rows=RowIndices(), first=ColumnStart(), lane=Lane()),
        (x.dtype, bits, pack, k, rows, slots, banks, shared, per_slot, ArgumentType(whole(ids))),
        MetalType(
            "SelectedStep",
            (x.dtype, bits, pack, k, n, group_size, rows),
            tuple(whole(a) for a in (w, s, bias, sw, ss, sb)),
        ),
        pack,
    )


@kernel(source=ROUTE_DRIVER, function="magnitude_ordered_fragments")
def route_sum(projected, scores, shared_score, *, shared):
    rows, slots, width = projected.shape
    if scores.shape != (rows, slots - shared):
        raise ValueError("route scores disagree with projected slots")
    args = tuple(whole(a) for a in (scores, shared_score))
    return OrderedReduction(
        projected.value,
        Tensor((rows, width), projected.dtype),
        MetalType(
            "RouteStep", (projected.dtype, slots, shared, *(ArgumentType(a) for a in args)), args
        ),
    )
