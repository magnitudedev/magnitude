"""Bounded split-context attention with cooperative head/query and key tiles."""

import mlx.core as mx

from .. import kernel
from ..core.graph import Tensor
from ..core.metal import Dispatch
from ..core.plan import Launch, Source

TILED_PARTIALS = Source("attention/tiled_partials.metal")
TILED_COMBINE = Source("attention/tiled_combine.metal")


def tiled_attention(
    q, k, v, pages, positions, *, page_size, table_width, covered, scale, window, tail
):
    partial, maximum, denominator = tiled_partials(
        q,
        k,
        v,
        pages,
        positions,
        tail or (k, v, positions),
        mx.array([scale], mx.float32),
        page_size=page_size,
        table_width=table_width,
        covered=covered,
        window=window,
        tail_tokens=tail[0].shape[2] if tail else 0,
    )
    shape = (*q.shape[:-1], v.shape[-1])
    if covered <= 512:
        return partial.reshape(shape)
    return tiled_combine(partial, maximum, denominator, shape=shape, dtype=q.dtype)


PARTITIONED_PARTIALS = Source("attention/partitioned_partials.metal")
PARTITIONED_COMBINE = Source("attention/partitioned_combine.metal")


def attend(
    queries,
    keys,
    values,
    pages,
    positions,
    *,
    page_size,
    table_width,
    covered,
    scale,
    window=None,
    tail=None,
    partition_tokens=128,
    heads_per_group=2,
):
    if (covered <= 512 or covered > 64 * partition_tokens) and queries.shape[1] // keys.shape[
        0
    ] <= 32:
        return tiled_attention(
            queries,
            keys,
            values,
            pages,
            positions,
            page_size=page_size,
            table_width=table_width,
            covered=covered,
            scale=scale,
            window=window,
            tail=tail,
        )
    splits = (covered + partition_tokens - 1) // partition_tokens
    layout = mx.array([keys.shape[1], page_size, splits, table_width, window or 0], mx.int32)
    partial = partitioned_partials(
        queries,
        keys,
        values,
        pages,
        positions,
        layout,
        mx.array([scale], mx.float32),
        tail if tail is not None else (keys, values, positions),
        span=partition_tokens,
        splits=splits,
        heads_per_group=heads_per_group,
        tail_tokens=tail[0].shape[2] if tail else 0,
    )
    return partitioned_combine(
        partial, layout, shape=(*queries.shape[:-1], values.shape[-1]), dtype=queries.dtype
    )


@kernel(source=TILED_PARTIALS)
def tiled_partials(
    q,
    k,
    v,
    pages,
    positions,
    tail_inputs,
    scale,
    *,
    page_size,
    table_width,
    covered,
    window,
    tail_tokens,
):
    batch, hq, count, dk = q.shape
    hk, capacity, dv = v.shape
    group = hq // hk
    key_tile = 1 if count > 1 else max(1, min(4, 1024 // (dk + dv)))
    query_tile = min(count, max(1, 32 // (max(dk, dv) // 32)))
    single_pass = covered <= 512
    head_tile = 1 if single_pass else group
    heads = min(head_tile, max(1, 32 // (query_tile * max(dk, dv) // 32)))
    while group % heads:
        heads -= 1
    blocks = 1 if single_pass else min(64, (covered + 127) // 128)
    query_groups = (count + query_tile - 1) // query_tile
    subchunks = max(
        1,
        min(
            32 if single_pass else 8,
            24576 // (head_tile * (dv + 2) * 4),
            32 // (head_tile // heads),
            512 // (hk * blocks * (group // heads) * query_groups),
        ),
    )
    rows = batch * hq * count
    return Dispatch(
        dict(
            (
                ("q", q),
                ("k", k),
                ("v", v),
                ("pages", pages),
                ("positions", positions),
                *zip(("tk", "tv", "starts"), tail_inputs, strict=True),
                ("scale", scale),
            )
        ),
        {
            "partial": Tensor((rows, blocks, dv), q.dtype if single_pass else mx.float32),
            "maximum": Tensor((rows, blocks), mx.float32),
            "denominator": Tensor((rows, blocks), mx.float32),
        },
        Launch(
            (
                32 * (head_tile // heads) * subchunks,
                blocks,
                batch * hk * query_groups * (group // head_tile),
            ),
            (32 * (head_tile // heads) * subchunks, 1, 1),
        ),
        (
            ("In", q.dtype),
            ("Out", q.dtype if single_pass else mx.float32),
            ("HQ", hq),
            ("HK", hk),
            ("G", group),
            ("HG", head_tile),
            ("HEAD_GROUPS", group // head_tile),
            ("DK", dk),
            ("DV", dv),
            ("TQ", count),
            ("QT", query_tile),
            ("KT", key_tile),
            ("QGROUPS", query_groups),
            ("HP", heads),
            ("NC", subchunks),
            ("BLOCKS", blocks),
            ("PAGE", page_size),
            ("TABLE", table_width),
            ("CAPACITY", capacity),
            ("WINDOW", window or 0),
            ("TAIL", tail_tokens),
        ),
    )


@kernel(source=TILED_COMBINE)
def tiled_combine(partial, maximum, denominator, *, shape, dtype):
    rows, blocks, dv = partial.shape
    groups = min(8, blocks)
    return Dispatch(
        {"partial": partial, "maximum": maximum, "denominator": denominator},
        {"output": Tensor(shape, dtype)},
        Launch((32 * groups, rows, 1), (32 * groups, 1, 1)),
        (("In", dtype), ("DV", dv), ("BLOCKS", blocks), ("GROUPS", groups)),
    )


@kernel(source=PARTITIONED_PARTIALS)
def partitioned_partials(
    queries,
    keys,
    values,
    pages,
    positions,
    layout,
    scale,
    tail_inputs,
    *,
    span,
    splits,
    heads_per_group,
    tail_tokens,
):
    count = queries.shape[2]
    rows = queries.shape[0] * queries.shape[1] * count
    dk, dv = (keys.shape[-1], values.shape[-1])
    group = queries.shape[1] // keys.shape[0]
    heads = min(heads_per_group, group)
    while group % heads:
        heads -= 1
    threadgroup = (32, 4, 1)
    cells = group // heads * count
    if cells > 1:
        sharing = min(4, cells)
        while cells % sharing:
            sharing -= 1
        threadgroup = (32, 1, sharing)
    return Dispatch(
        dict(
            (
                ("queries", queries),
                ("keys", keys),
                ("values", values),
                ("pages", pages),
                ("positions", positions),
                ("layout", layout),
                ("scale", scale),
                *zip(("tail_keys", "tail_values", "tail_starts"), tail_inputs, strict=True),
            )
        ),
        {"partial": Tensor((rows, splits, dv + 2), mx.float32)},
        Launch((32, splits, rows // heads), threadgroup),
        (
            ("DK", dk),
            ("DV", dv),
            ("HQ", queries.shape[1]),
            ("HK", keys.shape[0]),
            ("TQ", count),
            ("SPAN", span),
            ("HEADS", heads),
            ("TAIL", tail_tokens),
        ),
    )


@kernel(source=PARTITIONED_COMBINE)
def partitioned_combine(partial, layout, *, shape, dtype):
    rows = partial.shape[0]
    dv = shape[-1]
    return Dispatch(
        {"partial": partial, "layout": layout},
        {"output": Tensor(shape, dtype)},
        Launch((32, rows, 1), (32, 1, 1)),
        (("Out", dtype), ("DV", dv)),
    )
