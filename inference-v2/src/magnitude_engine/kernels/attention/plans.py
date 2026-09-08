"""Bounded split-context attention with cooperative head/query and key tiles."""

import mlx.core as mx

from ..core.graph import Tensor
from ..core.kernel import dispatch
from ..core.plan import Launch, Source

TILED_PARTIALS = Source("attention/tiled_partials.metal")
TILED_COMBINE = Source("attention/tiled_combine.metal")


def tiled_attention(
    q, k, v, pages, positions, *, page_size, table_width, covered, scale, window, tail
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
    partial, maximum, denominator = dispatch(
        TILED_PARTIALS,
        inputs=(
            ("q", q),
            ("k", k),
            ("v", v),
            ("pages", pages),
            ("positions", positions),
            *(
                (n, v)
                for n, v in zip(("tk", "tv", "starts"), tail or (k, v, positions), strict=True)
            ),
            ("scale", mx.array([scale], mx.float32)),
        ),
        outputs=(
            ("partial", Tensor((rows, blocks, dv), q.dtype if single_pass else mx.float32)),
            ("maximum", Tensor((rows, blocks), mx.float32)),
            ("denominator", Tensor((rows, blocks), mx.float32)),
        ),
        launch=Launch(
            (
                32 * (head_tile // heads) * subchunks,
                blocks,
                batch * hk * query_groups * (group // head_tile),
            ),
            (32 * (head_tile // heads) * subchunks, 1, 1),
        ),
        template=(
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
            ("TAIL", tail[0].shape[2] if tail else 0),
        ),
    )
    if single_pass:
        return partial.reshape(batch, hq, count, dv)
    groups = min(8, blocks)
    return dispatch(
        TILED_COMBINE,
        inputs=(("partial", partial), ("maximum", maximum), ("denominator", denominator)),
        outputs=(("output", Tensor((batch, hq, count, dv), q.dtype)),),
        launch=Launch((32 * groups, rows, 1), (32 * groups, 1, 1)),
        template=(("In", q.dtype), ("DV", dv), ("BLOCKS", blocks), ("GROUPS", groups)),
    )[0]


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
    count = queries.shape[2]
    span = partition_tokens
    splits = (covered + span - 1) // span
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
    layout = mx.array([keys.shape[1], page_size, splits, table_width, window or 0], mx.int32)
    partial = dispatch(
        PARTITIONED_PARTIALS,
        inputs=(
            ("queries", queries),
            ("keys", keys),
            ("values", values),
            ("pages", pages),
            ("positions", positions),
            ("layout", layout),
            ("scale", mx.array([scale], mx.float32)),
            *(
                (n, v)
                for n, v in zip(
                    ("tail_keys", "tail_values", "tail_starts"),
                    tail if tail is not None else (keys, values, positions),
                    strict=True,
                )
            ),
        ),
        outputs=(("partial", Tensor((rows, splits, dv + 2), mx.float32)),),
        launch=Launch((32, splits, rows // heads), threadgroup),
        template=(
            ("DK", dk),
            ("DV", dv),
            ("HQ", queries.shape[1]),
            ("HK", keys.shape[0]),
            ("TQ", count),
            ("SPAN", span),
            ("HEADS", heads),
            ("TAIL", tail[0].shape[2] if tail is not None else 0),
        ),
    )[0]
    return dispatch(
        PARTITIONED_COMBINE,
        inputs=(("partial", partial), ("layout", layout)),
        outputs=(
            ("output", Tensor((queries.shape[0], queries.shape[1], count, dv), queries.dtype)),
        ),
        launch=Launch((32, rows, 1), (32, 1, 1)),
        template=(("Out", queries.dtype), ("DV", dv)),
    )[0]
