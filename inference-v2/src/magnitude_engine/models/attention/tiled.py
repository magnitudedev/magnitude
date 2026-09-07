"""Bounded split-context attention with cooperative head/query and key tiles."""

from functools import cache
from typing import Any

import mlx.core as mx


@cache
def _partials() -> Any:
    return mx.fast.metal_kernel(
        name="magnitude_attention_tiles",
        header='#define MAG_UNROLL _Pragma("clang loop unroll(full)")\n',
        input_names=["q", "k", "v", "pages", "positions", "tk", "tv", "starts", "scale"],
        output_names=["partial", "maximum", "denominator"],
        source="""
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint ci = sg % NC, hi = sg / NC;
        uint block = threadgroup_position_in_grid.y;
        uint item = threadgroup_position_in_grid.z;
        uint qt = item % QGROUPS, ht = (item / QGROUPS) % HEAD_GROUPS;
        uint kh = (item / (QGROUPS * HEAD_GROUPS)) % HK;
        uint row = item / (QGROUPS * HEAD_GROUPS * HK);
        uint local_head = hi * HP, head = ht * HG + local_head;
        constexpr int PK = DK / 32, PV = DV / 32;
        uint pos = positions[row];
        uint origin = WINDOW > 0 && pos + 1 > WINDOW ? pos + 1 - WINDOW : 0;
        // Partition logical visibility, independent of page-table capacity padding.
        uint visible = pos + TQ - origin;
        uint subspan = (visible + BLOCKS * NC - 1) / (BLOCKS * NC);
        uint span = subspan * NC;
        uint begin = origin + block * span + ci * subspan;
        uint end = min(origin + (block + 1) * span, begin + subspan);
        end = min(end, pos + min(uint(TQ), (qt + 1) * QT));
        // Fixed tile loops must unroll: dynamically indexed arrays spill registers.
        float query[HP][QT][PK], output[HP][QT][PV];
        float maxima[HP][QT], sums[HP][QT];
        MAG_UNROLL for (uint h = 0; h < HP; ++h)
        MAG_UNROLL for (uint t = 0; t < QT; ++t) {
            uint token = qt * QT + t;
            MAG_UNROLL for (uint i = 0; i < PK; ++i)
                query[h][t][i] = token < TQ
                    ? float(q[((row * HQ + kh * G + head + h) * TQ + token) * DK + lane * PK + i])
                    : 0.0f;
            MAG_UNROLL for (uint i = 0; i < PV; ++i) output[h][t][i] = 0.0f;
            maxima[h][t] = -INFINITY; sums[h][t] = 0.0f;
        }
        uint p = begin;
        while (p < end) {
            bool tail = TAIL > 0 && p >= uint(starts[row]);
            uint stop = tail ? end : min(end, (p / PAGE + 1) * PAGE);
            if (TAIL > 0 && !tail) stop = min(stop, uint(starts[row]));
            size_t address = tail ? (size_t(row) * HK + kh) * TAIL + p - starts[row]
                : size_t(kh) * CAPACITY + uint(pages[row * TABLE + p / PAGE]) * PAGE + p % PAGE;
            auto kp = tail ? tk : k;
            auto vp = tail ? tv : v;
            for (; p < stop; ++p, ++address) {
                float key[PK], value[PV];
                if (PK % 4 == 0) {
                    auto src =
                        reinterpret_cast<const device vec<In, 4>*>(kp + address * DK + lane * PK);
                    MAG_UNROLL for (uint i = 0; i < PK / 4; ++i) {
                        float4 x = float4(src[i]);
                        MAG_UNROLL for (uint j = 0; j < 4; ++j) key[i * 4 + j] = x[j];
                    }
                } else for (uint i =
                    0; i < PK; ++i) key[i] = float(kp[address * DK + lane * PK + i]);
                if (PV % 4 == 0) {
                    auto src =
                        reinterpret_cast<const device vec<In, 4>*>(vp + address * DV + lane * PV);
                    MAG_UNROLL for (uint i = 0; i < PV / 4; ++i) {
                        float4 x = float4(src[i]);
                        MAG_UNROLL for (uint j = 0; j < 4; ++j) value[i * 4 + j] = x[j];
                    }
                } else for (uint i =
                    0; i < PV; ++i) value[i] = float(vp[address * DV + lane * PV + i]);
                MAG_UNROLL for (uint h = 0; h < HP; ++h)
                MAG_UNROLL for (uint t = 0; t < QT; ++t) {
                    uint token = qt * QT + t;
                    if (token >= TQ || p > pos + token || (WINDOW > 0 && p + WINDOW <= pos + token))
                        continue;
                    float score = 0.0f;
                    MAG_UNROLL for (uint i = 0; i < PK; ++i) score += query[h][t][i] * key[i];
                    score = simd_sum(score) * scale[0];
                    float peak = max(maxima[h][t], score);
                    float a = exp(maxima[h][t] - peak), b = exp(score - peak);
                    sums[h][t] = sums[h][t] * a + b;
                    for (uint i =
                        0; i < PV; ++i) output[h][t][i] = output[h][t][i] * a + b * value[i];
                    maxima[h][t] = peak;
                }
            }
        }
        threadgroup float scratch[NC > 1 ? HG * NC * (DV + 2) : 1];
        MAG_UNROLL for (uint t = 0; t < QT; ++t) {
            if (NC > 1) {
                MAG_UNROLL for (uint h = 0; h < HP; ++h) {
                    uint at = ((local_head + h) * NC + ci) * (DV + 2);
                    MAG_UNROLL for (uint i = 0; i < PV; ++i)
                        scratch[at + lane * PV + i] = output[h][t][i];
                    if (lane == 0) { scratch[at + DV] =
                        maxima[h][t]; scratch[at + DV + 1] = sums[h][t]; }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (ci == 0 && qt * QT + t < TQ) MAG_UNROLL for (uint h = 0; h < HP; ++h) {
                float peak = maxima[h][t], denom = sums[h][t];
                float result[PV];
                MAG_UNROLL for (uint i = 0; i < PV; ++i) result[i] = output[h][t][i];
                if (NC > 1) {
                    uint at = (local_head + h) * NC * (DV + 2);
                    peak = -INFINITY;
                    for (uint c = 0; c < NC; ++c) peak = max(peak, scratch[at + c * (DV + 2) + DV]);
                    denom = 0.0f;
                    MAG_UNROLL for (uint i = 0; i < PV; ++i) result[i] = 0.0f;
                    for (uint c = 0; c < NC; ++c) {
                        uint src = at + c * (DV + 2);
                        float factor =
                            scratch[src + DV + 1] > 0 ? exp(scratch[src + DV] - peak) : 0.0f;
                        denom += factor * scratch[src + DV + 1];
                        for (uint i =
                            0; i < PV; ++i) result[i] += factor * scratch[src + lane * PV + i];
                    }
                }
                size_t out =
                    ((size_t(row) * HQ + kh * G + head + h) * TQ + qt * QT + t) * BLOCKS + block;
                MAG_UNROLL for (uint i = 0; i < PV; ++i)
                    partial[out * DV + lane * PV + i] =
                        Out(BLOCKS == 1 ? result[i] / denom : result[i]);
                if (lane == 0) { maximum[out] = peak; denominator[out] = denom; }
            }
            if (NC > 1) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        """,
    )


@cache
def _combine() -> Any:
    return mx.fast.metal_kernel(
        name="magnitude_attention_tile_sum",
        input_names=["partial", "maximum", "denominator"],
        output_names=["output"],
        source="""
        uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
        uint row = threadgroup_position_in_grid.y;
        size_t base = size_t(row) * BLOCKS;
        float peak = -INFINITY;
        for (uint b = lane; b < BLOCKS; b += 32) peak = max(peak, maximum[base + b]);
        peak = simd_max(peak);
        float norm = 0.0f;
        for (uint b = lane; b < BLOCKS; b += 32)
            if (denominator[base + b] > 0)
                norm += exp(maximum[base + b] - peak) * denominator[base + b];
        norm = simd_sum(norm);
        float values[DV / 32];
        for (uint i = 0; i < DV / 32; ++i) values[i] = 0.0f;
        for (uint b = sg; b < BLOCKS; b += GROUPS) {
            float factor = denominator[base + b] > 0 ? exp(maximum[base + b] - peak) : 0.0f;
            for (uint i = 0; i < DV / 32; ++i)
                values[i] += factor * partial[(base + b) * DV + lane + 32 * i];
        }
        threadgroup float scratch[GROUPS * DV];
        for (uint i = 0; i < DV / 32; ++i) scratch[sg * DV + lane + 32 * i] = values[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) for (uint i = 0; i < DV / 32; ++i) {
            float value = 0.0f;
            for (uint g = 0; g < GROUPS; ++g) value += scratch[g * DV + lane + 32 * i];
            output[row * DV + lane + 32 * i] = In(value / norm);
        }
        """,
    )


def apply(q, k, v, pages, positions, *, page_size, table_width, covered, scale, window, tail):
    batch, hq, count, dk = q.shape
    hk, capacity, dv = v.shape
    group = hq // hk
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
            512 // (batch * hk * blocks * (group // heads) * query_groups),
        ),
    )
    rows = batch * hq * count
    partial, maximum, denominator = _partials()(
        inputs=[
            q,
            k,
            v,
            pages,
            positions,
            *(tail or (k, v, positions)),
            mx.array([scale], mx.float32),
        ],
        template=[
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
            ("QGROUPS", query_groups),
            ("HP", heads),
            ("NC", subchunks),
            ("BLOCKS", blocks),
            ("PAGE", page_size),
            ("TABLE", table_width),
            ("CAPACITY", capacity),
            ("WINDOW", window or 0),
            ("TAIL", tail[0].shape[2] if tail else 0),
        ],
        grid=(
            32 * (head_tile // heads) * subchunks,
            blocks,
            batch * hk * query_groups * (group // head_tile),
        ),
        threadgroup=(32 * (head_tile // heads) * subchunks, 1, 1),
        output_shapes=[(rows, blocks, dv), (rows, blocks), (rows, blocks)],
        output_dtypes=[q.dtype if single_pass else mx.float32, mx.float32, mx.float32],
    )
    if single_pass:
        return partial.reshape(batch, hq, count, dv)
    groups = min(8, blocks)
    return _combine()(
        inputs=[partial, maximum, denominator],
        template=[("In", q.dtype), ("DV", dv), ("BLOCKS", blocks), ("GROUPS", groups)],
        grid=(32 * groups, rows, 1),
        threadgroup=(32 * groups, 1, 1),
        output_shapes=[(batch, hq, count, dv)],
        output_dtypes=[q.dtype],
    )[0]
