// indexer_score on Metal (contract and portable body in indexer.seismic).
//
// A threadgroup of 256 threads per (256 scores, row). Each simdgroup first
// scans the row's spans 32 at a time (`indexer::simd_span_row`) to get the
// listed row count and each lane's key-plane row for ordinal j = 256 x + t.
// A tile holding some of the row's listed rows then prepares the row's H
// query heads into threadgroup memory, a simdgroup per head (8 heads at a
// time): widened to F32, RMS-normalized with the query norm when NQ = 1,
// rotated at the row's coordinates and rounded with the indexer's trained
// rounding, as `indexer_keys` prepares a key.
//
// When all H heads (padded by 4 floats per head for conflict-free cross-head
// reads) plus the 8 per-simdgroup key buffers fit in 32 KiB, all H heads are
// prepared once and each simdgroup scores its 32 ordinals one key at a time
// with each lane owning ceil(H / 32) heads (ascending FMA dot products).
// Otherwise (e.g. W = 512, H = 32), the threadgroup tiles over power-of-two
// chunks of heads (`SCORE_TILED_HEAD_TILE`) that fit alongside the 8 key
// buffers, splitting each head's dot product across
// `32 / SCORE_TILED_HEAD_TILE` lanes so all 32 lanes stay active and
// accumulating across head tiles. Every score past the list is -inf.
//
// Dynamic extents and parameters read `seismic_words`.

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"
#include "lib/core/head.h"
#include "lib/core/rotary.h"
#include "lib/core/rounding.h"
#include "lib/indexer/indexer.h"
#include <seismic/slab.h>

#define SCORE_W uint(2 * SEISMIC_DIM_P + SEISMIC_DIM_S)
#define SCORE_W4 (SCORE_W / 4u)
#define SCORE_E (SCORE_W / 32u)
#define SCORE_H uint(SEISMIC_DIM_H)
#define SCORE_SMEM_FLOATS 8192u
#define SCORE_SINGLE_TILE ((SCORE_H * (SCORE_W + 4u) + 8u * SCORE_W) <= SCORE_SMEM_FLOATS)
#define SCORE_AVAIL_HEADS ((SCORE_SMEM_FLOATS - 8u * SCORE_W) / SCORE_W)
#define SCORE_TILED_HEAD_TILE \
    (SCORE_AVAIL_HEADS >= 32u ? 32u : \
     SCORE_AVAIL_HEADS >= 16u ? 16u : \
     SCORE_AVAIL_HEADS >= 8u  ? 8u  : \
     SCORE_AVAIL_HEADS >= 4u  ? 4u  : \
     SCORE_AVAIL_HEADS >= 2u  ? 2u  : 1u)
#define SCORE_HEAD_TILE (SCORE_SINGLE_TILE ? SCORE_H : SCORE_TILED_HEAD_TILE)
#define SCORE_Q_STRIDE (SCORE_SINGLE_TILE ? (SCORE_W + 4u) : SCORE_W)
#define SCORE_HEADS_PER_LANE ((SCORE_H + 31u) / 32u)
#define SCORE_LANES_PER_HEAD (32u / SCORE_TILED_HEAD_TILE)

typedef activation::Act::native Scalar;
typedef rotary::layout<SCORE_W, SEISMIC_DIM_P, (SEISMIC_DIM_RT != 0 ? SEISMIC_DIM_S : 0), (SEISMIC_DIM_RA != 0)>
    query_layout;

static inline void prepare_query_head(
    device const Scalar *query,
    device const float *query_norm,
    thread const query_layout::table &table,
    const rounding::Rounding trained,
    const float epsilon,
    const ulong row,
    const uint head,
    const uint lane,
    threadgroup float *prepared_head) {
    float x[SCORE_E];
    device const Scalar *raw = query + (row * SCORE_H + head) * SCORE_W;
    for (uint i = 0; i < SCORE_E; ++i)
        x[i] = float(raw[lane * SCORE_E + i]);
    // A rounded query is rounded from its norm and rotation held in A.
    if (SEISMIC_DIM_NQ != 0) {
        head::rms_norm(x, query_norm, epsilon, lane);
        rounding::hold_lanes<activation::Act>(x, trained);
    }
    query_layout::rotate(table, lane, x);
    rounding::hold_lanes<activation::Act>(x, trained);
    rounding::round_lanes<SCORE_E>(x, trained);
    for (uint i = 0; i < SCORE_E; ++i)
        prepared_head[lane * SCORE_E + i] = x[i];
}

kernel void indexer_score(
    device const Scalar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const float *query_norm [[buffer(SEISMIC_BUFFER_QUERY_NORM)]],
    device const Scalar *head_weights [[buffer(SEISMIC_BUFFER_HEAD_WEIGHTS)]],
    device const int *rotary_components [[buffer(SEISMIC_BUFFER_ROTARY_COMPONENTS)]],
    device const float *rotary_frequencies [[buffer(SEISMIC_BUFFER_ROTARY_FREQUENCIES)]],
    device const float *rotary_amplitudes [[buffer(SEISMIC_BUFFER_ROTARY_AMPLITUDES)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    device const ulong *keys [[buffer(SEISMIC_BUFFER_KEYS)]],
    device const int *spans [[buffer(SEISMIC_BUFFER_SPANS)]],
    SEISMIC_RESULT_0_PARAM(score_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup float *smem [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    // Threadgroup memory layout:
    //   prepared[SCORE_HEAD_TILE][SCORE_Q_STRIDE]  (padded by 4 in single-tile mode)
    //   k_buf[8][SCORE_W]                          (per-simdgroup coalesced key buffer)
    threadgroup float *prepared = smem;
    threadgroup float *k_buf = prepared + SCORE_HEAD_TILE * SCORE_Q_STRIDE;

    const ulong row = group.y;
    const ulong n = ulong(SEISMIC_DIM_N);
    const uint r = uint(SEISMIC_DIM_R);
    device const int *row_spans = spans + row * r * 2;
    const ulong ordinal = ulong(group.x) * 256 + thread_index;
    // The row's listed rows (those of its spans, at most N scored) and each
    // lane's key-plane row.
    int at = -1;
    ulong listed = indexer::simd_span_row(row_spans, r, uint(ordinal), lane, at);
    listed = listed < n ? listed : n;
    const element::Out<element::F32> out = element::Out<element::F32>(score_cells).offset(row * n);
    if (ulong(group.x) * 256 >= listed) {
        if (ordinal < n)
            out.put(ordinal, -INFINITY);
        return;
    }
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const rounding::Rounding trained = rounding::of(uint(SEISMIC_PARAM_BLOCK), uint(SEISMIC_PARAM_ELEMENT),
        uint(SEISMIC_PARAM_ROUNDING_SCALE), uint(SEISMIC_PARAM_AMAX_FLOOR));
    // One table rotates every head of the row.
    const query_layout::table table(coordinates + row * 4, rotary_components, rotary_frequencies,
        rotary_amplitudes, lane);

    const ulong simd_base = ulong(group.x) * 256 + ulong(simd) * 32;
    const uint simd_keys = simd_base < listed ? uint(listed - simd_base < 32 ? listed - simd_base : 32) : 0u;
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    threadgroup float *k_simd = k_buf + simd * SCORE_W;
    float my_score = -INFINITY;

    if (SCORE_SINGLE_TILE) {
        for (uint head = simd; head < SCORE_H; head += 8) {
            prepare_query_head(query, query_norm, table, trained, epsilon, row, head, lane,
                prepared + head * SCORE_Q_STRIDE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load each lane's head weight(s) once into registers.
        float w[SCORE_HEADS_PER_LANE];
        for (uint p = 0; p < SCORE_HEADS_PER_LANE; ++p) {
            const uint head = p * 32 + lane;
            w[p] = (SEISMIC_DIM_HW != 0 && head < SCORE_H) ? float(head_weights[row * SCORE_H + head]) : 1.0f;
        }

        for (uint j = 0; j < simd_keys; ++j) {
            const int key_row = simd_broadcast(at, ushort(j));
            device const float *k = slab::row<float>(keys, ulong(key_row), ulong(SEISMIC_PARAM_SLAB_ROWS), SCORE_W);
            for (uint c = lane; c < SCORE_W4; c += 32)
                reinterpret_cast<threadgroup float4 *>(k_simd)[c] = reinterpret_cast<device const float4 *>(k)[c];
            simdgroup_barrier(mem_flags::mem_threadgroup);

            float dot[SCORE_HEADS_PER_LANE];
            for (uint p = 0; p < SCORE_HEADS_PER_LANE; ++p)
                dot[p] = 0.0f;
            for (uint column = 0; column < SCORE_W; column += 4) {
                const float4 kx = *reinterpret_cast<threadgroup const float4 *>(k_simd + column);
                for (uint p = 0; p < SCORE_HEADS_PER_LANE; ++p) {
                    const uint head = p * 32 + lane;
                    if (head < SCORE_H) {
                        const float4 q = *reinterpret_cast<threadgroup const float4 *>(prepared + head * SCORE_Q_STRIDE + column);
                        float d = dot[p];
                        d = metal::fma(q.x, kx.x, d);
                        d = metal::fma(q.y, kx.y, d);
                        d = metal::fma(q.z, kx.z, d);
                        d = metal::fma(q.w, kx.w, d);
                        dot[p] = d;
                    }
                }
            }
            float val = 0.0f;
            for (uint p = 0; p < SCORE_HEADS_PER_LANE; ++p) {
                const uint head = p * 32 + lane;
                if (head < SCORE_H)
                    val += metal::max(dot[p], 0.0f) * w[p];
            }
            const float total = simd_sum(val);
            if (lane == j)
                my_score = scale * total;
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        const uint head_in_tile = lane / SCORE_LANES_PER_HEAD;
        const uint sub_lane = lane % SCORE_LANES_PER_HEAD;
        const uint step = SCORE_LANES_PER_HEAD * 4u;
        const uint phase = SCORE_LANES_PER_HEAD < 8u ? ((head_in_tile * step) & 31u) : 0u;
        if (lane < simd_keys)
            my_score = 0.0f;

        for (uint h_base = 0; h_base < SCORE_H; h_base += SCORE_TILED_HEAD_TILE) {
            const uint tile_h = (SCORE_H - h_base) < SCORE_TILED_HEAD_TILE ? (SCORE_H - h_base) : SCORE_TILED_HEAD_TILE;
            for (uint h = simd; h < tile_h; h += 8) {
                prepare_query_head(query, query_norm, table, trained, epsilon, row, h_base + h, lane,
                    prepared + h * SCORE_Q_STRIDE);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint head = h_base + head_in_tile;
            const bool active_head = head < SCORE_H;
            const float w = (SEISMIC_DIM_HW != 0 && active_head) ? float(head_weights[row * SCORE_H + head]) : 1.0f;

            for (uint j = 0; j < simd_keys; ++j) {
                const int key_row = simd_broadcast(at, ushort(j));
                device const float *k = slab::row<float>(keys, ulong(key_row), ulong(SEISMIC_PARAM_SLAB_ROWS), SCORE_W);
                for (uint c = lane; c < SCORE_W4; c += 32)
                    reinterpret_cast<threadgroup float4 *>(k_simd)[c] = reinterpret_cast<device const float4 *>(k)[c];
                simdgroup_barrier(mem_flags::mem_threadgroup);

                float dot = 0.0f;
                if (active_head) {
                    threadgroup const float *q_head = prepared + head_in_tile * SCORE_Q_STRIDE;
                    for (uint base = 0; base < SCORE_W; base += step) {
                        const uint col = (base ^ phase) + sub_lane * 4u;
                        if (col < SCORE_W) {
                            const float4 kx = *reinterpret_cast<threadgroup const float4 *>(k_simd + col);
                            const float4 q = *reinterpret_cast<threadgroup const float4 *>(q_head + col);
                            dot = metal::fma(q.x, kx.x, dot);
                            dot = metal::fma(q.y, kx.y, dot);
                            dot = metal::fma(q.z, kx.z, dot);
                            dot = metal::fma(q.w, kx.w, dot);
                        }
                    }
                }
                for (uint offset = SCORE_LANES_PER_HEAD / 2u; offset > 0u; offset >>= 1u)
                    dot += simd_shuffle_xor(dot, ushort(offset));
                const float val = (sub_lane == 0u && active_head) ? metal::max(dot, 0.0f) * w : 0.0f;
                const float tile_total = simd_sum(val);
                if (lane == j)
                    my_score += tile_total;
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (h_base + SCORE_TILED_HEAD_TILE < SCORE_H)
                threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane < simd_keys)
            my_score *= scale;
    }

    if (ordinal < n)
        out.put(ordinal, my_score);
}
