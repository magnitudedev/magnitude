// indexer_score on Metal (contract and portable body in indexer.seismic).
//
// A threadgroup of 256 threads per (256 scores, row). Each simdgroup first
// scans the row's spans 32 at a time (`indexer::simd_span_row`) to get the
// listed row count and each lane's key-plane row for ordinal j = 256 x + t.
// A tile holding some of the row's listed rows then prepares the row's H
// query heads into threadgroup memory (padded by 4 floats per head so
// cross-head lane reads are bank-conflict-free), a simdgroup per head (8
// heads at a time): widened to F32, RMS-normalized with the query norm when
// NQ = 1, rotated at the row's coordinates and rounded with the indexer's
// trained rounding, as `indexer_keys` prepares a key, while staging the row's
// head weights once per threadgroup. Then each simdgroup cooperatively scores
// its 32 ordinals one key at a time: all 32 lanes load k[at, 0..W) coalesced
// in float4s into the simdgroup's threadgroup key buffer, each lane owns
// ceil(H / 32) heads (computing each head's ascending FMA dot product against
// the broadcast threadgroup key), reduces relu(dot_h) * w_h across the
// simdgroup, and the owning lane keeps its score so all 32 lanes write their
// scores coalesced. Every score past the list is -inf.
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
#define SCORE_W4 (SCORE_W / 4)
#define SCORE_E (SCORE_W / 32)
#define SCORE_H uint(SEISMIC_DIM_H)
#define SCORE_Q_STRIDE (SCORE_W + 4)
#define SCORE_HEADS_PER_LANE ((SCORE_H + 31) / 32)

typedef activation::Act::native Scalar;
typedef rotary::layout<SCORE_W, SEISMIC_DIM_P, (SEISMIC_DIM_RT != 0 ? SEISMIC_DIM_S : 0), (SEISMIC_DIM_RA != 0)>
    query_layout;

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
    //   prepared[SCORE_H][SCORE_Q_STRIDE]  (padded by 4 floats, conflict-free)
    //   k_buf[8][SCORE_W]                  (per-simdgroup coalesced key buffer)
    //   weights[HW]                        (threadgroup-uniform head weights)
    threadgroup float *prepared = smem;
    threadgroup float *k_buf = prepared + SCORE_H * SCORE_Q_STRIDE;
    threadgroup float *weights = k_buf + 8 * SCORE_W;

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
    // The row's query heads, prepared in F32, and its head weights.
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const rounding::Rounding trained = rounding::of(uint(SEISMIC_PARAM_BLOCK), uint(SEISMIC_PARAM_ELEMENT),
        uint(SEISMIC_PARAM_ROUNDING_SCALE), uint(SEISMIC_PARAM_AMAX_FLOOR));
    // One table rotates every head of the row.
    const query_layout::table table(coordinates + row * 4, rotary_components, rotary_frequencies,
        rotary_amplitudes, lane);
    for (uint head = simd; head < SCORE_H; head += 8) {
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
        threadgroup float *prepared_head = prepared + head * SCORE_Q_STRIDE;
        for (uint i = 0; i < SCORE_E; ++i)
            prepared_head[lane * SCORE_E + i] = x[i];
    }
    if (SEISMIC_DIM_HW != 0 && thread_index < SCORE_H)
        weights[thread_index] = float(head_weights[row * SCORE_H + thread_index]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load each lane's head weight(s) once into registers.
    float w[SCORE_HEADS_PER_LANE];
    for (uint p = 0; p < SCORE_HEADS_PER_LANE; ++p) {
        const uint head = p * 32 + lane;
        w[p] = SEISMIC_DIM_HW != 0 ? (head < SCORE_H ? weights[head] : 0.0f) : 1.0f;
    }

    const ulong simd_base = ulong(group.x) * 256 + ulong(simd) * 32;
    const uint simd_keys = simd_base < listed ? uint(listed - simd_base < 32 ? listed - simd_base : 32) : 0u;
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    threadgroup float *k_simd = k_buf + simd * SCORE_W;
    float my_score = -INFINITY;

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

    if (ordinal < n)
        out.put(ordinal, my_score);
}
