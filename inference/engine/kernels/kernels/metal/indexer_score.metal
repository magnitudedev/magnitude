// indexer_score on Metal (contract and portable body in indexer.seismic).
//
// A threadgroup of 256 threads per (256 scores, row). A tile holding some
// of the row's listed rows first prepares the row's H query heads into
// threadgroup memory, a simdgroup per head (8 heads at a time): widened to
// F32, RMS-normalized with the query norm when NQ = 1, rotated at the row's
// coordinates and rounded with the indexer's trained rounding, as
// `indexer_keys` prepares a key. Then thread t scores ordinal j = 256 x + t
// of the row's list: the j-th key-plane row of its spans (each thread walks
// the spans to it), streaming the key's columns once in fours into one
// accumulator per head, so each head's dot is the ascending FMA chain of
// the portable body; s = scale * sum_h relu(dot_h) * w_h, heads ascending.
// Every score past the list is -inf.
//
// Dynamic extents and parameters read `seismic_words`.

#include "lib/core/activation.h"
#include "lib/core/rounding.h"
#include "lib/indexer/indexer.h"
#include <seismic/slab.h>

#define SCORE_W uint(2 * SEISMIC_DIM_P + SEISMIC_DIM_S)
#define SCORE_E (SCORE_W / 32)
#define SCORE_H uint(SEISMIC_DIM_H)

static_assert(SCORE_W % 32 == 0, "indexer head width must be a multiple of 32");
static_assert(SEISMIC_DIM_P % SCORE_E == 0, "each rotary half must cover whole lanes");

typedef element::Act::native Scalar;
typedef indexer::Lanes<SCORE_E, SEISMIC_DIM_P, (SEISMIC_DIM_RT != 0 ? SEISMIC_DIM_S : 0), (SEISMIC_DIM_RA != 0)>
    query_lanes;

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
    device float *scores [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup float *prepared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong row = group.y;
    const ulong n = ulong(SEISMIC_DIM_N);
    const uint r = uint(SEISMIC_DIM_R);
    device const int *row_spans = spans + row * r * 2;
    // The row's listed rows: those of its spans, at most N scored.
    ulong listed = 0;
    for (uint span = 0; span < r; ++span)
        listed += indexer::span_length(row_spans[span * 2], row_spans[span * 2 + 1]);
    listed = listed < n ? listed : n;
    const ulong ordinal = ulong(group.x) * 256 + thread_index;
    device float *out = scores + row * n;
    if (ulong(group.x) * 256 >= listed) {
        if (ordinal < n)
            out[ordinal] = -INFINITY;
        return;
    }
    // The row's query heads, prepared in F32.
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const rounding::Rounding trained = {int(uint(SEISMIC_PARAM_BLOCK)), int(uint(SEISMIC_PARAM_ELEMENT)),
        int(uint(SEISMIC_PARAM_ROUNDING_SCALE)), as_type<float>(uint(SEISMIC_PARAM_AMAX_FLOOR))};
    for (uint head = simd; head < SCORE_H; head += 8) {
        float x[SCORE_E];
        device const Scalar *raw = query + (row * SCORE_H + head) * SCORE_W;
        for (uint i = 0; i < SCORE_E; ++i)
            x[i] = float(raw[lane * SCORE_E + i]);
        // A rounded query is rounded from its norm and rotation held in A.
        if (SEISMIC_DIM_NQ != 0) {
            query_lanes::normalize(x, query_norm, epsilon, lane);
            rounding::hold_lanes<element::Act>(x, trained);
        }
        query_lanes::rotate(x, coordinates + row * 4, rotary_components, rotary_frequencies, rotary_amplitudes,
            lane);
        rounding::hold_lanes<element::Act>(x, trained);
        rounding::round_lanes<SCORE_E>(x, trained);
        for (uint i = 0; i < SCORE_E; ++i)
            prepared[head * SCORE_W + lane * SCORE_E + i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ordinal >= n)
        return;
    if (ordinal >= listed) {
        out[ordinal] = -INFINITY;
        return;
    }
    // The ordinal's key-plane row: its span's first row plus its place.
    int at = 0;
    ulong before = 0;
    for (uint span = 0; span < r; ++span) {
        const int lo = row_spans[span * 2];
        const ulong length = indexer::span_length(lo, row_spans[span * 2 + 1]);
        if (ordinal < before + length) {
            at = lo + int(ordinal - before);
            break;
        }
        before += length;
    }
    device const float *k = slab::row<float>(keys, ulong(at), ulong(SEISMIC_PARAM_SLAB_ROWS), SCORE_W);
    float dot[SCORE_H];
    for (uint head = 0; head < SCORE_H; ++head)
        dot[head] = 0.0f;
    for (uint column = 0; column < SCORE_W; column += 4) {
        const float4 kx = *reinterpret_cast<device const float4 *>(k + column);
        for (uint head = 0; head < SCORE_H; ++head) {
            const float4 q = *reinterpret_cast<threadgroup const float4 *>(prepared + head * SCORE_W + column);
            float d = dot[head];
            d = metal::fma(q.x, kx.x, d);
            d = metal::fma(q.y, kx.y, d);
            d = metal::fma(q.z, kx.z, d);
            d = metal::fma(q.w, kx.w, d);
            dot[head] = d;
        }
    }
    float score = 0.0f;
    for (uint head = 0; head < SCORE_H; ++head) {
        const float weight = SEISMIC_DIM_HW != 0 ? float(head_weights[row * SCORE_H + head]) : 1.0f;
        score = score + metal::max(dot[head], 0.0f) * weight;
    }
    out[ordinal] = as_type<float>(uint(SEISMIC_PARAM_SCALE)) * score;
}
