// select_compact_rows: fused single-launch shortlist selection + softmax
// confidence directly over `[M, S]` compact scores and canonical `token_rows`.
// One 64-thread (2-simdgroup) threadgroup per row.
#include <seismic/element.h>
#include "lib/core/gumbel.h"
#include "lib/core/reduce.h"

constant constexpr uint compact_no_token = 0xffffffffu;

inline bool compact_better(float score, uint token, float best_score, uint best_token) {
    return token != compact_no_token &&
        (best_token == compact_no_token || score > best_score || (score == best_score && token < best_token));
}

kernel void select_compact_rows(
    device const float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device const int *token_rows [[buffer(SEISMIC_BUFFER_TOKEN_ROWS)]],
    device const uint *mask [[buffer(SEISMIC_BUFFER_MASK)]],
    device const int *constrained [[buffer(SEISMIC_BUFFER_CONSTRAINED)]],
    device const uint *draws [[buffer(SEISMIC_BUFFER_DRAWS)]],
    device int *result [[buffer(SEISMIC_BUFFER_RESULT)]],
    SEISMIC_OUT_CONFIDENCE_PARAM(confidence_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sg_scores[2];
    threadgroup float sg_logits[2];
    threadgroup uint sg_tokens[2];
    threadgroup uint sg_bads[2];
    threadgroup reduce::OnlineSoftmax sg_sm[2];

    const ulong W = SEISMIC_DIM_W;
    const ulong S = SEISMIC_DIM_S;
    device const float *score_row = scores + ulong(row) * SEISMIC_SCORES_STRIDE_0;
    device const int *tok_row = token_rows + ulong(row) * SEISMIC_TOKEN_ROWS_STRIDE_0;
    device const uint *words = mask + ulong(row) * SEISMIC_MASK_STRIDE_0;
    device const uint *draw = draws + ulong(row) * SEISMIC_DRAWS_STRIDE_0;
    const bool masked = constrained[ulong(row) * SEISMIC_CONSTRAINED_STRIDE_0] != 0;
    const bool noisy = draw[0] == 1u;

    float best_score = -INFINITY;
    float best_logit = -INFINITY;
    uint best_token = compact_no_token;
    uint bad = 0u;
    reduce::OnlineSoftmax sm = reduce::OnlineSoftmax::empty();

    for (ulong col = ulong(tid); col < S; col += 64ul) {
        const int raw_token = tok_row[col * SEISMIC_TOKEN_ROWS_STRIDE_1];
        if (raw_token < 0 || (ulong(raw_token) >> 5ul) >= W)
            continue;
        const uint token = uint(raw_token);
        const float value = score_row[col * SEISMIC_SCORES_STRIDE_1];
        bad |= uint(metal::isnan(value) || value == INFINITY);
        sm.add(value);
        if (!(value > -INFINITY && value < INFINITY))
            continue;
        if (masked && ((words[ulong(token >> 5u) * SEISMIC_MASK_STRIDE_1] >> (token & 31u)) & 1u) == 0u)
            continue;
        const float candidate = noisy ? (value + gumbel::noise(token, draw, SEISMIC_DRAWS_STRIDE_1)) : value;
        if (compact_better(candidate, token, best_score, best_token)) {
            best_score = candidate;
            best_logit = value;
            best_token = token;
        }
    }

    bad = simd_max(bad);
    const float sg_max_score = simd_max(best_token == compact_no_token ? -INFINITY : best_score);
    const bool sg_any = simd_any(best_token != compact_no_token);
    const uint winner = simd_min((best_token != compact_no_token && best_score == sg_max_score) ? best_token : compact_no_token);
    const float winner_logit = simd_max((best_token != compact_no_token && best_token == winner) ? best_logit : -INFINITY);
    sm = reduce::warp_online_softmax(sm);

    if (lane == 0u) {
        sg_scores[sg] = sg_max_score;
        sg_logits[sg] = winner_logit;
        sg_tokens[sg] = sg_any ? winner : compact_no_token;
        sg_bads[sg] = bad;
        sg_sm[sg] = sm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float final_score = sg_scores[0];
        float final_logit = sg_logits[0];
        uint final_token = sg_tokens[0];
        if (compact_better(sg_scores[1], sg_tokens[1], final_score, final_token)) {
            final_score = sg_scores[1];
            final_logit = sg_logits[1];
            final_token = sg_tokens[1];
        }
        const uint final_bad = sg_bads[0] | sg_bads[1];
        const reduce::OnlineSoftmax final_sm = reduce::OnlineSoftmax::merge(sg_sm[0], sg_sm[1]);
        const int status = final_bad != 0u ? 2 : (final_token == compact_no_token ? 1 : 0);
        device int *out = result + ulong(row) * SEISMIC_RESULT_STRIDE_0;
        out[0] = status == 0 ? int(final_token) : -1;
        out[SEISMIC_RESULT_STRIDE_1] = status;
        const element::Out<element::F32> confidence(confidence_cells);
        confidence.put(
            ulong(row) * SEISMIC_CONFIDENCE_STRIDE_0,
            status == 0 && final_sm.sum != 0.0f ? metal::exp(final_logit - final_sm.max) / final_sm.sum : 0.0f);
    }
}
