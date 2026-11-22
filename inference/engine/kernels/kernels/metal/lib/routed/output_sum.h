// The join of Qwen's fused routed feed-forward's decode output, shared by
// `routed_output` (over its threadgroup's projections) and
// `routed_output_sited` (over the published ones, with the CPU's answer):
// the thread of channel n of row m publishes
//     residual + selected + round_A(shared) * coefficient,
// where `selected` accumulates score * round_A(projection) in slot order,
// then adds the CPU's answer where it answered (`lib/routed/sited.h`; none
// in `routed_output`); slot K of the projections is the shared expert's.

// The includer includes `routed.h` first.

namespace routed {

#ifndef ROUTED_ANSWER
#define ROUTED_ANSWER
// The CPU's answer at a channel of a row (`site_output`): its weighted sum
// of the CPU's choices, added after the slot-order sum where `answered`
// (`decode_selected`); none in the resident combines.
struct Answer {
    bool answered;
    float value;
};
#endif

#ifndef ROUTED_ANSWER4
#define ROUTED_ANSWER4
struct Answer4 {
    bool answered;
    float4 value;
};
#endif


// Channel n's publication, over the K + 1 A-rounded projections
// `projections.at(k)` and the CPU's `answer`.
template <typename Projections>
inline void output_publish(device const float *residual, device const float *inject, device const float *scores,
    device const float *coefficient, element::Out<element::F32> value, ulong m, uint n,
    thread const Projections &projections, thread const Answer &answer, constant ulong *seismic_words) {
    constexpr uint K = uint(SEISMIC_DIM_K);
    float selected = 0.0f;
    for (uint k = 0; k < K; ++k)
        selected = metal::fma(scores[m * SEISMIC_SCORES_STRIDE_0 + k * SEISMIC_SCORES_STRIDE_1], projections.at(k),
            selected);
    if (answer.answered)
        selected += answer.value;
    projection::routed_join<SEISMIC_DIM_SN, (SEISMIC_DIM_IN != 0)>(value.offset(m * SEISMIC_RESULT_0_STRIDE_0),
        SEISMIC_RESULT_0_STRIDE_1, residual + m * SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1,
        inject + m * SEISMIC_INJECT_STRIDE_0, SEISMIC_DIM_H, n, selected,
        projections.at(K) * coefficient[m * SEISMIC_COEFFICIENT_STRIDE_0]);
}

// Channels n .. n + 3's publication, over the K + 1 A-rounded projections
// `projections.at4(k)` and the CPU's `answer`: hoists `scores[m, k]` and
// `coefficient[m]` once across all four channels and publishes via
// `projection::routed_join_quad`.
template <typename Projections>
inline void output_publish4(device const float *residual, device const float *inject, device const float *scores,
    device const float *coefficient, element::Out<element::F32> value, ulong m, uint n,
    thread const Projections &projections, thread const Answer4 &answer, constant ulong *seismic_words) {
    constexpr uint K = uint(SEISMIC_DIM_K);
    device const float *row_scores = scores + m * SEISMIC_SCORES_STRIDE_0;
    float4 selected = float4(0.0f);
    for (uint k = 0; k < K; ++k) {
        const float score = row_scores[k * SEISMIC_SCORES_STRIDE_1];
        selected = metal::fma(float4(score), projections.at4(k), selected);
    }
    if (answer.answered)
        selected += answer.value;
    const float coeff = coefficient[m * SEISMIC_COEFFICIENT_STRIDE_0];
    projection::routed_join_quad<SEISMIC_DIM_SN, (SEISMIC_DIM_IN != 0)>(value.offset(m * SEISMIC_RESULT_0_STRIDE_0),
        SEISMIC_RESULT_0_STRIDE_1, residual + m * SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1,
        inject + m * SEISMIC_INJECT_STRIDE_0, SEISMIC_DIM_H, n, selected, projections.at4(K) * coeff);
}

} // namespace routed
