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


// Channel n's publication, over the K + 1 A-rounded projections
// `projections.at(k)` and the CPU's `answer`.
template <typename Projections>
inline void output_publish(device const float *residual, device const float *inject, device const float *scores,
    device const float *coefficient, device float *value, ulong m, uint n,
    thread const Projections &projections, thread const Answer &answer, constant ulong *seismic_words) {
    constexpr uint K = uint(SEISMIC_DIM_K);
    float selected = 0.0f;
    for (uint k = 0; k < K; ++k)
        selected = metal::fma(scores[m * SEISMIC_SCORES_STRIDE_0 + k * SEISMIC_SCORES_STRIDE_1], projections.at(k),
            selected);
    if (answer.answered)
        selected += answer.value;
    projection::routed_join<SEISMIC_DIM_SN, (SEISMIC_DIM_IN != 0)>(value + m * SEISMIC_RESULT_0_STRIDE_0,
        SEISMIC_RESULT_0_STRIDE_1, residual + m * SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1,
        inject + m * SEISMIC_INJECT_STRIDE_0, SEISMIC_DIM_H, n, selected,
        projections.at(K) * coefficient[m * SEISMIC_COEFFICIENT_STRIDE_0]);
}

} // namespace routed
