// The slot-order sum of the general routed feed-forward's decode combine,
// shared by `routed_down` (over its threadgroup's projections) and
// `routed_down_sited` (over the published ones, with the CPU's answer): the
// thread of channel n of row m publishes
//     base + selected
// in R, where `selected` accumulates weight * round_A(projection) in slot
// order, then adds the CPU's answer where it answered
// (`lib/routed/sited.h`; none in `routed_down`).

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


// Channel n's publication, over the K choices' A-rounded projections
// `projections.at(k)` and the CPU's `answer`.
template <typename Projections>
inline void down_publish(device const float *base, device const float *weights, device uchar *value, ulong m, uint n,
    thread const Projections &projections, thread const Answer &answer, constant ulong *seismic_words) {
    typedef ELEMENT_OF(SEISMIC_ELEMENT_R) Published;
    constexpr uint K = uint(SEISMIC_DIM_K);
    float selected = 0.0f;
    for (uint choice = 0; choice < K; ++choice)
        selected = metal::fma(weights[m * SEISMIC_WEIGHTS_STRIDE_0 + choice * SEISMIC_WEIGHTS_STRIDE_1],
            projections.at(choice), selected);
    if (answer.answered)
        selected += answer.value;
    reinterpret_cast<device typename Published::storage *>(value)[m * SEISMIC_RESULT_0_STRIDE_0
        + n * SEISMIC_RESULT_0_STRIDE_1] = Published::store(base[m * SEISMIC_BASE_STRIDE_0 + n * SEISMIC_BASE_STRIDE_1]
        + selected);
}

} // namespace routed
