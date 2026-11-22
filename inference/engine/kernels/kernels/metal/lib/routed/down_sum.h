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

#ifndef ROUTED_ANSWER4
#define ROUTED_ANSWER4
// Four adjacent channels n .. n + 3 of the CPU's answer (`site_output`).
struct Answer4 {
    bool answered;
    float4 value;
};
#endif


// Channel n's publication, over the K choices' A-rounded projections
// `projections.at(k)` and the CPU's `answer`.
template <typename Projections>
inline void down_publish(device const float *base, device const float *weights,
    element::Out<ELEMENT_OF(SEISMIC_ELEMENT_R)> value, ulong m, uint n, thread const Projections &projections,
    thread const Answer &answer, constant ulong *seismic_words) {
    constexpr uint K = uint(SEISMIC_DIM_K);
    float selected = 0.0f;
    for (uint choice = 0; choice < K; ++choice)
        selected = metal::fma(weights[m * SEISMIC_WEIGHTS_STRIDE_0 + choice * SEISMIC_WEIGHTS_STRIDE_1],
            projections.at(choice), selected);
    if (answer.answered)
        selected += answer.value;
    value.put(m * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1,
        base[m * SEISMIC_BASE_STRIDE_0 + n * SEISMIC_BASE_STRIDE_1] + selected);
}

// Four adjacent channels n .. n + 3's publication, over the K choices'
// A-rounded projections `projections.at4(k)` and the CPU's `answer`: hoists
// `weights[m, choice]` once across all four channels, and uses `packed_float4`
// loads for `base` and `value.put4` stores when their column strides are unit.
template <typename Projections>
inline void down_publish4(device const float *base, device const float *weights,
    element::Out<ELEMENT_OF(SEISMIC_ELEMENT_R)> value, ulong m, uint n, thread const Projections &projections,
    thread const Answer4 &answer, constant ulong *seismic_words) {
    constexpr uint K = uint(SEISMIC_DIM_K);
    float4 selected = float4(0.0f);
    for (uint choice = 0; choice < K; ++choice) {
        const float weight = weights[m * SEISMIC_WEIGHTS_STRIDE_0 + choice * SEISMIC_WEIGHTS_STRIDE_1];
        selected = metal::fma(float4(weight), projections.at4(choice), selected);
    }
    if (answer.answered)
        selected += answer.value;
    const ulong base_row = m * SEISMIC_BASE_STRIDE_0;
    const ulong out_row = m * SEISMIC_RESULT_0_STRIDE_0;
    float4 base4;
    if (SEISMIC_BASE_STRIDE_1 == 1) {
        base4 = float4(*reinterpret_cast<device const packed_float4 *>(base + base_row + ulong(n)));
    } else {
        base4 = float4(
            base[base_row + ulong(n + 0u) * SEISMIC_BASE_STRIDE_1],
            base[base_row + ulong(n + 1u) * SEISMIC_BASE_STRIDE_1],
            base[base_row + ulong(n + 2u) * SEISMIC_BASE_STRIDE_1],
            base[base_row + ulong(n + 3u) * SEISMIC_BASE_STRIDE_1]);
    }
    const float4 sum = base4 + selected;
    if (SEISMIC_RESULT_0_STRIDE_1 == 1) {
        value.put4(out_row + ulong(n), sum);
    } else {
        for (uint i = 0u; i < 4u; ++i)
            value.put(out_row + ulong(n + i) * SEISMIC_RESULT_0_STRIDE_1, sum[i]);
    }
}

} // namespace routed
