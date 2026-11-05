// The slot-order sum of the general routed feed-forward's decode combine,
// shared by `routed_down` (over its workgroup's projections, in every lane's
// registers) and `routed_down_sited` (over the published ones, with the
// CPU's answer): the counterpart of `metal/lib/routed/down_sum.h`. Channel
// n of row m is base + selected in R, `selected` accumulating weight *
// projection in slot order (`routed_down_term`), then the CPU's weighted
// sum of the row where it answered (ROUTED_SITED, `lib/routed/sited.glsl`),
// absent from `routed_down`. The includer includes
// `lib/core/activation.glsl` first.
#if defined(SEISMIC_ELEMENT_R_REPRESENTATION_F32)
#define ROUTED_PUBLISHED ELEMENT_F32
#elif defined(SEISMIC_ELEMENT_R_REPRESENTATION_BF16)
#define ROUTED_PUBLISHED ELEMENT_BF16
#elif defined(SEISMIC_ELEMENT_R_REPRESENTATION_F16)
#define ROUTED_PUBLISHED ELEMENT_F16
#else
#error "the down sum requires a dense published element"
#endif

// The sum `selected` of row m's choices before slot k with choice k's
// A-rounded projection `projected` weighted in.
float routed_down_term(const float selected, const uint64_t m, const uint64_t k, const float projected) {
    const float weight = element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHTS)
        + (m * SEISMIC_WEIGHTS_STRIDE_0 + k * SEISMIC_WEIGHTS_STRIDE_1) * 4ul);
    return seismic_fma_rn(weight, projected, selected);
}

// Channel n of row m's publication: base + `selected`, in R.
void routed_down_put(const uint64_t m, const uint64_t n, const float selected) {
    const float base = element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_BASE)
        + (m * SEISMIC_BASE_STRIDE_0 + n * SEISMIC_BASE_STRIDE_1) * 4ul);
    element_put(ROUTED_PUBLISHED, SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER),
        m * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1, base + selected);
}

#ifdef ROUTED_PROJECTED_BASE
// Channel n of row m over the includer's published projections, in A (slot
// k at ROUTED_PROJECTED_INDEX(m, k, n) of ROUTED_PROJECTED_BASE), the CPU's
// answer `answered` of flight slot `slot`: invocation (n, m) of a workgroup
// of 256 channels of row m.
void routed_down_sum_channel(const bool answered, const uint64_t slot) {
    const uint64_t m = gl_WorkGroupID.y;
    const uint64_t n = uint64_t(gl_WorkGroupID.x) * gl_WorkGroupSize.x + gl_LocalInvocationIndex;
    if (n >= SEISMIC_DIM_H)
        return;
    float selected = 0.0;
    for (uint64_t k = 0ul; k < SEISMIC_DIM_K; ++k) {
        const float projected = element_at(ELEMENT_ACT, ROUTED_PROJECTED_BASE, ROUTED_PROJECTED_INDEX(m, k, n));
        selected = routed_down_term(selected, m, k, projected);
    }
#ifdef ROUTED_SITED
    selected = routed_sited(selected, answered, slot, m, n);
#endif
    routed_down_put(m, n, selected);
}
#endif
