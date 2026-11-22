// The slot-order sum of the general routed feed-forward's decode combine,
// shared by `routed_down` (over its workgroup's projections, in each lane's
// scalar register) and `routed_down_sited` (over the published ones, with the
// CPU's answer): the counterpart of `metal/lib/routed/down_sum.h`. Channel
// n of row m is base + selected in R, `selected` accumulating weight *
// projection in slot order (`routed_down_term_weighted`), then the CPU's
// weighted sum of the row where it answered (ROUTED_SITED,
// `lib/routed/sited.glsl`), absent from `routed_down`. The includer includes
// `lib/core/activation.glsl` first.

// The base address of row m's weights (`weights[m, 0]`).
uint64_t routed_down_weight_row(const uint64_t m) {
    return SEISMIC_PTR(SEISMIC_BUFFER_WEIGHTS) + m * SEISMIC_WEIGHTS_STRIDE_0 * 4ul;
}

// One choice's weight at `weight_at`, loaded once per logical subgroup and
// broadcast to every lane. Every lane of the logical subgroup calls it
// convergently.
float routed_down_weight_subgroup(const uint64_t weight_at) {
    float weight = 0.0;
    if (SEISMIC_LANE == 0u)
        weight = element_f32_at(weight_at);
    return seismic_shuffle(weight, 0u);
}

// The sum `selected` with a choice's pre-loaded `weight` and A-rounded
// projection `projected` weighted in.
float routed_down_term_weighted(const float selected, const float weight, const float projected) {
    return seismic_fma_rn(weight, projected, selected);
}

// Channel n of row m's publication: base + `selected`, in R.
void routed_down_put(const uint64_t m, const uint64_t n, const float selected) {
    const float base = element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_BASE)
        + (m * SEISMIC_BASE_STRIDE_0 + n * SEISMIC_BASE_STRIDE_1) * 4ul);
    out_put(SEISMIC_RESULT_0_OUT,
        m * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1, base + selected);
}

#ifdef ROUTED_PROJECTED_BASE
// Channel n of row m over the includer's published projections, in A (slot
// k at ROUTED_PROJECTED_INDEX(m, k, n) of ROUTED_PROJECTED_BASE), the CPU's
// answer `answered` of flight slot `slot`: invocation (n, m) of a workgroup
// of 256 channels of row m.
void routed_down_sum_channel(const bool answered, const uint64_t slot) {
    const uint64_t m = gl_WorkGroupID.y;
    const uint64_t n = uint64_t(gl_WorkGroupID.x) * gl_WorkGroupSize.x + uint64_t(SEISMIC_THREAD);
    const bool live = n < SEISMIC_DIM_H;
    const uint64_t weight_step = SEISMIC_WEIGHTS_STRIDE_1 * 4ul;
    uint64_t weight_at = routed_down_weight_row(m);
    const uint64_t projected_base = ROUTED_PROJECTED_BASE;
    const uint64_t projected_step = ROUTED_PROJECTED_INDEX(0ul, 1ul, 0ul);
    uint64_t projected_at = ROUTED_PROJECTED_INDEX(m, 0ul, n);
    float selected = 0.0;
    // A 32-bit counter: see `streaming/contract.hpp`.
    for (uint k = 0u; k < uint(SEISMIC_DIM_K); ++k) {
        const float weight = routed_down_weight_subgroup(weight_at);
        weight_at += weight_step;
        if (live) {
            const float projected = element_at(ACTIVATION_A, projected_base, projected_at);
            selected = routed_down_term_weighted(selected, weight, projected);
            projected_at += projected_step;
        }
    }
    if (!live)
        return;
#ifdef ROUTED_SITED
    selected = routed_sited(selected, answered, slot, m, n);
#endif
    routed_down_put(m, n, selected);
}
#endif
