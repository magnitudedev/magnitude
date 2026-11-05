// The join of the decode output (`routed_output`, and `routed_output_sited`
// with the CPU's answer, ROUTED_SITED): the counterpart of
// `metal/lib/routed/output_sum.h`, over the includer's A-rounded projections
// (ROUTED_PROJECTED_KIND at ROUTED_PROJECTED_BASE, slot k of row m at
// ROUTED_PROJECTED_INDEX(m, k, n), slot K the shared expert's).

// The combine launch: row m's choices weighted in slot order, then the
// CPU's weighted sum of the row where it answered the step (`answered`, of
// flight slot `slot`; an entry with the answer's words, ROUTED_SITED), then
// the shared expert and the residual.
void routed_output_join(const bool answered, const uint64_t slot) {
    const uint64_t m = gl_WorkGroupID.y;
    const uint64_t n = uint64_t(gl_WorkGroupID.x) * 256ul + gl_LocalInvocationIndex;
    if (n >= SEISMIC_DIM_H)
        return;
    float selected = 0.0;
    for (uint64_t choice = 0ul; choice < SEISMIC_DIM_K; ++choice) {
        const float score = element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_SCORES)
            + (m * SEISMIC_SCORES_STRIDE_0 + choice * SEISMIC_SCORES_STRIDE_1) * 4ul);
        const float projected =
            element_at(ROUTED_PROJECTED_KIND, ROUTED_PROJECTED_BASE, ROUTED_PROJECTED_INDEX(m, choice, n));
        selected = seismic_fma_rn(score, projected, selected);
    }
#ifdef ROUTED_SITED
    selected = routed_sited(selected, answered, slot, m, n);
#endif
    const float projected_shared =
        element_at(ROUTED_PROJECTED_KIND, ROUTED_PROJECTED_BASE, ROUTED_PROJECTED_INDEX(m, SEISMIC_DIM_K, n));
    const float coefficient = element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENT) + m * SEISMIC_COEFFICIENT_STRIDE_0 * 4ul);
    projection_routed_join(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1,
        SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL), SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1,
        SEISMIC_PTR(SEISMIC_BUFFER_INJECT), SEISMIC_INJECT_STRIDE_0, uint(SEISMIC_DIM_IN), uint(SEISMIC_DIM_SN),
        uint(SEISMIC_DIM_H), m, n, selected,
        projected_shared * coefficient);
}
