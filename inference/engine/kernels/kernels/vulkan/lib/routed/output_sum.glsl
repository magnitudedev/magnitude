// The join of the decode output (`routed_output`, and `routed_output_sited`
// with the CPU's answer, ROUTED_SITED): the counterpart of
// `metal/lib/routed/output_sum.h`, over the includer's A-rounded projections
// (ROUTED_PROJECTED_KIND at ROUTED_PROJECTED_BASE, slot k of row m at
// ROUTED_PROJECTED_INDEX(m, k, n), slot K the shared expert's).

// Four contiguous channels of slot k of row m from the includer's projected
// tensor, vectorized when unit-strided and aligned (16 bytes for F32, 8 bytes
// for 16-bit), scalar otherwise.
vec4 routed_output_projected4(const uint64_t m, const uint64_t k, const uint64_t n, const bool unit_stride) {
    const uint64_t idx = ROUTED_PROJECTED_INDEX(m, k, n);
    if (unit_stride) {
        if (ROUTED_PROJECTED_KIND == ELEMENT_F32) {
            const uint64_t addr = ROUTED_PROJECTED_BASE + idx * 4ul;
            if ((addr & 15ul) == 0ul)
                return element_vec4_at(addr);
        } else {
            const uint64_t addr = ROUTED_PROJECTED_BASE + idx * 2ul;
            if ((addr & 7ul) == 0ul) {
                const uvec2 words = element_uvec2_at(addr);
                return vec4(element_unpack2(ROUTED_PROJECTED_KIND, words.x),
                    element_unpack2(ROUTED_PROJECTED_KIND, words.y));
            }
        }
    }
    return vec4(
        element_at(ROUTED_PROJECTED_KIND, ROUTED_PROJECTED_BASE, idx),
        element_at(ROUTED_PROJECTED_KIND, ROUTED_PROJECTED_BASE, ROUTED_PROJECTED_INDEX(m, k, n + 1ul)),
        element_at(ROUTED_PROJECTED_KIND, ROUTED_PROJECTED_BASE, ROUTED_PROJECTED_INDEX(m, k, n + 2ul)),
        element_at(ROUTED_PROJECTED_KIND, ROUTED_PROJECTED_BASE, ROUTED_PROJECTED_INDEX(m, k, n + 3ul)));
}

// Four contiguous F32 residual elements at row m, column `column`, vectorized
// when unit-strided and 16-byte aligned.
vec4 routed_output_residual4(const uint64_t residual, const uint64_t residual0, const uint64_t residual1,
    const uint64_t m, const uint64_t column) {
    const uint64_t row_base = residual + m * residual0 * 4ul;
    if (residual1 == 1ul) {
        const uint64_t addr = row_base + column * 4ul;
        if ((addr & 15ul) == 0ul)
            return element_vec4_at(addr);
    }
    return vec4(
        element_f32_at(row_base + column * residual1 * 4ul),
        element_f32_at(row_base + (column + 1ul) * residual1 * 4ul),
        element_f32_at(row_base + (column + 2ul) * residual1 * 4ul),
        element_f32_at(row_base + (column + 3ul) * residual1 * 4ul));
}

// Four output channels at row m, column `column`, using `out_put4` when
// unit-strided and scalar `out_put` otherwise.
void routed_output_put4(Out y, const uint64_t y0, const uint64_t y1, const uint64_t m, const uint64_t column,
    const vec4 value) {
    if (y1 == 1ul) {
        out_put4(y, m * y0 + column, value);
    } else {
        const uint64_t row_at = m * y0;
        out_put(y, row_at + column * y1, value.x);
        out_put(y, row_at + (column + 1ul) * y1, value.y);
        out_put(y, row_at + (column + 2ul) * y1, value.z);
        out_put(y, row_at + (column + 3ul) * y1, value.w);
    }
}

// The combine launch: row m's choices weighted in slot order, then the
// CPU's weighted sum of the row where it answered the step (`answered`, of
// flight slot `slot`; an entry with the answer's words, ROUTED_SITED), then
// the shared expert and the residual. Each invocation processes 4 contiguous
// channels (1024 channels per 256-thread workgroup), hoisting scores, shared
// coefficient, and inject weights across the 4 channels.
void routed_output_join(const bool answered, const uint64_t slot) {
    const uint64_t m = gl_WorkGroupID.y;
    const uint64_t n = (uint64_t(gl_WorkGroupID.x) * 256ul + uint64_t(gl_LocalInvocationIndex)) * 4ul;
    if (n >= SEISMIC_DIM_H)
        return;

    if (n + 4ul > SEISMIC_DIM_H) {
        for (uint64_t c = n; c < SEISMIC_DIM_H; ++c) {
            float selected = 0.0;
            for (uint64_t choice = 0ul; choice < SEISMIC_DIM_K; ++choice) {
                const float score = element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_SCORES)
                    + (m * SEISMIC_SCORES_STRIDE_0 + choice * SEISMIC_SCORES_STRIDE_1) * 4ul);
                const float projected =
                    element_at(ROUTED_PROJECTED_KIND, ROUTED_PROJECTED_BASE, ROUTED_PROJECTED_INDEX(m, choice, c));
                selected = seismic_fma_rn(score, projected, selected);
            }
#ifdef ROUTED_SITED
            selected = routed_sited(selected, answered, slot, m, c);
#endif
            const float projected_shared =
                element_at(ROUTED_PROJECTED_KIND, ROUTED_PROJECTED_BASE, ROUTED_PROJECTED_INDEX(m, SEISMIC_DIM_K, c));
            const float coefficient = element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENT)
                + m * SEISMIC_COEFFICIENT_STRIDE_0 * 4ul);
            projection_routed_join(SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1,
                SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL), SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1,
                SEISMIC_PTR(SEISMIC_BUFFER_INJECT), SEISMIC_INJECT_STRIDE_0, uint(SEISMIC_DIM_IN), uint(SEISMIC_DIM_SN),
                uint(SEISMIC_DIM_H), m, c, selected,
                projected_shared * coefficient);
        }
        return;
    }

    const bool projected_unit =
        (ROUTED_PROJECTED_INDEX(0ul, 0ul, 1ul) - ROUTED_PROJECTED_INDEX(0ul, 0ul, 0ul)) == 1ul;
    vec4 selected = vec4(0.0);
    for (uint64_t choice = 0ul; choice < SEISMIC_DIM_K; ++choice) {
        const float score = element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_SCORES)
            + (m * SEISMIC_SCORES_STRIDE_0 + choice * SEISMIC_SCORES_STRIDE_1) * 4ul);
        const vec4 projected = routed_output_projected4(m, choice, n, projected_unit);
        selected = vec4(
            seismic_fma_rn(score, projected.x, selected.x),
            seismic_fma_rn(score, projected.y, selected.y),
            seismic_fma_rn(score, projected.z, selected.z),
            seismic_fma_rn(score, projected.w, selected.w));
    }
#ifdef ROUTED_SITED
    selected = routed_sited4(selected, answered, slot, m, n);
#endif
    const vec4 projected_shared = routed_output_projected4(m, SEISMIC_DIM_K, n, projected_unit);
    const float coefficient = element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENT)
        + m * SEISMIC_COEFFICIENT_STRIDE_0 * 4ul);
    const vec4 expert = projected_shared * coefficient;
    const uint64_t residual = SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL);
    if (SEISMIC_DIM_IN == 0ul) {
        const vec4 value = (routed_output_residual4(residual, SEISMIC_RESIDUAL_STRIDE_0,
            SEISMIC_RESIDUAL_STRIDE_1, m, n) + selected) + expert;
        routed_output_put4(SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1,
            m, n, value);
        return;
    }
    const vec4 combined = selected + expert;
    for (uint s = 0u; s < uint(SEISMIC_DIM_SN); ++s) {
        const uint64_t column = uint64_t(s) * SEISMIC_DIM_H + n;
        const float weight = element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_INJECT)
            + (m * SEISMIC_INJECT_STRIDE_0 + uint64_t(s)) * 4ul);
        const vec4 value = routed_output_residual4(residual, SEISMIC_RESIDUAL_STRIDE_0,
            SEISMIC_RESIDUAL_STRIDE_1, m, column) + combined * weight;
        routed_output_put4(SEISMIC_RESULT_0_OUT, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1,
            m, column, value);
    }
}
