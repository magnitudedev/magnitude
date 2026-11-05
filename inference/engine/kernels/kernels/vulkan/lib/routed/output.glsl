// The decode output projections (M <= DECODE_ROWS) of `routed_output` and
// `routed_output_project`: the counterpart of `metal/lib/routed/output.h`.
// Each (row, choice) runs its down projection in a separate workgroup, its
// down weights its source's record (`routed_source_weights`: a bank record,
// or a host tier record read in place; none for a choice with no record,
// `routed_decode_source`). Choice K is the shared expert. Their
// join is `lib/routed/output_sum.glsl`'s.
#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#define KERNEL_W1 SEISMIC_SHARED_DOWN
#include "../core/activation.glsl"
#include "routed.glsl"

// The includer defines ROUTED_SOURCE_ROWS, the sources' rows: the entry's
// XS, or a streamed entry's fixed SOURCE_ROWS.

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16) || defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
#define ROUTED_ACT ELEMENT_ACT
#else
#error "routed_output requires a bf16 or f16 activation"
#endif
// The down weights' operand rows (the product rows, or NVFP4 operands), each
// weight's own, which each (row, choice)'s GEMV forms while staging them.
#if defined(SEISMIC_EXPERT_DOWN_OPERANDS_NVFP4)
#define EXPERT_OPERANDS PROJECTION_OPERANDS_NVFP4
#else
#define EXPERT_OPERANDS PROJECTION_OPERANDS_ROUNDED
#endif
#if defined(SEISMIC_SHARED_DOWN_OPERANDS_NVFP4)
#define SHARED_OPERANDS PROJECTION_OPERANDS_NVFP4
#else
#define SHARED_OPERANDS PROJECTION_OPERANDS_ROUNDED
#endif



#define OUTPUT_R SEISMIC_TUNE_ROWS

// The projection launch: each (row, choice)'s, and the shared expert's,
// A-rounded down projection at ROUTED_PROJECTED_INDEX(m, slot, n) of the
// includer's projections (ROUTED_PROJECTED_KIND at ROUTED_PROJECTED_BASE).
void routed_output_projection() {
    const uint tile = gl_WorkGroupID.x;
    const uint64_t m = gl_WorkGroupID.y / (SEISMIC_DIM_K + 1ul);
    const uint64_t choice = gl_WorkGroupID.y % (SEISMIC_DIM_K + 1ul);
    const uint hidden = uint(SEISMIC_DIM_H);
    const uint first = projection_gemv_first_row(OUTPUT_R, 32u, tile);
    const uint bytes = ELEMENT_BYTES(ROUTED_ACT);
    float totals[PROJECTION_GEMV_SUMS], unused[PROJECTION_GEMV_SUMS];
    if (choice < SEISMIC_DIM_K) {
        const int source = routed_decode_source(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES), ROUTED_SOURCE_ROWS,
            SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, m, choice,
            element_i32_at(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES)
                + (m * SEISMIC_ROUTES_STRIDE_0 + choice * SEISMIC_ROUTES_STRIDE_1) * 4ul));
        if (source == -1) {
            // A choice with no record (the CPU's, or a failed one) reads none: a zero
            // projection.
            [[unroll]] for (uint sum = 0u; sum < PROJECTION_GEMV_SUMS; ++sum)
                totals[sum] = 0.0;
        } else {
            const projection_prologue x = projection_operands(projection_plain(ROUTED_ACT,
                    SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_PRODUCT)
                    + (m * SEISMIC_EXPERT_PRODUCT_STRIDE_0 + choice * SEISMIC_EXPERT_PRODUCT_STRIDE_1) * uint64_t(bytes),
                0ul, SEISMIC_EXPERT_PRODUCT_STRIDE_2, uint(SEISMIC_DIM_F), 0ul), EXPERT_OPERANDS);
            const projection_weights down = routed_source_weights(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN),
                SEISMIC_PTR(SEISMIC_BUFFER_TIER_DOWN), source, PACKETS_W0_STACKED_ROWS, uint(SEISMIC_DIM_F));
            projection_gemv_sums(PACKETS_W0, PACKETS_W0, false, OUTPUT_R, 32u, 1u, PROJECTION_NORM_NONE, x, down, down,
                1u, hidden, uint(SEISMIC_DIM_F), tile, totals, unused);
        }
    } else {
        const projection_prologue x = projection_operands(projection_plain(ROUTED_ACT,
                SEISMIC_PTR(SEISMIC_BUFFER_SHARED_PRODUCT) + m * SEISMIC_SHARED_PRODUCT_STRIDE_0 * uint64_t(bytes),
            0ul, SEISMIC_SHARED_PRODUCT_STRIDE_1, uint(SEISMIC_DIM_S), 0ul), SHARED_OPERANDS);
        const projection_weights down = routed_weights(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_DOWN), PACKETS_W1_ROWS, 0ul,
            uint(SEISMIC_DIM_S));
        projection_gemv_sums(PACKETS_W1, PACKETS_W1, false, OUTPUT_R, 32u, 1u, PROJECTION_NORM_NONE, x, down, down, 1u,
            hidden, uint(SEISMIC_DIM_S), tile, totals, unused);
    }
    [[unroll]] for (uint r = 0u; r < PROJECTION_GEMV_MAX_R; ++r) {
        const uint n = first + r;
        if (r < OUTPUT_R && n < hidden && SEISMIC_LANE == r)
            element_put(ROUTED_PROJECTED_KIND, ROUTED_PROJECTED_BASE, ROUTED_PROJECTED_INDEX(m, choice, uint64_t(n)),
                element_round(ROUTED_ACT, totals[r * PROJECTION_GEMV_MAX_M]));
    }
}
