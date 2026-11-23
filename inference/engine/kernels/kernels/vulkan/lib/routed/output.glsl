// The decode output projections (M <= VERIFY_ROWS) of `routed_output` and
// `routed_output_project`: the counterpart of `metal/lib/routed/output.h`.
// Each (row, choice) runs its down projection in a separate workgroup, its
// down weights its source's record (`routed_expert_matrix`: a bank record,
// or a host tier record read in place; none for a choice with no record,
// `routed_decode_source`). Choice K is the shared expert. Their
// join is `lib/routed/output_sum.glsl`'s.
#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#define KERNEL_W1 SEISMIC_SHARED_DOWN
#include "../core/activation.glsl"
#include "routed.glsl"

// The includer defines ROUTED_SOURCE_ROWS, the sources' rows: the entry's
// XS, or a streamed entry's fixed SOURCE_ROWS; and ROUTED_PROJECTED_PUT(m, k,
// n, value), which stores slot k of row m's channel n in A through `Out`.

#if ELEMENT_KIND(SEISMIC_ELEMENT_A) != ELEMENT_BF16 && ELEMENT_KIND(SEISMIC_ELEMENT_A) != ELEMENT_F16
#error "routed_output requires a bf16 or f16 activation"
#endif
#define ROUTED_ACT ACTIVATION_A
// The down weights' operand rows (the product rows, or NVFP4 operands), each
// weight's own, which each (row, choice)'s GEMV forms while staging them.
#if PACKETS_OPERANDS_NVFP4(SEISMIC_EXPERT_DOWN)
#define EXPERT_OPERANDS PROJECTION_OPERANDS_NVFP4
#else
#define EXPERT_OPERANDS PROJECTION_OPERANDS_ROUNDED
#endif
#if PACKETS_OPERANDS_NVFP4(SEISMIC_SHARED_DOWN)
#define SHARED_OPERANDS PROJECTION_OPERANDS_NVFP4
#else
#define SHARED_OPERANDS PROJECTION_OPERANDS_ROUNDED
#endif



#define OUTPUT_R SEISMIC_TUNE_ROWS
// Lanes a weight row: a down row is short (20 packets at Flash Next's 640
// columns), so fewer than a subgroup keep every lane loading.
#define OUTPUT_LANES SEISMIC_TUNE_LANES

// The projection launch: each (row, choice)'s, and the shared expert's,
// A-rounded down projection, published through the includer's
// ROUTED_PROJECTED_PUT(m, slot, n, value).
void routed_output_projection() {
    const uint tile = gl_WorkGroupID.x;
    uint64_t m, choice;
    seismic_split(gl_WorkGroupID.y, SEISMIC_DIM_K + 1ul, m, choice);
    const uint hidden = uint(SEISMIC_DIM_H);
    const uint first = projection_gemv_first_row(OUTPUT_R, OUTPUT_LANES, tile);
    const uint bytes = ELEMENT_BYTES(ROUTED_ACT);
    float totals[PROJECTION_GEMV_SUMS], unused[PROJECTION_GEMV_SUMS];
    if (choice < SEISMIC_DIM_K) {
        const slab_source source = routed_decode_source(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES), ROUTED_SOURCE_ROWS,
            SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, m, choice,
            element_i32_at(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES)
                + (m * SEISMIC_ROUTES_STRIDE_0 + choice * SEISMIC_ROUTES_STRIDE_1) * 4ul));
        if (slab_source_none(source)) {
            // A choice with no record (the CPU's, or a failed one) reads none: a zero
            // projection.
            [[unroll]] for (uint sum = 0u; sum < PROJECTION_GEMV_SUMS; ++sum)
                totals[sum] = 0.0;
        } else {
            const projection_prologue x = projection_operands(projection_plain(ROUTED_ACT,
                    SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_PRODUCT)
                    + (m * SEISMIC_EXPERT_PRODUCT_STRIDE_0 + choice * SEISMIC_EXPERT_PRODUCT_STRIDE_1) * uint64_t(bytes),
                0ul, SEISMIC_EXPERT_PRODUCT_STRIDE_2, uint(SEISMIC_DIM_F), 0ul), EXPERT_OPERANDS);
            const projection_weights down = routed_expert_matrix(slab_experts(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN),
                SEISMIC_PTR(SEISMIC_BUFFER_TIER_DOWN)), source, PACKETS_W0_STACKED_ROWS, uint(SEISMIC_DIM_F));
            projection_gemv_sums(PACKETS_W0, PACKETS_W0, false, OUTPUT_R, OUTPUT_LANES, 1u, false, false,
                PROJECTION_NORM_NONE, x, down, down, 1u, hidden, uint(SEISMIC_DIM_F), tile, totals, unused);
        }
    } else {
        const projection_prologue x = projection_operands(projection_plain(ROUTED_ACT,
                SEISMIC_PTR(SEISMIC_BUFFER_SHARED_PRODUCT) + m * SEISMIC_SHARED_PRODUCT_STRIDE_0 * uint64_t(bytes),
            0ul, SEISMIC_SHARED_PRODUCT_STRIDE_1, uint(SEISMIC_DIM_S), 0ul), SHARED_OPERANDS);
        const projection_weights down = routed_weights(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_DOWN), PACKETS_W1_ROWS, 0ul,
            uint(SEISMIC_DIM_S));
        projection_gemv_sums(PACKETS_W1, PACKETS_W1, false, OUTPUT_R, OUTPUT_LANES, 1u, false, false,
            PROJECTION_NORM_NONE, x, down, down, 1u, hidden, uint(SEISMIC_DIM_S), tile, totals, unused);
    }
    [[unroll]] for (uint r = 0u; r < PROJECTION_GEMV_MAX_R; ++r) {
        const uint n = first + r;
        if (r < OUTPUT_R && n < hidden && SEISMIC_LANE % OUTPUT_LANES == r)
            ROUTED_PROJECTED_PUT(m, choice, uint64_t(n), totals[r * PROJECTION_GEMV_MAX_M]);
    }
}

void routed_output_gather_worklist() {
    const uint per_group = SEISMIC_TUNE_SIMDGROUPS * OUTPUT_R * (32u / OUTPUT_LANES);
    const uint x_groups = (uint(SEISMIC_DIM_H) + per_group - 1u) / per_group;
    const uint mode = (SEISMIC_EXPERT_PRODUCT_STRIDE_0 == SEISMIC_DIM_K * SEISMIC_EXPERT_PRODUCT_STRIDE_1) ? 1u : 2u;
    routed_build_worklist(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES), SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,
        SEISMIC_PTR(SEISMIC_BUFFER_SOURCES), ROUTED_SOURCE_ROWS,
        SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2,
        uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_K),
        SEISMIC_EXPERT_PRODUCT_STRIDE_0, SEISMIC_EXPERT_PRODUCT_STRIDE_1, mode, x_groups, 1u,
        SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_WORKLIST), SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_GATHER_GRID));
}

// Gathered projection launch at activation-row bound `maxm`: workgroup `(tile, u)`
// projects unique expert `u < count` across all its choices in slices of at most
// `maxm` rows, or the shared expert at `u == count` across all M rows.
void routed_output_gathered_bounded(const uint maxm) {
    const uint tile = gl_WorkGroupID.x;
    const uint hidden = uint(SEISMIC_DIM_H);
    const uint first = projection_gemv_first_row(OUTPUT_R, OUTPUT_LANES, tile);
    const uint64_t worklist = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_WORKLIST);
    const uint unique_count = routed_worklist_count(worklist);
    float totals[PROJECTION_GEMV_SUMS], unused[PROJECTION_GEMV_SUMS];
    if (gl_WorkGroupID.y < unique_count) {
        const routed_work_entry entry = routed_worklist_entry(worklist, gl_WorkGroupID.y);
        if (slab_source_none(entry.source)) {
            for (uint i = 0u; i < entry.count; ++i) {
                const uint packed = uint(element_i32_at(entry.outputs + uint64_t(i) * 4ul));
                const uint64_t m = uint64_t(packed >> 16u);
                const uint64_t k = uint64_t(packed & 0xffffu);
                [[unroll]] for (uint r = 0u; r < PROJECTION_GEMV_MAX_R; ++r) {
                    const uint n = first + r;
                    if (r < OUTPUT_R && n < hidden && SEISMIC_LANE % OUTPUT_LANES == r)
                        ROUTED_PROJECTED_PUT(m, k, uint64_t(n), 0.0);
                }
            }
            return;
        }
        const uint64_t row_stride =
            (SEISMIC_EXPERT_PRODUCT_STRIDE_0 == SEISMIC_DIM_K * SEISMIC_EXPERT_PRODUCT_STRIDE_1)
            ? SEISMIC_EXPERT_PRODUCT_STRIDE_1 : 1ul;
        const projection_weights down = routed_expert_matrix(slab_experts(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN),
            SEISMIC_PTR(SEISMIC_BUFFER_TIER_DOWN)), entry.source, PACKETS_W0_STACKED_ROWS, uint(SEISMIC_DIM_F));
        for (uint row_base = 0u; row_base < entry.count; row_base += maxm) {
            const uint slice_m = min(maxm, entry.count - row_base);
            const projection_prologue x = projection_operands(projection_plain(ROUTED_ACT,
                SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_PRODUCT), row_stride, SEISMIC_EXPERT_PRODUCT_STRIDE_2,
                uint(SEISMIC_DIM_F), entry.rows + uint64_t(row_base) * 4ul), EXPERT_OPERANDS);
            projection_gemv_sums(PACKETS_W0, PACKETS_W0, false, OUTPUT_R, OUTPUT_LANES, maxm, false, false,
                PROJECTION_NORM_NONE, x, down, down, slice_m, hidden, uint(SEISMIC_DIM_F), tile, totals, unused);
            for (uint i = 0u; i < maxm; ++i) {
                if (i < slice_m) {
                    const uint packed = uint(element_i32_at(entry.outputs + uint64_t(row_base + i) * 4ul));
                    const uint64_t m = uint64_t(packed >> 16u);
                    const uint64_t k = uint64_t(packed & 0xffffu);
                    [[unroll]] for (uint r = 0u; r < PROJECTION_GEMV_MAX_R; ++r) {
                        const uint n = first + r;
                        if (r < OUTPUT_R && n < hidden && SEISMIC_LANE % OUTPUT_LANES == r)
                            ROUTED_PROJECTED_PUT(m, k, uint64_t(n), totals[r * PROJECTION_GEMV_MAX_M + i]);
                    }
                }
            }
            if (row_base + maxm < entry.count)
                barrier();
        }
        return;
    }
    const projection_prologue x = projection_operands(projection_plain(ROUTED_ACT,
        SEISMIC_PTR(SEISMIC_BUFFER_SHARED_PRODUCT), SEISMIC_SHARED_PRODUCT_STRIDE_0, SEISMIC_SHARED_PRODUCT_STRIDE_1,
        uint(SEISMIC_DIM_S), 0ul), SHARED_OPERANDS);
    const projection_weights down = routed_weights(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_DOWN), PACKETS_W1_ROWS, 0ul,
        uint(SEISMIC_DIM_S));
    const uint m_rows = uint(SEISMIC_DIM_M);
    for (uint row_base = 0u; row_base < m_rows; row_base += maxm) {
        const uint slice_m = min(maxm, m_rows - row_base);
        projection_prologue slice_x = x;
        if (row_base != 0u)
            slice_x.x = stored_offset(slice_x.x, uint64_t(row_base) * slice_x.x0);
        projection_gemv_sums(PACKETS_W1, PACKETS_W1, false, OUTPUT_R, OUTPUT_LANES, maxm, false, false,
            PROJECTION_NORM_NONE, slice_x, down, down, slice_m, hidden, uint(SEISMIC_DIM_S), tile, totals, unused);
        for (uint i = 0u; i < maxm; ++i) {
            if (i < slice_m) {
                [[unroll]] for (uint r = 0u; r < PROJECTION_GEMV_MAX_R; ++r) {
                    const uint n = first + r;
                    if (r < OUTPUT_R && n < hidden && SEISMIC_LANE % OUTPUT_LANES == r)
                        ROUTED_PROJECTED_PUT(uint64_t(row_base + i), SEISMIC_DIM_K, uint64_t(n),
                            totals[r * PROJECTION_GEMV_MAX_M + i]);
                }
            }
        }
        if (row_base + maxm < m_rows)
            barrier();
    }
}
