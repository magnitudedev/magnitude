// The decode down projections of the general routed feed-forward
// (`routed_down`, `routed_down_project`): the counterpart of
// `metal/lib/routed/down.h`. Workgroup (x, m) owns output channels of tile x
// for row m; `routed_down_choice` runs the K1 GEMV sums of choice k's down
// projection (its source's record, `routed_source_weights`: a bank record,
// or a host tier record read in place; none for a choice with no record,
// `routed_decode_source`). Every lane of a lane group (a whole
// subgroup) receives each channel's total; where the includer publishes them
// (`routed_down_project`), `routed_down_publish` writes every choice's
// projection in A at ROUTED_PROJECTED_INDEX(m, k, n) of ROUTED_PROJECTED_BASE.
#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#include "../core/activation.glsl"
#include "routed.glsl"

// The includer defines ROUTED_SOURCE_ROWS, the sources' rows: the entry's
// XS, or a streamed entry's fixed SOURCE_ROWS.

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16) || defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
#define ROUTED_ACT ELEMENT_ACT
#else
#error "routed_down requires a bf16 or f16 activation"
#endif
// The down weight's operand row (the product row, or NVFP4 operands), which
// each choice's GEMV forms while staging it.
#if defined(SEISMIC_EXPERT_DOWN_OPERANDS_NVFP4)
#define DOWN_OPERANDS PROJECTION_OPERANDS_NVFP4
#else
#define DOWN_OPERANDS PROJECTION_OPERANDS_ROUNDED
#endif

#define DOWN_R SEISMIC_TUNE_ROWS

// Choice (m, k)'s projection sums of workgroup tile `tile` into `totals`
// (row r's at r * PROJECTION_GEMV_MAX_M): zero where the device computes
// nothing of it.
void routed_down_choice(const uint64_t m, const uint64_t k, const uint tile, out float totals[PROJECTION_GEMV_SUMS]) {
    const uint hidden = uint(SEISMIC_DIM_H);
    const uint bytes = ELEMENT_BYTES(ROUTED_ACT);
    float unused[PROJECTION_GEMV_SUMS];
    const int source = routed_decode_source(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES), ROUTED_SOURCE_ROWS,
        SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, m, k,
        element_i32_at(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES)
            + (m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1) * 4ul));
    if (source == -1) {
        // A choice with no record (the CPU's, or a failed one) reads none:
        // a zero projection.
        [[unroll]] for (uint sum = 0u; sum < PROJECTION_GEMV_SUMS; ++sum)
            totals[sum] = 0.0;
        return;
    }
    const projection_prologue x = projection_operands(projection_plain(ROUTED_ACT,
            SEISMIC_PTR(SEISMIC_BUFFER_PRODUCT)
            + (m * SEISMIC_PRODUCT_STRIDE_0 + k * SEISMIC_PRODUCT_STRIDE_1) * uint64_t(bytes),
        0ul, SEISMIC_PRODUCT_STRIDE_2, uint(SEISMIC_DIM_F), 0ul), DOWN_OPERANDS);
    const projection_weights down = routed_source_weights(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN),
        SEISMIC_PTR(SEISMIC_BUFFER_TIER_DOWN), source, PACKETS_W0_STACKED_ROWS, uint(SEISMIC_DIM_F));
    projection_gemv_sums(PACKETS_W0, PACKETS_W0, false, DOWN_R, 32u, 1u, PROJECTION_NORM_NONE, x, down, down, 1u,
        hidden, uint(SEISMIC_DIM_F), tile, totals, unused);
}

#ifdef ROUTED_PROJECTED_BASE
// Workgroup (x, m)'s projections of every choice, each channel's by one lane.
void routed_down_publish() {
    const uint tile = gl_WorkGroupID.x;
    const uint64_t m = gl_WorkGroupID.y;
    const uint hidden = uint(SEISMIC_DIM_H);
    const uint first = projection_gemv_first_row(DOWN_R, 32u, tile);
    float totals[PROJECTION_GEMV_SUMS];
    for (uint64_t k = 0ul; k < SEISMIC_DIM_K; ++k) {
        routed_down_choice(m, k, tile, totals);
        [[unroll]] for (uint r = 0u; r < PROJECTION_GEMV_MAX_R; ++r) {
            const uint n = first + r;
            if (r < DOWN_R && n < hidden && SEISMIC_LANE == r)
                element_put(ROUTED_ACT, ROUTED_PROJECTED_BASE, ROUTED_PROJECTED_INDEX(m, k, uint64_t(n)),
                    totals[r * PROJECTION_GEMV_MAX_M]);
        }
    }
}
#endif
