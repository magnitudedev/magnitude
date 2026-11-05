// The shared half of Vulkan's routed_expand: same projection primitive,
// operand rounding, 32-lane reduction and SiLU-multiply epilogue.
#if defined(SEISMIC_SHARED_GATE_OPERANDS_NVFP4)
#define ROUTE_SHARED_OPERANDS PROJECTION_OPERANDS_NVFP4
#else
#define ROUTE_SHARED_OPERANDS PROJECTION_OPERANDS_ROUNDED
#endif

void routed_route_shared_product(const uint maxm, const uint tile) {
    const uint hidden = uint(SEISMIC_DIM_H);
    const projection_prologue x = projection_operands(projection_plain(ELEMENT_ACT,
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0,
        SEISMIC_RESULT_0_STRIDE_1, hidden, 0ul), ROUTE_SHARED_OPERANDS);
    const projection_weights gate = routed_weights(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_GATE),
        PACKETS_W1_ROWS, 0ul, hidden);
    const projection_weights up = routed_weights(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_UP),
        PACKETS_W2_ROWS, 0ul, hidden);
    const projection_epilogue product = projection_silu_mul(ELEMENT_ACT,
        SEISMIC_PTR(SEISMIC_RESULT_2_BUFFER), SEISMIC_RESULT_2_STRIDE_0, SEISMIC_RESULT_2_STRIDE_1);
    projection_gemv(PACKETS_W1, PACKETS_W2, true, SEISMIC_TUNE_ROWS, 32u, maxm,
        PROJECTION_NORM_NONE, x, product, gate, up, uint(SEISMIC_DIM_M), uint(SEISMIC_DIM_S),
        hidden, tile);
}

// After the logits dispatch, normalized rows are visible to every group.
// Group zero selects the rows; other groups expand the independent shared
// expert. A subgroup may own several rows, but every lane reaches the
// protocol's workgroup barrier only after its row loop has finished.
void routed_route_shared_dispatch(const uint maxm, const bool whole) {
    if (gl_WorkGroupID.x == 0u) {
        for (uint64_t row = SEISMIC_SUBGROUP; row < SEISMIC_DIM_M; row += SEISMIC_SUBGROUPS) {
            int winners[ROUTE_K];
            route_select_row(row, winners);
        }
        ROUTE_FINISH_SELECTION(whole);
    } else {
        routed_route_shared_product(maxm, gl_WorkGroupID.x - 1u);
    }
}
