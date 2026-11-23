// Decode expansion of gated experts (M <= VERIFY_ROWS) in Vulkan: what `routed_gate_up` and
// `routed_gate_up_part` share (see `vulkan/routed_gate_up.comp` for the forms). The includer defines
// the glu limit and scales, KERNEL_W0 and KERNEL_W1, the includes of `lib/core/activation.glsl` and
// `lib/routed/routed.glsl`, then the product's handle and strides (ROUTED_GATE_UP_PRODUCT,
// ROUTED_GATE_UP_PRODUCT_STRIDE_0..2) and ROUTED_GATE_UP_SOURCE_ROWS; a pass of
// `routed_gate_up_part` defines ROUTED_CHOICE_ACTIVE (`lib/routed/routed.glsl`) before including
// `routed.glsl`, so a choice it leaves to the other pass neither runs nor groups.

#if ELEMENT_KIND(SEISMIC_ELEMENT_A) != ELEMENT_BF16 && ELEMENT_KIND(SEISMIC_ELEMENT_A) != ELEMENT_F16
#error "routed_gate_up requires a bf16 or f16 activation"
#endif
#define ROUTED_ACT ACTIVATION_A
// The row both weights take (the normalized row, or NVFP4 operands), formed
// while the GEMV stages it.
#if PACKETS_OPERANDS_NVFP4(SEISMIC_EXPERT_GATE)
#define ROUTED_OPERANDS PROJECTION_OPERANDS_NVFP4
#else
#define ROUTED_OPERANDS PROJECTION_OPERANDS_ROUNDED
#endif

// The GEMV over the gate and up rows of choice `choice`'s expert for `count`
// activation rows: GEMV row r reads activation row `rows[r]` (0: the
// choice's own row) and publishes at `outputs[r]` (0: the choice's
// product), at activation-row bound `maxm`.
void routed_gate_up_rows(const uint64_t choice, uint64_t rows, uint64_t outputs, uint count, const uint maxm) {
    const uint rows_per_tile = SEISMIC_SUBGROUPS * SEISMIC_TUNE_ROWS;
    const uint tile = gl_WorkGroupID.x;
    const uint hidden = uint(SEISMIC_DIM_H);
    uint64_t m, k;
    seismic_split(uint(choice), SEISMIC_DIM_K, m, k);
    const uint64_t expert = uint64_t(element_i32_at(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES)
        + (m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1) * 4ul));
    const slab_source source = routed_decode_source(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES), ROUTED_GATE_UP_SOURCE_ROWS,
        SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, m, k, int(expert));
    const uint bytes = ELEMENT_BYTES(ROUTED_ACT);
    const uint64_t own = rows == 0ul ? m * SEISMIC_NORMALIZED_STRIDE_0 * uint64_t(bytes) : 0ul;
    const projection_prologue x = projection_operands(projection_plain(ROUTED_ACT,
        SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED) + own, SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1, hidden,
        rows), ROUTED_OPERANDS);
    const uint64_t product = outputs == 0ul
        ? m * ROUTED_GATE_UP_PRODUCT_STRIDE_0 + k * ROUTED_GATE_UP_PRODUCT_STRIDE_1
        : 0ul;
    const projection_weights gate = routed_expert_matrix(slab_experts(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_GATE),
        SEISMIC_PTR(SEISMIC_BUFFER_TIER_GATE)), source, PACKETS_W0_STACKED_ROWS, hidden);
    const projection_weights up = routed_expert_matrix(slab_experts(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP),
        SEISMIC_PTR(SEISMIC_BUFFER_TIER_UP)), source, PACKETS_W1_STACKED_ROWS, hidden);
    const projection_epilogue out_ = projection_scaled(projection_gathered(projection_glu(ROUTED_ACT,
            out_offset(ROUTED_GATE_UP_PRODUCT, product), 0ul, ROUTED_GATE_UP_PRODUCT_STRIDE_2,
            int(SEISMIC_PARAM_ACTIVATION)), outputs),
        projection_scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_GATE_SCALE), SEISMIC_DIM_GS, SEISMIC_GATE_SCALE_STRIDE_0, expert),
        projection_scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0, expert));
    if (!ROUTED_CHOICE_ACTIVE(uint(choice), source))
        return;
    // A choice with no record reads none: every tile zeroes its rows.
    if (slab_source_none(source)) {
        routed_zero_features(out_, count, tile * rows_per_tile, rows_per_tile, uint(SEISMIC_DIM_F));
        return;
    }
    projection_gemv(PACKETS_W0, PACKETS_W1, true, SEISMIC_TUNE_ROWS, 32u, maxm, false, false,
        PROJECTION_NORM_NONE, x, out_, gate, up, count, uint(SEISMIC_DIM_F), hidden, tile);
}

// Compact worklist entry `gl_WorkGroupID.y`'s GEMV over its record's rows
// (dispatched indirectly over only the active unique `(expert, source)` groups),
// at activation-row bound `maxm`.
void routed_gate_up_gathered_bounded(const uint maxm) {
    const routed_work_entry entry =
        routed_worklist_entry(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_WORKLIST), gl_WorkGroupID.y);
    const uint rows_per_tile = SEISMIC_SUBGROUPS * SEISMIC_TUNE_ROWS;
    const uint tile = gl_WorkGroupID.x;
    const uint hidden = uint(SEISMIC_DIM_H);
    const uint64_t expert = uint64_t(entry.expert);
    const projection_prologue x = projection_operands(projection_plain(ROUTED_ACT,
        SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED), SEISMIC_NORMALIZED_STRIDE_0, SEISMIC_NORMALIZED_STRIDE_1, hidden,
        entry.rows), ROUTED_OPERANDS);
    const projection_weights gate = routed_expert_matrix(slab_experts(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_GATE),
        SEISMIC_PTR(SEISMIC_BUFFER_TIER_GATE)), entry.source, PACKETS_W0_STACKED_ROWS, hidden);
    const projection_weights up = routed_expert_matrix(slab_experts(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP),
        SEISMIC_PTR(SEISMIC_BUFFER_TIER_UP)), entry.source, PACKETS_W1_STACKED_ROWS, hidden);
    const projection_epilogue out_ = projection_scaled(projection_gathered(projection_glu(ROUTED_ACT,
            ROUTED_GATE_UP_PRODUCT, 0ul, ROUTED_GATE_UP_PRODUCT_STRIDE_2,
            int(SEISMIC_PARAM_ACTIVATION)), entry.outputs),
        projection_scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_GATE_SCALE), SEISMIC_DIM_GS, SEISMIC_GATE_SCALE_STRIDE_0, expert),
        projection_scale_factor(SEISMIC_PTR(SEISMIC_BUFFER_UP_SCALE), SEISMIC_DIM_US, SEISMIC_UP_SCALE_STRIDE_0, expert));
    if (slab_source_none(entry.source)) {
        routed_zero_features(out_, entry.count, tile * rows_per_tile, rows_per_tile, uint(SEISMIC_DIM_F));
        return;
    }
    for (uint row_base = 0u; row_base < entry.count; row_base += maxm) {
        const uint slice_m = min(maxm, entry.count - row_base);
        projection_prologue slice_x = x;
        projection_epilogue slice_out = out_;
        if (row_base != 0u) {
            slice_x.rows += uint64_t(row_base) * 4ul;
            slice_out.outputs += uint64_t(row_base) * 4ul;
        }
        projection_gemv(PACKETS_W0, PACKETS_W1, true, SEISMIC_TUNE_ROWS, 32u, maxm, false, false,
            PROJECTION_NORM_NONE, slice_x, slice_out, gate, up, slice_m, uint(SEISMIC_DIM_F), hidden, tile);
        if (row_base + maxm < entry.count)
            barrier();
    }
}
