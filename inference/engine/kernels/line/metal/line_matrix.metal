// line_matrix: 8x8 simdgroup matrix multiply-adds of half into f32, the
// product of the projection GEMMs' simdgroup tiles: S multiply-adds a lane
// (a product is 16 a lane), in UNROLL independent accumulators seeded
// apart, the operands each simdgroup's first `x` in half; it writes `x`
// back (line.seismic). A simdgroup runs whole or not at all.
kernel void line_matrix(
    device const float *xs [[buffer(SEISMIC_BUFFER_X)]],
    device float *out [[buffer(SEISMIC_BUFFER_OUT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]],
    uint lane_in [[thread_index_in_simdgroup]]) {
    constexpr uint UNROLL = SEISMIC_TUNE_UNROLL;
    constexpr uint PRODUCTS = SEISMIC_STATIC_S / 16;
    const ulong lane = ulong(index);
    const ulong first = lane - lane_in;
    if (first >= SEISMIC_DIM_N)
        return;
    const half value = half(xs[first * SEISMIC_X_STRIDE_0]);
    const simdgroup_half8x8 a = make_filled_simdgroup_matrix<half, 8, 8>(value);
    const simdgroup_half8x8 b = make_filled_simdgroup_matrix<half, 8, 8>(value);
    simdgroup_float8x8 d[UNROLL];
    for (uint k = 0; k < UNROLL; ++k)
        d[k] = make_filled_simdgroup_matrix<float, 8, 8>(float(k));
    for (uint step = 0; step < PRODUCTS / UNROLL; ++step) {
        for (uint k = 0; k < UNROLL; ++k)
            simdgroup_multiply_accumulate(d[k], a, b, d[k]);
    }
    float total = 0.0f;
    for (uint k = 0; k < UNROLL; ++k)
        total += d[k].thread_elements()[0] + d[k].thread_elements()[1];
    if (lane < SEISMIC_DIM_N)
        out[lane * SEISMIC_OUT_STRIDE_0] =
            xs[lane * SEISMIC_X_STRIDE_0] + (as_type<uint>(total) == 0xFFFFFFFFu ? 1.0f : 0.0f);
}
