kernel void slab_scale(
    device const ulong *x [[buffer(SEISMIC_BUFFER_X)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    const ulong rows = ulong(SEISMIC_PARAM_SLAB_ROWS);
    const ulong slab = ulong(row) / rows;
    const ulong local = ulong(row) % rows;
    device const float *source = reinterpret_cast<device const float *>(x[slab]);
    const float factor = as_type<float>(uint(SEISMIC_PARAM_FACTOR));
    for (ulong column = lane; column < SEISMIC_DIM_N; column += lanes) {
        result[ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1] =
            source[local * SEISMIC_DIM_N + column] * factor;
    }
}
