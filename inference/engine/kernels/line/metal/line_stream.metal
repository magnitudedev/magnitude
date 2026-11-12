// line_stream: a simdgroup sums a row of C values, each lane UNROLL 16-byte
// loads in flight at once (C is a multiple of 32 lanes x 4 values x 16).
kernel void line_stream(
    device const float *input [[buffer(SEISMIC_BUFFER_INPUT)]],
    device float *out [[buffer(SEISMIC_BUFFER_OUT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint UNROLL = SEISMIC_TUNE_UNROLL;
    const ulong row = ulong(group) * simdgroups + sg;
    if (row >= SEISMIC_DIM_R)
        return;
    device const float4 *line = (device const float4 *)(input + row * SEISMIC_INPUT_STRIDE_0);
    const ulong vectors = SEISMIC_DIM_C / 4;
    float partial[UNROLL];
    for (uint k = 0; k < UNROLL; ++k)
        partial[k] = 0.0f;
    for (ulong at = lane; at < vectors; at += 32 * UNROLL) {
        float4 loaded[UNROLL];
        for (uint k = 0; k < UNROLL; ++k)
            loaded[k] = line[at + 32 * k];
        for (uint k = 0; k < UNROLL; ++k)
            partial[k] += (loaded[k].x + loaded[k].y) + (loaded[k].z + loaded[k].w);
    }
    float total = 0.0f;
    for (uint k = 0; k < UNROLL; ++k)
        total += partial[k];
    total = simd_sum(total);
    if (lane == 0)
        out[row * SEISMIC_OUT_STRIDE_0] = total;
}
