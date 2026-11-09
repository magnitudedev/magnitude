// ngram_add (contract and portable body in ngram.seismic): each stream's
// gated value row added into its streams. A thread owns one channel of one
// row.

kernel void ngram_add(
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const float *scales [[buffer(SEISMIC_BUFFER_SCALES)]],
    device const float *value [[buffer(SEISMIC_BUFFER_VALUE)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    const ulong channel = ulong(group.x) * threadgroup_shape.x + thread_index;
    if (channel >= SEISMIC_DIM_S * SEISMIC_DIM_H)
        return;
    const ulong row = group.y;
    const ulong stream = channel / SEISMIC_DIM_H;
    const ulong column = channel % SEISMIC_DIM_H;
    const float gate = scales[row * SEISMIC_SCALES_STRIDE_0 + stream * SEISMIC_SCALES_STRIDE_1];
    result[row * SEISMIC_RESULT_0_STRIDE_0 + channel * SEISMIC_RESULT_0_STRIDE_1] =
        hidden[row * SEISMIC_HIDDEN_STRIDE_0 + channel * SEISMIC_HIDDEN_STRIDE_1]
        + gate * value[row * SEISMIC_VALUE_STRIDE_0 + column * SEISMIC_VALUE_STRIDE_1];
}
