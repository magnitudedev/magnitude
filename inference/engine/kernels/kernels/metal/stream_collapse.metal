// stream_collapse (contract and portable body in streams.seismic): one
// threadgroup per row, each thread summing its columns' streams by the row's
// pre in stream order.

constant constexpr uint STREAMS = uint(SEISMIC_DIM_S);

kernel void stream_collapse(device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const float *pre [[buffer(SEISMIC_BUFFER_PRE)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    const uint h = uint(SEISMIC_DIM_H);
    float carried[STREAMS];
    for (uint stream = 0; stream < STREAMS; ++stream)
        carried[stream] = pre[ulong(row) * SEISMIC_PRE_STRIDE_0 + ulong(stream) * SEISMIC_PRE_STRIDE_1];
    device const float *x = hidden + ulong(row) * SEISMIC_HIDDEN_STRIDE_0;
    for (uint column = thread_index; column < h; column += threads) {
        float sum = 0.0f;
        for (uint stream = 0; stream < STREAMS; ++stream)
            sum += carried[stream] * x[ulong(stream) * SEISMIC_HIDDEN_STRIDE_1 + ulong(column) * SEISMIC_HIDDEN_STRIDE_2];
        result[ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + ulong(column) * SEISMIC_RESULT_0_STRIDE_1] = sum;
    }
}
