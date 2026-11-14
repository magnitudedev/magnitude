// stream_collapse (contract and portable body in streams.seismic): one
// threadgroup per row, each thread summing its columns' streams in stream
// order (`streams::collapse`), by the row's pre (P = 1) or at weight 1 and
// divided by S (P = 0, the mean).
#include "lib/core/streams.h"

constant constexpr uint STREAMS = uint(SEISMIC_DIM_S);

kernel void stream_collapse(device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const float *pre [[buffer(SEISMIC_BUFFER_PRE)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    device const float *streams = hidden + ulong(row) * SEISMIC_HIDDEN_STRIDE_0;
    device float *out = result + ulong(row) * SEISMIC_RESULT_0_STRIDE_0;
    if (SEISMIC_DIM_P == 1) {
        streams::collapse<STREAMS>(streams, SEISMIC_HIDDEN_STRIDE_1, SEISMIC_HIDDEN_STRIDE_2,
            pre + ulong(row) * SEISMIC_PRE_STRIDE_0, SEISMIC_PRE_STRIDE_1, out, SEISMIC_RESULT_0_STRIDE_1,
            uint(SEISMIC_DIM_H), thread_index, threads);
    } else {
        float ones[STREAMS];
        for (uint stream = 0; stream < STREAMS; ++stream)
            ones[stream] = 1.0f;
        streams::collapse<STREAMS>(streams, SEISMIC_HIDDEN_STRIDE_1, SEISMIC_HIDDEN_STRIDE_2, ones, out,
            SEISMIC_RESULT_0_STRIDE_1, uint(SEISMIC_DIM_H), thread_index, threads);
        // Each thread divides the columns it wrote.
        for (uint column = thread_index; column < uint(SEISMIC_DIM_H); column += threads)
            out[ulong(column) * SEISMIC_RESULT_0_STRIDE_1] /= float(STREAMS);
    }
}
