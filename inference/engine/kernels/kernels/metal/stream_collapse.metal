// stream_collapse (contract and portable body in streams.seismic): one
// threadgroup per row, each thread summing its columns' streams by the row's
// pre in stream order (`streams::collapse`).
#include "lib/core/streams.h"

constant constexpr uint STREAMS = uint(SEISMIC_DIM_S);

kernel void stream_collapse(device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const float *pre [[buffer(SEISMIC_BUFFER_PRE)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    streams::collapse<STREAMS>(hidden + ulong(row) * SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1,
        SEISMIC_HIDDEN_STRIDE_2, pre + ulong(row) * SEISMIC_PRE_STRIDE_0, SEISMIC_PRE_STRIDE_1,
        result + ulong(row) * SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, uint(SEISMIC_DIM_H), thread_index,
        threads);
}
