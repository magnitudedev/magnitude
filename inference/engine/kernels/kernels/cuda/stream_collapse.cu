// stream_collapse (contract and portable body in streams.seismic): one block
// per row, each thread summing its columns' streams in stream order
// (`streams::collapse`), by the row's pre (P = 1) or at weight 1 and divided
// by S (P = 0, the mean).
#include "lib/core/streams.cuh"

constexpr unsigned STREAMS = SEISMIC_DIM_S;

extern "C" __global__ void stream_collapse(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const u64 row = blockIdx.x;
    const float *hidden =
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)) + row * SEISMIC_HIDDEN_STRIDE_0;
    float *out = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)) + row * SEISMIC_RESULT_0_STRIDE_0;
    if (SEISMIC_DIM_P == 1) {
        const float *pre = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PRE)) + row * SEISMIC_PRE_STRIDE_0;
        streams::collapse<STREAMS>(hidden, SEISMIC_HIDDEN_STRIDE_1, SEISMIC_HIDDEN_STRIDE_2, pre, SEISMIC_PRE_STRIDE_1,
                                   out, SEISMIC_RESULT_0_STRIDE_1, SEISMIC_DIM_H, threadIdx.x, blockDim.x);
    } else {
        float ones[STREAMS];
#pragma unroll
        for (unsigned stream = 0; stream < STREAMS; ++stream)
            ones[stream] = 1.0f;
        streams::collapse<STREAMS>(hidden, SEISMIC_HIDDEN_STRIDE_1, SEISMIC_HIDDEN_STRIDE_2, ones, out,
                                   SEISMIC_RESULT_0_STRIDE_1, SEISMIC_DIM_H, threadIdx.x, blockDim.x);
        // Each thread divides the columns it wrote.
        for (unsigned column = threadIdx.x; column < SEISMIC_DIM_H; column += blockDim.x)
            out[(u64)column * SEISMIC_RESULT_0_STRIDE_1] =
                __fdiv_rn(out[(u64)column * SEISMIC_RESULT_0_STRIDE_1], float(STREAMS));
    }
}
