// stream_collapse (contract and portable body in streams.seismic): one block
// per row, each thread summing its columns' streams by the row's pre in
// stream order (`streams::collapse`).
#include "lib/core/streams.cuh"

constexpr unsigned STREAMS = SEISMIC_DIM_S;

extern "C" __global__ void stream_collapse(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *hidden = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN));
    const float *pre = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PRE));
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const unsigned row = blockIdx.x;
    streams::collapse<STREAMS>(hidden + (u64)row * SEISMIC_HIDDEN_STRIDE_0, SEISMIC_HIDDEN_STRIDE_1,
                               SEISMIC_HIDDEN_STRIDE_2, pre + (u64)row * SEISMIC_PRE_STRIDE_0, SEISMIC_PRE_STRIDE_1,
                               result + (u64)row * SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1,
                               SEISMIC_DIM_H, threadIdx.x, blockDim.x);
}
