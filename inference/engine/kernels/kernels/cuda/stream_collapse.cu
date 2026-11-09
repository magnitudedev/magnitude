// stream_collapse (contract and portable body in streams.seismic): one block
// per row, each thread summing its columns' streams by the row's pre in
// stream order.

constexpr unsigned STREAMS = SEISMIC_DIM_S;

extern "C" __global__ void stream_collapse(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *hidden = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN));
    const float *pre = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PRE));
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const unsigned row = blockIdx.x, h = SEISMIC_DIM_H;
    float carried[STREAMS];
#pragma unroll
    for (unsigned stream = 0; stream < STREAMS; ++stream)
        carried[stream] = pre[(u64)row * SEISMIC_PRE_STRIDE_0 + (u64)stream * SEISMIC_PRE_STRIDE_1];
    const float *x = hidden + (u64)row * SEISMIC_HIDDEN_STRIDE_0;
    for (unsigned column = threadIdx.x; column < h; column += blockDim.x) {
        float sum = 0.0f;
#pragma unroll
        for (unsigned stream = 0; stream < STREAMS; ++stream)
            sum += carried[stream] * x[(u64)stream * SEISMIC_HIDDEN_STRIDE_1 + (u64)column * SEISMIC_HIDDEN_STRIDE_2];
        result[(u64)row * SEISMIC_RESULT_0_STRIDE_0 + (u64)column * SEISMIC_RESULT_0_STRIDE_1] = sum;
    }
}
