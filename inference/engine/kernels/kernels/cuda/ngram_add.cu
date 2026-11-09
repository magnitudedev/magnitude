// ngram_add (contract and portable body in ngram.seismic): each stream's
// gated value row added into its streams. A thread owns one channel; the y
// grid visits the rows with a stride (it is capped at 65535).

extern "C" __global__ void ngram_add(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *hidden = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN));
    const float *scales = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALES));
    const float *value = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_VALUE));
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const u64 channel = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= SEISMIC_DIM_S * SEISMIC_DIM_H)
        return;
    const u64 stream = channel / SEISMIC_DIM_H;
    const u64 column = channel % SEISMIC_DIM_H;
    for (u64 row = blockIdx.y; row < SEISMIC_DIM_M; row += gridDim.y) {
        const float gated = __fmul_rn(scales[row * SEISMIC_SCALES_STRIDE_0 + stream * SEISMIC_SCALES_STRIDE_1],
                                      value[row * SEISMIC_VALUE_STRIDE_0 + column * SEISMIC_VALUE_STRIDE_1]);
        result[row * SEISMIC_RESULT_0_STRIDE_0 + channel * SEISMIC_RESULT_0_STRIDE_1] =
            __fadd_rn(hidden[row * SEISMIC_HIDDEN_STRIDE_0 + channel * SEISMIC_HIDDEN_STRIDE_1], gated);
    }
}
