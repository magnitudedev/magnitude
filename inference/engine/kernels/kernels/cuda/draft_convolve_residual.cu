// draft_convolve_residual: DFlash2's half-1 grouped dynamic causal convolution
// of a sublayer's F32 output rows within each draft block, plus the residual;
// one thread per output element, summing its taps in offset order.

extern "C" __global__ void draft_convolve_residual(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *residual = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    const float *output = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_OUTPUT));
    const float *dynamic = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_DYNAMIC));
    const float *base = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BASE));
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 channels = SEISMIC_DIM_C, width = SEISMIC_DIM_G * channels;
    if (index >= (u64)SEISMIC_DIM_M * width)
        return;
    const u64 row = index / width, column = index % width, group = column / channels;
    const u64 position = row % (u64)SEISMIC_PARAM_BLOCK;
    float sum = 0.0f;
    for (u64 offset = 0; offset < SEISMIC_DIM_K && offset <= position; ++offset) {
        const float coefficient =
            base[SEISMIC_BASE_STRIDE_0 + offset * SEISMIC_BASE_STRIDE_1 + column * SEISMIC_BASE_STRIDE_2] +
            dynamic[row * SEISMIC_DYNAMIC_STRIDE_0 + SEISMIC_DYNAMIC_STRIDE_1 + offset * SEISMIC_DYNAMIC_STRIDE_2 +
                    group * SEISMIC_DYNAMIC_STRIDE_3];
        sum = seismic_fma_rn(coefficient, output[(row - offset) * SEISMIC_OUTPUT_STRIDE_0 + column * SEISMIC_OUTPUT_STRIDE_1], sum);
    }
    result[row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1] =
        residual[row * SEISMIC_RESIDUAL_STRIDE_0 + column * SEISMIC_RESIDUAL_STRIDE_1] + sum;
}
