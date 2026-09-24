// qwen_conditioning_overlay: out[row, column] = input[row, column] (F32);
// one thread per element.

extern "C" __global__ void qwen_conditioning_overlay(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *input = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INPUT));
    float *out = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT));
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (u64)SEISMIC_DIM_M * SEISMIC_DIM_D)
        return;
    const u64 row = index / SEISMIC_DIM_D;
    const u64 column = index % SEISMIC_DIM_D;
    out[row * SEISMIC_OUT_STRIDE_0 + column * SEISMIC_OUT_STRIDE_1] =
        input[row * SEISMIC_INPUT_STRIDE_0 + column * SEISMIC_INPUT_STRIDE_1];
}
