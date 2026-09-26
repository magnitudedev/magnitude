// One warp per 32-lane row.

extern "C" __global__ void ptx_shuffle(SEISMIC_KERNEL_PARAMS) {
    const float *x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const unsigned long long row = blockIdx.x;
    const unsigned lane = threadIdx.x;
    const float value = x[row * SEISMIC_X_STRIDE_0 + lane * SEISMIC_X_STRIDE_1];
    const float outputs[5] = {
        seismic_shfl_idx_f32(value, 31 - lane),
        seismic_shfl_down_f32(value, 3),
        seismic_shfl_up_f32(value, 3),
        seismic_warp_sum_f32(value),
        seismic_warp_max_f32(value),
    };
    for (unsigned plane = 0; plane < 5; ++plane) {
        result[plane * SEISMIC_RESULT_0_STRIDE_0 + row * SEISMIC_RESULT_0_STRIDE_1 +
               lane * SEISMIC_RESULT_0_STRIDE_2] = outputs[plane];
    }
}
