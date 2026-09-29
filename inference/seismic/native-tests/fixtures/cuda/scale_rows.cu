extern "C" __global__ void scale_rows(SEISMIC_KERNEL_PARAMS) {
    const float *x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const float factor = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_FACTOR));
    for (unsigned local = 0; local < SEISMIC_TUNE_ROWS; ++local) {
        const unsigned long long row =
            static_cast<unsigned long long>(blockIdx.x) * SEISMIC_TUNE_ROWS + local;
        if (row >= SEISMIC_DIM_M) {
            return;
        }
        for (unsigned long long column = threadIdx.x; column < SEISMIC_DIM_N;
             column += blockDim.x) {
            result[row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1] =
                x[row * SEISMIC_X_STRIDE_0 + column * SEISMIC_X_STRIDE_1] * factor;
        }
    }
}
