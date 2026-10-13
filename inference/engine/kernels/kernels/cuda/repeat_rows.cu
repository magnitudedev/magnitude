// repeat_rows: S scaled copies of each row, one thread per value.

extern "C" __global__ void repeat_rows(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *rows = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROWS));
    const float *scales = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALES));
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 h = SEISMIC_DIM_H, width = SEISMIC_DIM_S * h;
    if (index >= (u64)SEISMIC_DIM_M * width)
        return;
    const u64 row = index / width, stream = index % width / h, column = index % h;
    result[row * SEISMIC_RESULT_0_STRIDE_0 + (stream * h + column) * SEISMIC_RESULT_0_STRIDE_1] =
        rows[row * SEISMIC_ROWS_STRIDE_0 + column * SEISMIC_ROWS_STRIDE_1] * scales[stream * SEISMIC_SCALES_STRIDE_0];
}
