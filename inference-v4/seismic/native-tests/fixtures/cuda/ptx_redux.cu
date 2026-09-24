// One warp per 32-lane row; lane 0 publishes the row's sum, minimum and
// maximum.

extern "C" __global__ void ptx_redux_s32(SEISMIC_KERNEL_PARAMS) {
    const int *x = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    int *result = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const unsigned long long row = blockIdx.x;
    const int value = x[row * SEISMIC_X_STRIDE_0 + threadIdx.x * SEISMIC_X_STRIDE_1];
    const int outputs[3] = {seismic_redux_add_s32(value), seismic_redux_min_s32(value),
                            seismic_redux_max_s32(value)};
    if (threadIdx.x == 0) {
        for (unsigned column = 0; column < 3; ++column) {
            result[row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1] =
                outputs[column];
        }
    }
}

extern "C" __global__ void ptx_redux_u32(SEISMIC_KERNEL_PARAMS) {
    const unsigned *x = reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    unsigned *result = reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const unsigned long long row = blockIdx.x;
    const unsigned value = x[row * SEISMIC_X_STRIDE_0 + threadIdx.x * SEISMIC_X_STRIDE_1];
    const unsigned outputs[3] = {seismic_redux_add_u32(value), seismic_redux_min_u32(value),
                                 seismic_redux_max_u32(value)};
    if (threadIdx.x == 0) {
        for (unsigned column = 0; column < 3; ++column) {
            result[row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1] =
                outputs[column];
        }
    }
}
