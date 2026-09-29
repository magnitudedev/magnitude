// feature_rows: features[row, column] = A(fused[out_rows[row], column]); one
// thread per copied element.
#include "lib/core/activation.cuh"

extern "C" __global__ void feature_rows(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *fused = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_FUSED));
    const int *out_rows = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS));
    element::u8 *features = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 width = SEISMIC_DIM_D;
    if (index >= (u64)SEISMIC_DIM_O * width)
        return;
    const u64 row = index / width;
    const u64 column = index % width;
    const u64 source = (u64)out_rows[row * SEISMIC_OUT_ROWS_STRIDE_0];
    element::put<element::Act>(features, row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1,
                               fused[source * SEISMIC_FUSED_STRIDE_0 + column * SEISMIC_FUSED_STRIDE_1]);
}
