// feature_rows: features[row, column] = A(fused[out_rows[row], column]); four
// elements per thread on a (column, row) grid.
#include "lib/core/activation.cuh"

extern "C" __global__ void feature_rows(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *fused = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_FUSED));
    const int *out_rows = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS));
    const auto features = SEISMIC_RESULT_0_OUT;
    const u64 width = SEISMIC_DIM_D;
    const u64 column = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const u64 row = (u64)blockIdx.y;
    if (column >= width || row >= SEISMIC_DIM_O)
        return;
    const u64 source = (u64)out_rows[row * SEISMIC_OUT_ROWS_STRIDE_0];
    if (column + 4 <= width && SEISMIC_FUSED_STRIDE_1 == 1 && SEISMIC_RESULT_0_STRIDE_1 == 1) {
        float values[4];
        activation::load4_f32_aligned(fused + source * SEISMIC_FUSED_STRIDE_0 + column, values);
        activation::put4_aligned(features, row * SEISMIC_RESULT_0_STRIDE_0 + column, values);
        return;
    }
    const u64 src_row = source * SEISMIC_FUSED_STRIDE_0;
    const u64 dst_row = row * SEISMIC_RESULT_0_STRIDE_0;
    const u64 limit = (column + 4 < width) ? (column + 4) : width;
    for (u64 c = column; c < limit; ++c) {
        features.put(dst_row + c * SEISMIC_RESULT_0_STRIDE_1,
                     fused[src_row + c * SEISMIC_FUSED_STRIDE_1]);
    }
}
