// tap_rows: taps[row, index[0] * D + column] = A(residual[row, column]); four
// elements per thread on a (column, row) grid.
#include "lib/core/activation.cuh"

extern "C" __global__ void tap_rows(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *residual = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    const int *tap_index = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_INDEX));
    const auto taps = SEISMIC_OUT_TAPS;
    const u64 width = SEISMIC_DIM_D;
    const u64 column = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const u64 row = (u64)blockIdx.y;
    if (column >= width || row >= SEISMIC_DIM_M)
        return;
    const u64 tap = (u64)tap_index[0];
    if (column + 4 <= width && SEISMIC_RESIDUAL_STRIDE_1 == 1 && SEISMIC_TAPS_STRIDE_1 == 1) {
        float values[4];
        activation::load4_f32_aligned(residual + row * SEISMIC_RESIDUAL_STRIDE_0 + column, values);
        activation::put4_aligned(taps, row * SEISMIC_TAPS_STRIDE_0 + tap * width + column, values);
        return;
    }
    const u64 src_row = row * SEISMIC_RESIDUAL_STRIDE_0;
    const u64 dst_row = row * SEISMIC_TAPS_STRIDE_0;
    const u64 tap_base = tap * width;
    const u64 limit = (column + 4 < width) ? (column + 4) : width;
    for (u64 c = column; c < limit; ++c) {
        taps.put(dst_row + (tap_base + c) * SEISMIC_TAPS_STRIDE_1,
                 residual[src_row + c * SEISMIC_RESIDUAL_STRIDE_1]);
    }
}
