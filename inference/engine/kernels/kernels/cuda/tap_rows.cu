// tap_rows: taps[row, index[0] * D + column] = A(residual[row, column]); one
// thread per copied element.
#include "lib/core/activation.cuh"

extern "C" __global__ void tap_rows(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *residual = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    const int *tap_index = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_INDEX));
    element::u8 *taps = SEISMIC_PTR(SEISMIC_BUFFER_TAPS);
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 width = SEISMIC_DIM_D;
    if (index >= (u64)SEISMIC_DIM_M * width)
        return;
    const u64 row = index / width;
    const u64 column = index % width;
    const u64 tap = (u64)tap_index[0];
    element::put<element::Act>(taps, row * SEISMIC_TAPS_STRIDE_0 + (tap * width + column) * SEISMIC_TAPS_STRIDE_1,
                               residual[row * SEISMIC_RESIDUAL_STRIDE_0 + column * SEISMIC_RESIDUAL_STRIDE_1]);
}
