// vision_clamp: four contiguous columns per thread on a 2-D (column, row)
// grid, A(clamp(x, minimum, maximum)).
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"

using element::u64;
using element::u8;
typedef activation::Act A;

extern "C" __global__ void vision_clamp(SEISMIC_KERNEL_PARAMS) {
    const u64 columns = SEISMIC_DIM_N;
    const u64 col = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 4ull;
    if (col >= columns)
        return;
    const float minimum = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_MINIMUM))[0];
    const float maximum = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_MAXIMUM))[0];
    const u8 *x = SEISMIC_PTR(SEISMIC_BUFFER_X);
    const auto result = SEISMIC_RESULT_0_OUT;
    constexpr u64 align_mask = 4ull * A::bytes - 1ull;
    const bool full_vec = col + 4ull <= columns &&
                          SEISMIC_X_STRIDE_1 == 1ull &&
                          SEISMIC_RESULT_0_STRIDE_1 == 1ull;
    const u64 limit = col + 4ull < columns ? col + 4ull : columns;
    for (u64 row = blockIdx.y; row < SEISMIC_DIM_M; row += gridDim.y) {
        const u64 x_row = row * SEISMIC_X_STRIDE_0;
        const u64 out_row = row * SEISMIC_RESULT_0_STRIDE_0;
        const bool aligned = ((reinterpret_cast<u64>(x + x_row * A::bytes) |
                               reinterpret_cast<u64>(result.stored(out_row))) & align_mask) == 0ull;
        if (full_vec && aligned) {
            float xv[4];
            element::span<A, 4, false>(x, x_row + col, xv);
            float out[4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
                out[k] = fminf(fmaxf(xv[k], minimum), maximum);
            result.put_span<4>(out_row + col, out);
        } else {
            for (u64 c = col; c < limit; ++c) {
                const float value = element::at<A>(x, x_row + c * SEISMIC_X_STRIDE_1);
                result.put(out_row + c * SEISMIC_RESULT_0_STRIDE_1, fminf(fmaxf(value, minimum), maximum));
            }
        }
    }
}
