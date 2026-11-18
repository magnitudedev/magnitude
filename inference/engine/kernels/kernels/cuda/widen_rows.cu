// widen_rows: widened[row, column] = f32(rows[row, column]); one thread per
// element.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"

extern "C" __global__ void widen_rows(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const element::u8 *rows = SEISMIC_PTR(SEISMIC_BUFFER_ROWS);
    const auto widened = SEISMIC_RESULT_0_OUT;
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 width = SEISMIC_DIM_D;
    if (index >= (u64)SEISMIC_DIM_M * width)
        return;
    const u64 row = index / width, column = index % width;
    widened.put(row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1,
                element::at<activation::Act>(rows, row * SEISMIC_ROWS_STRIDE_0 + column * SEISMIC_ROWS_STRIDE_1));
}
