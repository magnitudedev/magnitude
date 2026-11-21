// widen_rows: widened[row, column] = f32(rows[row, column]); four elements per
// thread on a (column, row) grid.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"

extern "C" __global__ void widen_rows(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const element::u8 *rows = SEISMIC_PTR(SEISMIC_BUFFER_ROWS);
    const auto widened = SEISMIC_RESULT_0_OUT;
    const u64 width = SEISMIC_DIM_D;
    const u64 column = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const u64 row = (u64)blockIdx.y;
    if (column >= width || row >= SEISMIC_DIM_M)
        return;
    if (column + 4 <= width && SEISMIC_ROWS_STRIDE_1 == 1 && SEISMIC_RESULT_0_STRIDE_1 == 1) {
        const u64 src = row * SEISMIC_ROWS_STRIDE_0 + column;
        const u64 dst = row * SEISMIC_RESULT_0_STRIDE_0 + column;
        constexpr u64 src_align_mask = 4u * activation::Act::bytes - 1u;
        float values[4];
        if ((reinterpret_cast<u64>(rows + src * activation::Act::bytes) & src_align_mask) == 0) {
            element::span<activation::Act, 4, true>(rows, src, values);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                values[i] = element::at<activation::Act>(rows, src + i);
        }
        activation::put4_aligned(widened, dst, values);
        return;
    }
    const u64 src_row = row * SEISMIC_ROWS_STRIDE_0;
    const u64 dst_row = row * SEISMIC_RESULT_0_STRIDE_0;
    const u64 limit = (column + 4 < width) ? (column + 4) : width;
    for (u64 c = column; c < limit; ++c) {
        widened.put(dst_row + c * SEISMIC_RESULT_0_STRIDE_1,
                    element::at<activation::Act>(rows, src_row + c * SEISMIC_ROWS_STRIDE_1));
    }
}
