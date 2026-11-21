// tap_rows: taps[row, index[0] * D + column] = A(residual[row, column]); four
// elements per thread on a (column, row) grid, so no thread divides.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"

typedef activation::Act activation_element;

kernel void tap_rows(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const int *tap_index [[buffer(SEISMIC_BUFFER_INDEX)]],
    SEISMIC_OUT_TAPS_PARAM(taps),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong width = SEISMIC_DIM_D;
    const ulong column = ulong(position.x) * 4;
    const ulong row = ulong(position.y);
    if (column >= width || row >= SEISMIC_DIM_M) return;
    const ulong tap = ulong(tap_index[0]);
    const auto out = element::Out<activation_element>(taps);
    if (column + 4 <= width && SEISMIC_RESIDUAL_STRIDE_1 == 1 && SEISMIC_TAPS_STRIDE_1 == 1) {
        const float4 values = element::F32::load4(residual + row * SEISMIC_RESIDUAL_STRIDE_0 + column);
        out.put4(row * SEISMIC_TAPS_STRIDE_0 + tap * width + column, values);
        return;
    }
    const ulong src_row = row * SEISMIC_RESIDUAL_STRIDE_0;
    const ulong dst_row = row * SEISMIC_TAPS_STRIDE_0;
    const ulong tap_base = tap * width;
    const ulong limit = (column + 4 < width) ? (column + 4) : width;
    for (ulong c = column; c < limit; ++c) {
        out.put(dst_row + (tap_base + c) * SEISMIC_TAPS_STRIDE_1,
                residual[src_row + c * SEISMIC_RESIDUAL_STRIDE_1]);
    }
}
