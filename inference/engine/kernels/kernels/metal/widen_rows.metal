// widen_rows: widened[row, column] = f32(rows[row, column]); one thread per
// element on a (column, row) grid.
#include "lib/core/activation.h"

typedef element::Act activation;

kernel void widen_rows(
    device const uchar *rows [[buffer(SEISMIC_BUFFER_ROWS)]],
    device float *widened [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong column = ulong(position.x);
    const ulong row = ulong(position.y);
    if (column >= SEISMIC_DIM_D || row >= SEISMIC_DIM_M) return;
    widened[row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1] =
        element::at<activation>(rows, row * SEISMIC_ROWS_STRIDE_0 + column * SEISMIC_ROWS_STRIDE_1);
}
