// tap_rows: taps[row, index[0] * D + column] = A(residual[row, column]); one
// thread per copied element on a (column, row) grid, so no thread divides.
#include "lib/core/activation.h"

typedef element::Act activation;

kernel void tap_rows(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const int *tap_index [[buffer(SEISMIC_BUFFER_INDEX)]],
    device uchar *taps [[buffer(SEISMIC_BUFFER_TAPS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong width = SEISMIC_DIM_D;
    const ulong column = ulong(position.x);
    const ulong row = ulong(position.y);
    if (column >= width || row >= SEISMIC_DIM_M) return;
    const ulong tap = ulong(tap_index[0]);
    element::put<activation>(taps, row * SEISMIC_TAPS_STRIDE_0 + (tap * width + column) * SEISMIC_TAPS_STRIDE_1,
        residual[row * SEISMIC_RESIDUAL_STRIDE_0 + column * SEISMIC_RESIDUAL_STRIDE_1]);
}
