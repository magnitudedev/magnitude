// feature_rows: features[row, column] = A(fused[out_rows[row], column]); one
// thread per copied element on a (column, row) grid, so no thread divides.
#include "lib/core/activation.h"

typedef element::Act activation;

kernel void feature_rows(
    device const float *fused [[buffer(SEISMIC_BUFFER_FUSED)]],
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],
    device uchar *features [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong column = ulong(position.x);
    const ulong row = ulong(position.y);
    if (column >= SEISMIC_DIM_D || row >= SEISMIC_DIM_O) return;
    const ulong source = ulong(out_rows[row * SEISMIC_OUT_ROWS_STRIDE_0]);
    element::put<activation>(features, row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1,
        fused[source * SEISMIC_FUSED_STRIDE_0 + column * SEISMIC_FUSED_STRIDE_1]);
}
