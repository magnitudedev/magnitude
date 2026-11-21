// feature_rows: features[row, column] = A(fused[out_rows[row], column]); four
// elements per thread on a (column, row) grid, so no thread divides.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"

typedef activation::Act activation_element;

kernel void feature_rows(
    device const float *fused [[buffer(SEISMIC_BUFFER_FUSED)]],
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],
    SEISMIC_RESULT_0_PARAM(features),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong width = SEISMIC_DIM_D;
    const ulong column = ulong(position.x) * 4;
    const ulong row = ulong(position.y);
    if (column >= width || row >= SEISMIC_DIM_O) return;
    const ulong source = ulong(out_rows[row * SEISMIC_OUT_ROWS_STRIDE_0]);
    const auto out = element::Out<activation_element>(features);
    if (column + 4 <= width && SEISMIC_FUSED_STRIDE_1 == 1 && SEISMIC_RESULT_0_STRIDE_1 == 1) {
        const float4 values = element::F32::load4(fused + source * SEISMIC_FUSED_STRIDE_0 + column);
        out.put4(row * SEISMIC_RESULT_0_STRIDE_0 + column, values);
        return;
    }
    const ulong src_row = source * SEISMIC_FUSED_STRIDE_0;
    const ulong dst_row = row * SEISMIC_RESULT_0_STRIDE_0;
    const ulong limit = (column + 4 < width) ? (column + 4) : width;
    for (ulong c = column; c < limit; ++c) {
        out.put(dst_row + c * SEISMIC_RESULT_0_STRIDE_1,
                fused[src_row + c * SEISMIC_FUSED_STRIDE_1]);
    }
}
