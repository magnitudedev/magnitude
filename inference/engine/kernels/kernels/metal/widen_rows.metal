// widen_rows: widened[row, column] = f32(rows[row, column]); four elements per
// thread on a (column, row) grid.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"

typedef activation::Act activation_element;

kernel void widen_rows(
    device const uchar *rows [[buffer(SEISMIC_BUFFER_ROWS)]],
    SEISMIC_RESULT_0_PARAM(widened),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong width = SEISMIC_DIM_D;
    const ulong column = ulong(position.x) * 4;
    const ulong row = ulong(position.y);
    if (column >= width || row >= SEISMIC_DIM_M) return;
    const auto out = element::Out<ELEMENT_OF(SEISMIC_RESULT_0)>(widened);
    if (column + 4 <= width && SEISMIC_ROWS_STRIDE_1 == 1 && SEISMIC_RESULT_0_STRIDE_1 == 1) {
        const auto src = reinterpret_cast<device const activation_element::storage *>(rows)
            + row * SEISMIC_ROWS_STRIDE_0 + column;
        const float4 values = activation_element::load4(src);
        out.put4(row * SEISMIC_RESULT_0_STRIDE_0 + column, values);
        return;
    }
    const ulong src_row = row * SEISMIC_ROWS_STRIDE_0;
    const ulong dst_row = row * SEISMIC_RESULT_0_STRIDE_0;
    const ulong limit = (column + 4 < width) ? (column + 4) : width;
    for (ulong c = column; c < limit; ++c) {
        out.put(dst_row + c * SEISMIC_RESULT_0_STRIDE_1,
                element::at<activation_element>(rows, src_row + c * SEISMIC_ROWS_STRIDE_1));
    }
}
