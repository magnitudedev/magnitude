// repeat_rows: S scaled copies of each row, four columns across all S streams
// per thread (loads each source quad once and scales into every stream).
#include <seismic/element.h>

kernel void repeat_rows(device const float *rows [[buffer(SEISMIC_BUFFER_ROWS)]],
    device const float *scales [[buffer(SEISMIC_BUFFER_SCALES)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    const uint h = uint(SEISMIC_DIM_H);
    const uint streams = uint(SEISMIC_DIM_S);
    const uint vecs_per_row = (h + 3u) / 4u;
    if (ulong(index) >= ulong(SEISMIC_DIM_M) * vecs_per_row)
        return;
    const uint row = index / vecs_per_row;
    const uint column = (index % vecs_per_row) * 4u;
    device const float *src_row = rows + ulong(row) * SEISMIC_ROWS_STRIDE_0;
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> dst_row =
        element::Out<ELEMENT_OF(SEISMIC_RESULT_0)>(result_cells).offset(ulong(row) * SEISMIC_RESULT_0_STRIDE_0);
    if (column + 4u <= h) {
        float4 values;
        if (SEISMIC_ROWS_STRIDE_1 == 1) {
            values = float4(*reinterpret_cast<device const packed_float4 *>(src_row + column));
        } else {
            values = float4(src_row[ulong(column) * SEISMIC_ROWS_STRIDE_1],
                src_row[ulong(column + 1u) * SEISMIC_ROWS_STRIDE_1],
                src_row[ulong(column + 2u) * SEISMIC_ROWS_STRIDE_1],
                src_row[ulong(column + 3u) * SEISMIC_ROWS_STRIDE_1]);
        }
        if (SEISMIC_RESULT_0_STRIDE_1 == 1) {
            for (uint s = 0; s < streams; ++s) {
                const float scale = scales[ulong(s) * SEISMIC_SCALES_STRIDE_0];
                dst_row.put4(ulong(s) * h + column, values * scale);
            }
        } else {
            for (uint s = 0; s < streams; ++s) {
                const float scale = scales[ulong(s) * SEISMIC_SCALES_STRIDE_0];
                const ulong offset = ulong(s) * h + column;
                dst_row.put(offset * SEISMIC_RESULT_0_STRIDE_1, values.x * scale);
                dst_row.put((offset + 1u) * SEISMIC_RESULT_0_STRIDE_1, values.y * scale);
                dst_row.put((offset + 2u) * SEISMIC_RESULT_0_STRIDE_1, values.z * scale);
                dst_row.put((offset + 3u) * SEISMIC_RESULT_0_STRIDE_1, values.w * scale);
            }
        }
    } else {
        for (uint c = column; c < h; ++c) {
            const float value = src_row[ulong(c) * SEISMIC_ROWS_STRIDE_1];
            for (uint s = 0; s < streams; ++s) {
                const float scale = scales[ulong(s) * SEISMIC_SCALES_STRIDE_0];
                dst_row.put((ulong(s) * h + c) * SEISMIC_RESULT_0_STRIDE_1, value * scale);
            }
        }
    }
}
