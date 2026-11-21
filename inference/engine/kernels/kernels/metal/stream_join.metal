// stream_join (contract and portable body in streams.seismic): four columns
// of a row per thread, joining the F32 output into the streams through the
// output entries' epilogue (`projection::OutputOf`: the residual add, or the
// `Scatter` with the inject values and, when the inject row carries one,
// their comb).
#include "lib/projection/projection.h"

typedef projection::OutputOf<element::F32, projection::AllRows, SEISMIC_DIM_SN, SEISMIC_DIM_IN != 0,
    (SEISMIC_DIM_IN > SEISMIC_DIM_SN)> output_join;

kernel void stream_join(device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const float *y [[buffer(SEISMIC_BUFFER_Y)]],
    device const float *inject [[buffer(SEISMIC_BUFFER_INJECT)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    const uint h = uint(SEISMIC_DIM_H);
    const uint vecs_per_row = (h + 3u) / 4u;
    if (index >= uint(SEISMIC_DIM_M) * vecs_per_row)
        return;
    const uint row = index / vecs_per_row;
    const uint column = (index % vecs_per_row) * 4u;
    const auto out = output_join::make(element::Out<element::F32>(result_cells), SEISMIC_RESULT_0_STRIDE_0,
        SEISMIC_RESULT_0_STRIDE_1, residual, SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, {}, inject,
        SEISMIC_INJECT_STRIDE_0, h).row(row);
    device const float *y_row = y + ulong(row) * SEISMIC_Y_STRIDE_0;
    if (column + 4u <= h) {
        float4 values;
        if (SEISMIC_Y_STRIDE_1 == 1) {
            values = float4(*reinterpret_cast<device const packed_float4 *>(y_row + column));
        } else {
            values = float4(y_row[ulong(column) * SEISMIC_Y_STRIDE_1],
                y_row[ulong(column + 1u) * SEISMIC_Y_STRIDE_1],
                y_row[ulong(column + 2u) * SEISMIC_Y_STRIDE_1],
                y_row[ulong(column + 3u) * SEISMIC_Y_STRIDE_1]);
        }
        out.store4(column, values);
    } else {
        for (uint c = column; c < h; ++c)
            out.store(c, y_row[ulong(c) * SEISMIC_Y_STRIDE_1]);
    }
}
