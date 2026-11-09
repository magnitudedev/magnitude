// stream_join (contract and portable body in streams.seismic): one thread
// per column of a row, joining the F32 output into the streams through the
// output entries' epilogue (`projection::OutputOf`: the residual add, or the
// `Scatter` with the inject values and, when the inject row carries one,
// their comb).
#include "lib/projection/projection.h"

typedef projection::OutputOf<element::F32, projection::AllRows, SEISMIC_DIM_SN, SEISMIC_DIM_IN != 0,
    (SEISMIC_DIM_IN > SEISMIC_DIM_SN)> output_join;

kernel void stream_join(device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const float *y [[buffer(SEISMIC_BUFFER_Y)]],
    device const float *inject [[buffer(SEISMIC_BUFFER_INJECT)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    const uint h = uint(SEISMIC_DIM_H);
    if (index >= uint(SEISMIC_DIM_M) * h)
        return;
    const uint row = index / h, column = index % h;
    const auto out = output_join::make(result, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, residual,
        SEISMIC_RESIDUAL_STRIDE_0, SEISMIC_RESIDUAL_STRIDE_1, {}, inject, SEISMIC_INJECT_STRIDE_0, h);
    out.store(row, column, y[ulong(row) * SEISMIC_Y_STRIDE_0 + ulong(column) * SEISMIC_Y_STRIDE_1]);
}
