// stream_join (contract and portable body in streams.seismic): four columns
// of a row per thread, joining the F32 output into the streams through the
// output entries' epilogue (`projection::OutputOf`: the residual add, or the
// `Scatter` with the inject values and, when the inject row carries one,
// their comb).
#include "lib/projection/join.cuh"

using Output = projection::OutputOf<element::F32, projection::AllRows, SEISMIC_DIM_SN, (SEISMIC_DIM_IN != 0),
                                    (SEISMIC_DIM_IN > SEISMIC_DIM_SN)>;

extern "C" __global__ void stream_join(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *y = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_Y));
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 h = SEISMIC_DIM_H;
    const u64 vecs_per_row = (h + 3) / 4;
    if (index >= (u64)SEISMIC_DIM_M * vecs_per_row)
        return;
    const unsigned row = (unsigned)(index / vecs_per_row);
    const u64 column = (index % vecs_per_row) * 4;
    const Output::type out =
        Output::make(reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,
                     projection::AllRows{}, SEISMIC_RESULT_0_OUT,
                     SEISMIC_RESULT_0_STRIDE_0, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INJECT)),
                     SEISMIC_INJECT_STRIDE_0, h);
    const float *y_row = y + (u64)row * SEISMIC_Y_STRIDE_0;
    if (column + 4 <= h) {
        float4 values;
        if (SEISMIC_Y_STRIDE_1 == 1 && (reinterpret_cast<u64>(y_row + column) & 15u) == 0) {
            values = *reinterpret_cast<const float4 *>(y_row + column);
        } else {
            values = make_float4(y_row[column * SEISMIC_Y_STRIDE_1], y_row[(column + 1) * SEISMIC_Y_STRIDE_1],
                                 y_row[(column + 2) * SEISMIC_Y_STRIDE_1], y_row[(column + 3) * SEISMIC_Y_STRIDE_1]);
        }
        out.quad(row, column, values);
    } else {
        for (u64 c = column; c < h; ++c)
            out(row, c, y_row[c * SEISMIC_Y_STRIDE_1], 0.0f);
    }
}
