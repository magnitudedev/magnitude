// stream_join (contract and portable body in streams.seismic): one thread
// per column of a row, joining the F32 output into the streams through the
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
    if (index >= (u64)SEISMIC_DIM_M * h)
        return;
    const unsigned row = (unsigned)(index / h);
    const u64 column = index % h;
    const Output::type out =
        Output::make(reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)), SEISMIC_RESIDUAL_STRIDE_0,
                     projection::AllRows{}, reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)),
                     SEISMIC_RESULT_0_STRIDE_0, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INJECT)),
                     SEISMIC_INJECT_STRIDE_0, h);
    out(row, column, y[(u64)row * SEISMIC_Y_STRIDE_0 + column * SEISMIC_Y_STRIDE_1], 0.0f);
}
