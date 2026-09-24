// readout_features_rows: the final RMS prologue over the demanded rows
// (`out_rows` gather), published as A. One block per output row.
#include "common/projection.cuh"

extern "C" __global__ void readout_features_rows(SEISMIC_KERNEL_PARAMS) {
    using Pro = projection::Rms<ELEMENT_OF(SEISMIC_NORM), projection::SelectedRows>;
    __shared__ float factors[1];
    __shared__ float scratch[32];
    const unsigned row = blockIdx.x;
    const Pro pro{reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,
                  SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON), SEISMIC_DIM_D,
                  projection::SelectedRows{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))}};
    pro.prepare_row(factors, row, scratch);
    __syncthreads();
    projection::u8 *features = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    for (projection::u64 k = threadIdx.x; k < SEISMIC_DIM_D; k += blockDim.x)
        element::put<element::Act>(features, row * SEISMIC_RESULT_0_STRIDE_0 + k, pro.value(factors, row, k));
}
