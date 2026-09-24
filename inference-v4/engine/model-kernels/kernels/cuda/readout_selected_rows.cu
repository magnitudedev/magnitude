// readout_selected_rows: F32 logits of the `out_rows` rows for a selected
// vocabulary subset.
//   stage (O blocks): the final RMS rows as A, into scratch.
//   logits (Sv blocks): one block per selected vocabulary row decodes
//     that weight row once per feature row (mma16 lane chunks) and sums the
//     products in a fixed thread order.
#define KERNEL_W0 SEISMIC_WEIGHT
#include "common/projection.cuh"

extern "C" __global__ void readout_selected_rows_stage(SEISMIC_KERNEL_PARAMS) {
    using Pro = projection::Rms<ELEMENT_OF(SEISMIC_NORM), projection::SelectedRows>;
    __shared__ float factors[1];
    __shared__ float scratch[32];
    const unsigned row = blockIdx.x;
    const Pro pro{reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,
                  SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON), SEISMIC_DIM_D,
                  projection::SelectedRows{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))}};
    pro.prepare_row(factors, row, scratch);
    __syncthreads();
    projection::u8 *features = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_FEATURES);
    for (projection::u64 k = threadIdx.x; k < SEISMIC_DIM_D; k += blockDim.x)
        element::put<element::Act>(features, row * SEISMIC_DIM_D + k, pro.value(factors, row, k));
}

extern "C" __global__ void readout_selected_rows_logits(SEISMIC_KERNEL_PARAMS) {
    __shared__ float scratch[32];
    const projection::u64 column = blockIdx.x;
    const projection::u64 vocabulary =
        (projection::u64)reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTED))[column * SEISMIC_SELECTED_STRIDE_0];
    const auto weight = KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT));
    const projection::u8 *features = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_FEATURES);
    float *logits = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    // Each work item (k-block, column pair t) owns 16 values of the row.
    const projection::u64 items = SEISMIC_DIM_D / 64 * 4;
    for (projection::u64 o = 0; o < SEISMIC_DIM_O; ++o) {
        float partial = 0.0f;
        for (projection::u64 item = threadIdx.x; item < items; item += blockDim.x) {
            const projection::u64 kb = item / 4;
            const projection::u32 t = (projection::u32)(item % 4);
            float values[16];
            packets::row_values16(weight, vocabulary, kb, t, values);
            const projection::u32 offsets[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
#pragma unroll
            for (int s = 0; s < 4; ++s)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float x = element::at<element::Act>(features, o * SEISMIC_DIM_D + kb * 64 + 16 * s + offsets[j]);
                    partial = seismic_fma_rn(x, values[4 * s + j], partial);
                }
        }
        const float total = reduce::group_sum(partial, scratch);
        if (threadIdx.x == 0)
            logits[o * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1] = total;
    }
}
