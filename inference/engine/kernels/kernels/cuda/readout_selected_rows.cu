// readout_selected_rows: F32 logits of the `out_rows` rows for a selected
// vocabulary subset.
//   stage (O blocks): the final RMS rows as A, into scratch.
//   logits (Sv blocks): one block per selected vocabulary row decodes
//     that weight row once per feature row (mma16 lane chunks) and sums the
//     products in a fixed thread order, softcapped when `softcap` > 0.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_WEIGHT
#include "lib/projection/projection.cuh"

// The weight's operand rows of the final RMS rows, as A, into scratch.
extern "C" __global__ void readout_selected_rows_stage(SEISMIC_KERNEL_PARAMS) {
    using Raw = projection::Input<ELEMENT_OF(SEISMIC_NORM), projection::SelectedRows, SEISMIC_DIM_NM, SEISMIC_DIM_PT>;
    using Pro = projection::Operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4, Raw>;
    const Pro pro{Raw{reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)), SEISMIC_HIDDEN_STRIDE_0,
                      SEISMIC_PTR(SEISMIC_BUFFER_NORM), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON),
                      SEISMIC_DIM_D,
                      projection::SelectedRows{reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS))}}};
    projection::form_row(pro, blockIdx.x, SEISMIC_DIM_D, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_FEATURES));
}

extern "C" __global__ void readout_selected_rows_logits(SEISMIC_KERNEL_PARAMS) {
    __shared__ float scratch[32];
    const projection::u64 column = blockIdx.x;
    const projection::u64 vocabulary =
        (projection::u64)reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTED))[column * SEISMIC_SELECTED_STRIDE_0];
    const auto weight = KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT));
    const projection::u8 *features = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_FEATURES);
    const auto logits = SEISMIC_RESULT_0_OUT;
    for (projection::u64 o = 0; o < SEISMIC_DIM_O; ++o) {
        // Each work item (k-block, column pair t) owns 16 values of the row,
        // a lane chunk's whatever the weight's layout.
        float partial = 0.0f;
        packets::row_items16(weight, vocabulary, 0, SEISMIC_DIM_D, threadIdx.x, blockDim.x,
                             [&](projection::u64 column, float value) {
                                 const float x = element::at<activation::Act>(features, o * SEISMIC_DIM_D + column);
                                 partial = seismic_fma_rn(x, value, partial);
                             });
        const float total = reduce::group_sum(partial, scratch);
        const float cap = __uint_as_float((unsigned)SEISMIC_PARAM_SOFTCAP);
        if (threadIdx.x == 0)
            logits.put(o * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1,
                       cap > 0.0f ? functions::softcap(cap, total) : total);
    }
}
