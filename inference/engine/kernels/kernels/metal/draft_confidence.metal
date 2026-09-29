// draft_confidence: one threadgroup per row reduces weight · [features ;
// memory] + bias and marks the row's selection declined (status 3) when its
// sigmoid is below the threshold.
#include "lib/core/activation.h"
#include "lib/core/reduce.h"

typedef element::Act activation;

kernel void draft_confidence(
    device const uchar *features [[buffer(SEISMIC_BUFFER_FEATURES)]],
    device const uchar *memory [[buffer(SEISMIC_BUFFER_MEMORY)]],
    device const float *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],
    device const float *bias [[buffer(SEISMIC_BUFFER_BIAS)]],
    device int *selection [[buffer(SEISMIC_BUFFER_SELECTION)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[8];
    const ulong width = SEISMIC_DIM_D, rank = SEISMIC_DIM_R;
    float partial = 0.0f;
    for (ulong i = thread_index; i < width; i += 256u)
        partial = metal::fma(element::at<activation>(features, ulong(row) * SEISMIC_FEATURES_STRIDE_0 +
                                                                   i * SEISMIC_FEATURES_STRIDE_1),
            weight[i * SEISMIC_WEIGHT_STRIDE_0], partial);
    for (ulong i = thread_index; i < rank; i += 256u)
        partial = metal::fma(element::at<activation>(memory, ulong(row) * SEISMIC_MEMORY_STRIDE_0 +
                                                                 i * SEISMIC_MEMORY_STRIDE_1),
            weight[(width + i) * SEISMIC_WEIGHT_STRIDE_0], partial);
    const float score = reduce::group_sum<8>(partial, partials, sg, lane) + bias[0];
    if (thread_index == 0 && 1.0f / (1.0f + metal::exp(-score)) < as_type<float>(uint(SEISMIC_PARAM_THRESHOLD)))
        selection[ulong(row) * SEISMIC_SELECTION_STRIDE_0 + SEISMIC_SELECTION_STRIDE_1] = 3;
}
