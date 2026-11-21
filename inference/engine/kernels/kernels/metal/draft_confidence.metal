// draft_confidence: one threadgroup per row reduces weight · [features ;
// memory] + bias and marks the row's selection declined (status 3) when its
// sigmoid is below the threshold.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"
#include "lib/core/reduce.h"

typedef activation::Act activation_element;

inline float dot_accumulate(
    device const uchar *act_row,
    ulong act_stride,
    device const float *w,
    ulong w_stride,
    ulong count,
    uint thread_index,
    float partial) {
    if (act_stride == 1ul && w_stride == 1ul) {
        const ulong count4 = count & ~3ul;
        device const activation_element::storage *act_base =
            reinterpret_cast<device const activation_element::storage *>(act_row);
        for (ulong i = ulong(thread_index) * 4ul; i < count4; i += 1024ul) {
            const float4 a = activation_element::load4(act_base + i);
            const float4 wv = float4(*reinterpret_cast<device const packed_float4 *>(w + i));
            partial = metal::fma(a.x, wv.x, partial);
            partial = metal::fma(a.y, wv.y, partial);
            partial = metal::fma(a.z, wv.z, partial);
            partial = metal::fma(a.w, wv.w, partial);
        }
        for (ulong i = count4 + thread_index; i < count; i += 256u)
            partial = metal::fma(activation_element::load(act_base[i]), w[i], partial);
    } else {
        for (ulong i = thread_index; i < count; i += 256u)
            partial = metal::fma(element::at<activation_element>(act_row, i * act_stride), w[i * w_stride], partial);
    }
    return partial;
}

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
    device const uchar *features_row = features + ulong(row) * SEISMIC_FEATURES_STRIDE_0 * activation_element::bytes;
    device const uchar *memory_row = memory + ulong(row) * SEISMIC_MEMORY_STRIDE_0 * activation_element::bytes;
    float partial = 0.0f;
    partial = dot_accumulate(features_row, SEISMIC_FEATURES_STRIDE_1, weight, SEISMIC_WEIGHT_STRIDE_0, width,
        thread_index, partial);
    partial = dot_accumulate(memory_row, SEISMIC_MEMORY_STRIDE_1, weight + width * SEISMIC_WEIGHT_STRIDE_0,
        SEISMIC_WEIGHT_STRIDE_0, rank, thread_index, partial);
    const float score = reduce::group_sum<8>(partial, partials, sg, lane) + bias[0];
    if (thread_index == 0 && 1.0f / (1.0f + metal::exp(-score)) < as_type<float>(uint(SEISMIC_PARAM_THRESHOLD)))
        selection[ulong(row) * SEISMIC_SELECTION_STRIDE_0 + SEISMIC_SELECTION_STRIDE_1] = 3;
}
