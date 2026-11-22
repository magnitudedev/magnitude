// corrector_features: one threadgroup of 256 threads per proposing row
// reduces its softmax normalizer over the V logits, then in rank tiles of at
// most 64 forms each candidate's five normed features once into threadgroup
// memory and projects the H outputs silu(up · normed + up_bias), loading each
// column's up_weight and up_bias once per tile, rounded once to A.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"
#include "lib/core/reduce.h"

typedef activation::Act activation_element;

#define FEATURES_THREADS 256u

kernel void corrector_features(
    device const float *logits [[buffer(SEISMIC_BUFFER_LOGITS)]],
    device const float *unary [[buffer(SEISMIC_BUFFER_UNARY)]],
    device const float *norm_weight [[buffer(SEISMIC_BUFFER_NORM_WEIGHT)]],
    device const float *norm_bias [[buffer(SEISMIC_BUFFER_NORM_BIAS)]],
    device const float *up_weight [[buffer(SEISMIC_BUFFER_UP_WEIGHT)]],
    device const float *up_bias [[buffer(SEISMIC_BUFFER_UP_BIAS)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup reduce::OnlineSoftmax partials[FEATURES_THREADS / 32u];
    constexpr uint K_TILE =
        SEISMIC_DIM_K < 64ul ? (SEISMIC_DIM_K > 0ul ? uint(SEISMIC_DIM_K) : 1u) : 64u;
    threadgroup float normed_shared[K_TILE * 5u];
    const element::Out<activation_element> result(result_cells);
    const ulong vocabulary = SEISMIC_DIM_V, count = SEISMIC_DIM_K, width = SEISMIC_DIM_H;
    device const float *line = logits + ulong(row) * SEISMIC_LOGITS_STRIDE_0;
    reduce::OnlineSoftmax state = reduce::OnlineSoftmax::empty();
    if (SEISMIC_LOGITS_STRIDE_1 == 1ul) {
        const ulong vocabulary4 = vocabulary & ~3ul;
        for (ulong i = ulong(thread_index) * 4ul; i < vocabulary4; i += ulong(FEATURES_THREADS) * 4ul) {
            const float4 v = float4(*reinterpret_cast<device const packed_float4 *>(line + i));
            state.add(v.x);
            state.add(v.y);
            state.add(v.z);
            state.add(v.w);
        }
        for (ulong i = vocabulary4 + thread_index; i < vocabulary; i += FEATURES_THREADS)
            state.add(line[i]);
    } else {
        for (ulong i = thread_index; i < vocabulary; i += FEATURES_THREADS)
            state.add(line[i * SEISMIC_LOGITS_STRIDE_1]);
    }
    const reduce::OnlineSoftmax softmax =
        reduce::group_online_softmax<FEATURES_THREADS / 32u>(state, partials, sg, lane);
    const float normalizer = softmax.max + metal::log(softmax.sum);
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    device const float *candidates = unary + ulong(row) * SEISMIC_UNARY_STRIDE_0;
    for (ulong base = 0; base < count; base += ulong(K_TILE)) {
        const ulong remaining = count - base;
        const ulong tile_count = remaining < ulong(K_TILE) ? remaining : ulong(K_TILE);
        if (base > 0ul)
            threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ulong local_rank = thread_index; local_rank < tile_count; local_rank += FEATURES_THREADS) {
            const ulong rank = base + local_rank;
            const float first = candidates[0];
            const float logit = candidates[rank * SEISMIC_UNARY_STRIDE_1];
            const float lp = logit - normalizer;
            const float features[5] = {lp, metal::exp(lp), logit - first,
                count > 1ul ? float(rank) / float(count - 1ul) : 0.0f, rank == 0ul ? 1.0f : 0.0f};
            float mean = 0.0f;
            for (uint f = 0; f < 5u; ++f)
                mean += features[f];
            mean /= 5.0f;
            float variance = 0.0f;
            for (uint f = 0; f < 5u; ++f)
                variance += (features[f] - mean) * (features[f] - mean);
            const float inverse = 1.0f / metal::sqrt(variance / 5.0f + epsilon);
            threadgroup float *dst = normed_shared + local_rank * 5ul;
            for (uint f = 0; f < 5u; ++f)
                dst[f] = (features[f] - mean) * inverse * norm_weight[f * SEISMIC_NORM_WEIGHT_STRIDE_0] +
                    norm_bias[f * SEISMIC_NORM_BIAS_STRIDE_0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const ulong out_base = (ulong(row) * count + base) * SEISMIC_RESULT_0_STRIDE_0;
        for (ulong column = thread_index; column < width; column += FEATURES_THREADS) {
            device const float *weights = up_weight + column * SEISMIC_UP_WEIGHT_STRIDE_0;
            const float bias = up_bias[column * SEISMIC_UP_BIAS_STRIDE_0];
            float w[5];
            for (uint f = 0; f < 5u; ++f)
                w[f] = weights[f * SEISMIC_UP_WEIGHT_STRIDE_1];
            const ulong out_col = column * SEISMIC_RESULT_0_STRIDE_1;
            for (ulong local_rank = 0; local_rank < tile_count; ++local_rank) {
                threadgroup const float *normed = normed_shared + local_rank * 5ul;
                float value = bias;
                for (uint f = 0; f < 5u; ++f)
                    value = metal::fma(w[f], normed[f], value);
                result.put(out_base + local_rank * SEISMIC_RESULT_0_STRIDE_0 + out_col,
                    value / (1.0f + metal::exp(-value)));
            }
        }
    }
}
