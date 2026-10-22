// corrector_features: one threadgroup of 256 threads per proposing row
// reduces its softmax normalizer over the V logits, then forms each of its K
// candidates' five features and their layer norm (every thread, redundantly)
// and the H outputs silu(up · normed + up_bias), one column per thread,
// rounded once to A.
#include "lib/core/activation.h"
#include "lib/core/reduce.h"

typedef element::Act activation;

#define FEATURES_THREADS 256u

kernel void corrector_features(
    device const float *logits [[buffer(SEISMIC_BUFFER_LOGITS)]],
    device const float *unary [[buffer(SEISMIC_BUFFER_UNARY)]],
    device const float *norm_weight [[buffer(SEISMIC_BUFFER_NORM_WEIGHT)]],
    device const float *norm_bias [[buffer(SEISMIC_BUFFER_NORM_BIAS)]],
    device const float *up_weight [[buffer(SEISMIC_BUFFER_UP_WEIGHT)]],
    device const float *up_bias [[buffer(SEISMIC_BUFFER_UP_BIAS)]],
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[FEATURES_THREADS / 32u];
    const ulong vocabulary = SEISMIC_DIM_V, count = SEISMIC_DIM_K, width = SEISMIC_DIM_H;
    device const float *line = logits + ulong(row) * SEISMIC_LOGITS_STRIDE_0;
    float local = -INFINITY;
    for (ulong i = thread_index; i < vocabulary; i += FEATURES_THREADS)
        local = metal::max(local, line[i * SEISMIC_LOGITS_STRIDE_1]);
    const float maximum = reduce::group_max<FEATURES_THREADS / 32u>(local, partials, sg, lane);
    float partial = 0.0f;
    for (ulong i = thread_index; i < vocabulary; i += FEATURES_THREADS)
        partial += metal::exp(line[i * SEISMIC_LOGITS_STRIDE_1] - maximum);
    const float normalizer =
        maximum + metal::log(reduce::group_sum<FEATURES_THREADS / 32u>(partial, partials, sg, lane));
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    device const float *candidates = unary + ulong(row) * SEISMIC_UNARY_STRIDE_0;
    const float first = candidates[0];
    for (ulong rank = 0; rank < count; ++rank) {
        const float logit = candidates[rank * SEISMIC_UNARY_STRIDE_1];
        const float lp = logit - normalizer;
        const float features[5] = {lp, metal::exp(lp), logit - first,
            count > 1 ? float(rank) / float(count - 1) : 0.0f, rank == 0 ? 1.0f : 0.0f};
        float mean = 0.0f;
        for (uint f = 0; f < 5u; ++f)
            mean += features[f];
        mean /= 5.0f;
        float variance = 0.0f;
        for (uint f = 0; f < 5u; ++f)
            variance += (features[f] - mean) * (features[f] - mean);
        const float inverse = 1.0f / metal::sqrt(variance / 5.0f + epsilon);
        float normed[5];
        for (uint f = 0; f < 5u; ++f)
            normed[f] = (features[f] - mean) * inverse * norm_weight[f * SEISMIC_NORM_WEIGHT_STRIDE_0] +
                norm_bias[f * SEISMIC_NORM_BIAS_STRIDE_0];
        const ulong out = (ulong(row) * count + rank) * SEISMIC_RESULT_0_STRIDE_0;
        for (ulong column = thread_index; column < width; column += FEATURES_THREADS) {
            device const float *weights = up_weight + column * SEISMIC_UP_WEIGHT_STRIDE_0;
            float value = up_bias[column * SEISMIC_UP_BIAS_STRIDE_0];
            for (uint f = 0; f < 5u; ++f)
                value = metal::fma(weights[f * SEISMIC_UP_WEIGHT_STRIDE_1], normed[f], value);
            element::put<activation>(
                result, out + column * SEISMIC_RESULT_0_STRIDE_1, value / (1.0f + metal::exp(-value)));
        }
    }
}
