// corrector_attention: one threadgroup of 256 threads per (node, head). Each
// thread scores whole keys of the block (the biased query against the biased
// key, scaled, plus the head's additive bias) into shared memory; the
// threadgroup reduces their maximum and softmax denominator, then each thread
// accumulates whole columns of the weighted biased values, published to A.
#include "lib/core/activation.h"
#include "lib/core/reduce.h"

typedef element::Act activation;

#define CORRECTOR_THREADS 256u

kernel void corrector_attention(
    device const float *qkv [[buffer(SEISMIC_BUFFER_QKV)]],
    device const float *input_bias [[buffer(SEISMIC_BUFFER_INPUT_BIAS)]],
    device const float *bias [[buffer(SEISMIC_BUFFER_BIAS)]],
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong nodes = SEISMIC_DIM_N, width = SEISMIC_DIM_W;
    // The block's scores, then the reduction partials (N * 4 + 64 bytes).
    threadgroup float *scores = shared;
    threadgroup float *partials = shared + nodes;
    const ulong row = ulong(group.x), head = ulong(group.y);
    const ulong block = row / nodes, query_node = row % nodes;
    device const float *block_rows = qkv + block * SEISMIC_QKV_STRIDE_0;
    device const float *query = block_rows + query_node * SEISMIC_QKV_STRIDE_1 + head * SEISMIC_QKV_STRIDE_3;
    device const float *query_bias = input_bias + head * SEISMIC_INPUT_BIAS_STRIDE_1;
    device const float *key_bias = query_bias + SEISMIC_INPUT_BIAS_STRIDE_0;
    device const float *value_bias = key_bias + SEISMIC_INPUT_BIAS_STRIDE_0;
    const float scale = 1.0f / metal::sqrt(float(width));
    float local = -INFINITY;
    for (ulong key = thread_index; key < nodes; key += CORRECTOR_THREADS) {
        device const float *k =
            block_rows + key * SEISMIC_QKV_STRIDE_1 + SEISMIC_QKV_STRIDE_2 + head * SEISMIC_QKV_STRIDE_3;
        float dot = 0.0f;
        for (ulong c = 0; c < width; ++c)
            dot = metal::fma(query[c * SEISMIC_QKV_STRIDE_4] + query_bias[c * SEISMIC_INPUT_BIAS_STRIDE_2],
                k[c * SEISMIC_QKV_STRIDE_4] + key_bias[c * SEISMIC_INPUT_BIAS_STRIDE_2], dot);
        const float score = dot * scale +
            bias[head * SEISMIC_BIAS_STRIDE_0 + query_node * SEISMIC_BIAS_STRIDE_1 + key * SEISMIC_BIAS_STRIDE_2];
        scores[key] = score;
        local = metal::max(local, score);
    }
    const float maximum = reduce::group_max<CORRECTOR_THREADS / 32u>(local, partials, sg, lane);
    float partial = 0.0f;
    for (ulong key = thread_index; key < nodes; key += CORRECTOR_THREADS) {
        const float weight = metal::exp(scores[key] - maximum);
        scores[key] = weight;
        partial += weight;
    }
    // The sum's barriers also publish every weight to the threadgroup.
    const float inverse = 1.0f / reduce::group_sum<CORRECTOR_THREADS / 32u>(partial, partials, sg, lane);
    const ulong out = block * SEISMIC_RESULT_0_STRIDE_0 + query_node * SEISMIC_RESULT_0_STRIDE_1 +
        head * SEISMIC_RESULT_0_STRIDE_2;
    device const float *values = block_rows + 2u * SEISMIC_QKV_STRIDE_2 + head * SEISMIC_QKV_STRIDE_3;
    for (ulong column = thread_index; column < width; column += CORRECTOR_THREADS) {
        const float b = value_bias[column * SEISMIC_INPUT_BIAS_STRIDE_2];
        float acc = 0.0f;
        for (ulong key = 0; key < nodes; ++key)
            acc = metal::fma(scores[key], values[key * SEISMIC_QKV_STRIDE_1 + column * SEISMIC_QKV_STRIDE_4] + b, acc);
        element::put<activation>(result, out + column * SEISMIC_RESULT_0_STRIDE_3, acc * inverse);
    }
}
