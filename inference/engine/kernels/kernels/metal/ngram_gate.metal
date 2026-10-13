// ngram_gate (contract and portable body in ngram.seismic): each row's
// per-stream gate sigmoid(g) and the inverse RMS of the gated value. One
// threadgroup per (stream, row) reduces the stream's key and query square
// sums, their normalized product, and the value row's square sum.
#include "lib/core/activation.h"
#include "lib/core/reduce.h"

typedef ELEMENT_OF(SEISMIC_KEY_NORM) KeyNorm;
typedef ELEMENT_OF(SEISMIC_QUERY_NORM) QueryNorm;

kernel void ngram_gate(
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const float *key [[buffer(SEISMIC_BUFFER_KEY)]],
    device const float *value [[buffer(SEISMIC_BUFFER_VALUE)]],
    device const uchar *key_norm [[buffer(SEISMIC_BUFFER_KEY_NORM)]],
    device const uchar *query_norm [[buffer(SEISMIC_BUFFER_QUERY_NORM)]],
    device float *scales [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[8];
    const ulong width = SEISMIC_DIM_H;
    const ulong stream = group.x;
    const ulong row = group.y;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    device const float *k = key + row * SEISMIC_KEY_STRIDE_0 + stream * SEISMIC_KEY_STRIDE_1;
    device const float *x = hidden + row * SEISMIC_HIDDEN_STRIDE_0 + stream * SEISMIC_HIDDEN_STRIDE_1;
    device const float *v = value + row * SEISMIC_VALUE_STRIDE_0;
    float keys = 0.0f;
    float queries = 0.0f;
    float dot = 0.0f;
    float values = 0.0f;
    for (ulong i = thread_index; i < width; i += 256u) {
        const float kv = k[i * SEISMIC_KEY_STRIDE_2];
        const float xv = x[i * SEISMIC_HIDDEN_STRIDE_2];
        const float vv = v[i * SEISMIC_VALUE_STRIDE_1];
        keys = metal::fma(kv, kv, keys);
        queries = metal::fma(xv, xv, queries);
        dot += kv * element::at<KeyNorm>(key_norm, stream * SEISMIC_KEY_NORM_STRIDE_0 + i * SEISMIC_KEY_NORM_STRIDE_1)
            * xv * element::at<QueryNorm>(query_norm, stream * SEISMIC_QUERY_NORM_STRIDE_0 + i * SEISMIC_QUERY_NORM_STRIDE_1);
        values = metal::fma(vv, vv, values);
    }
    keys = reduce::group_sum<8>(keys, partials, sg, lane);
    queries = reduce::group_sum<8>(queries, partials, sg, lane);
    dot = reduce::group_sum<8>(dot, partials, sg, lane);
    values = reduce::group_sum<8>(values, partials, sg, lane);
    if (thread_index != 0)
        return;
    const float g = dot * metal::rsqrt(keys / float(width) + epsilon) * metal::rsqrt(queries / float(width) + epsilon)
        / metal::sqrt(float(width));
    const float root = metal::sqrt(metal::max(metal::abs(g), 1e-6f));
    const float signed_root = g > 0.0f ? root : (g < 0.0f ? -root : 0.0f);
    const float gate = 1.0f / (1.0f + metal::exp(-signed_root));
    device float *out = scales + row * SEISMIC_RESULT_0_STRIDE_0;
    out[stream * SEISMIC_RESULT_0_STRIDE_1] = gate;
    out[(SEISMIC_DIM_S + stream) * SEISMIC_RESULT_0_STRIDE_1] =
        metal::rsqrt(gate * gate * (values / float(width)) + epsilon);
}
