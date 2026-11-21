// ngram_gate (contract and portable body in ngram.seismic): each row's
// per-stream gate sigmoid(g) and the inverse RMS of the gated value. One
// threadgroup per (stream, row) reduces the stream's key and query square
// sums, their normalized product, and the value row's square sum. Under the
// stored product (PR = 1) the query weight is one, which leaves each term's
// bits as the body's.
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
    SEISMIC_RESULT_0_PARAM(scales),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[4 * 8];
    const ulong width = SEISMIC_DIM_H;
    const ulong width4 = width & ~3ul;
    const ulong stream = group.x;
    const ulong row = group.y;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    device const float *k = key + row * SEISMIC_KEY_STRIDE_0 + stream * SEISMIC_KEY_STRIDE_1;
    device const float *x = hidden + row * SEISMIC_HIDDEN_STRIDE_0 + stream * SEISMIC_HIDDEN_STRIDE_1;
    device const float *v = value + row * SEISMIC_VALUE_STRIDE_0;
    const bool vec_ok = SEISMIC_KEY_STRIDE_2 == 1ul && SEISMIC_HIDDEN_STRIDE_2 == 1ul &&
                        SEISMIC_VALUE_STRIDE_1 == 1ul && SEISMIC_KEY_NORM_STRIDE_1 == 1ul &&
                        (SEISMIC_DIM_PR == 1 || SEISMIC_QUERY_NORM_STRIDE_1 == 1ul);
    float keys = 0.0f;
    float queries = 0.0f;
    float dot = 0.0f;
    float values = 0.0f;
    if (vec_ok) {
        device const KeyNorm::storage *kn_base =
            reinterpret_cast<device const KeyNorm::storage *>(key_norm) + stream * SEISMIC_KEY_NORM_STRIDE_0;
        device const QueryNorm::storage *qn_base =
            reinterpret_cast<device const QueryNorm::storage *>(query_norm) + stream * SEISMIC_QUERY_NORM_STRIDE_0;
        for (ulong i = ulong(thread_index) * 4ul; i < width4; i += 1024ul) {
            const float4 kv = float4(*reinterpret_cast<device const packed_float4 *>(k + i));
            const float4 xv = float4(*reinterpret_cast<device const packed_float4 *>(x + i));
            const float4 vv = float4(*reinterpret_cast<device const packed_float4 *>(v + i));
            const float4 kn = KeyNorm::load4(kn_base + i);
            const float4 qn = SEISMIC_DIM_PR == 1 ? float4(1.0f) : QueryNorm::load4(qn_base + i);
            keys = metal::fma(kv.x, kv.x, keys);
            keys = metal::fma(kv.y, kv.y, keys);
            keys = metal::fma(kv.z, kv.z, keys);
            keys = metal::fma(kv.w, kv.w, keys);
            queries = metal::fma(xv.x, xv.x, queries);
            queries = metal::fma(xv.y, xv.y, queries);
            queries = metal::fma(xv.z, xv.z, queries);
            queries = metal::fma(xv.w, xv.w, queries);
            dot += kv.x * kn.x * xv.x * qn.x;
            dot += kv.y * kn.y * xv.y * qn.y;
            dot += kv.z * kn.z * xv.z * qn.z;
            dot += kv.w * kn.w * xv.w * qn.w;
            values = metal::fma(vv.x, vv.x, values);
            values = metal::fma(vv.y, vv.y, values);
            values = metal::fma(vv.z, vv.z, values);
            values = metal::fma(vv.w, vv.w, values);
        }
        for (ulong i = width4 + thread_index; i < width; i += 256u) {
            const float kv = k[i];
            const float xv = x[i];
            const float vv = v[i];
            keys = metal::fma(kv, kv, keys);
            queries = metal::fma(xv, xv, queries);
            const float qn = SEISMIC_DIM_PR == 1 ? 1.0f : QueryNorm::load(qn_base[i]);
            dot += kv * KeyNorm::load(kn_base[i]) * xv * qn;
            values = metal::fma(vv, vv, values);
        }
    } else {
        for (ulong i = thread_index; i < width; i += 256u) {
            const float kv = k[i * SEISMIC_KEY_STRIDE_2];
            const float xv = x[i * SEISMIC_HIDDEN_STRIDE_2];
            const float vv = v[i * SEISMIC_VALUE_STRIDE_1];
            keys = metal::fma(kv, kv, keys);
            queries = metal::fma(xv, xv, queries);
            const float qn = SEISMIC_DIM_PR == 1 ? 1.0f
                : element::at<QueryNorm>(query_norm, stream * SEISMIC_QUERY_NORM_STRIDE_0 + i * SEISMIC_QUERY_NORM_STRIDE_1);
            dot += kv * element::at<KeyNorm>(key_norm, stream * SEISMIC_KEY_NORM_STRIDE_0 + i * SEISMIC_KEY_NORM_STRIDE_1)
                * xv * qn;
            values = metal::fma(vv, vv, values);
        }
    }
    float sums[4] = {keys, queries, dot, values};
    reduce::group_sum_n<4, 8>(sums, partials, sg, lane);
    keys = sums[0];
    queries = sums[1];
    dot = sums[2];
    values = sums[3];
    if (thread_index != 0)
        return;
    const float g = dot * metal::rsqrt(keys / float(width) + epsilon) * metal::rsqrt(queries / float(width) + epsilon)
        / metal::sqrt(float(width));
    const float root = metal::sqrt(metal::max(metal::abs(g), 1e-6f));
    const float signed_root = g > 0.0f ? root : (g < 0.0f ? -root : 0.0f);
    const float gate = 1.0f / (1.0f + metal::exp(-signed_root));
    const element::Out<element::F32> out = element::Out<element::F32>(scales).offset(row * SEISMIC_RESULT_0_STRIDE_0);
    out.put(stream * SEISMIC_RESULT_0_STRIDE_1, gate);
    out.put((SEISMIC_DIM_S + stream) * SEISMIC_RESULT_0_STRIDE_1,
        metal::rsqrt(gate * gate * (values / float(width)) + epsilon));
}
