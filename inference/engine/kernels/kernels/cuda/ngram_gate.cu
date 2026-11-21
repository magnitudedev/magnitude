// ngram_gate (contract and portable body in ngram.seismic): each row's
// per-stream gate sigmoid(g) and the inverse RMS of the gated value. One
// block per (stream, row), the y grid visiting the rows with a stride:
// the stream's key and query square sums, their normalized product, and the
// value row's square sum. Under the stored product (PR = 1) the query weight
// is one, which leaves each term's bits as the body's.
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

using element::u64;
using element::u8;

extern "C" __global__ void ngram_gate(SEISMIC_KERNEL_PARAMS) {
    using KeyNorm = ELEMENT_OF(SEISMIC_KEY_NORM);
    using QueryNorm = ELEMENT_OF(SEISMIC_QUERY_NORM);
    __shared__ float partials[4 * 32];
    const u64 width = SEISMIC_DIM_H;
    const u64 width4 = width & ~3ull;
    const u64 stream = blockIdx.x;
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const float *key = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_KEY));
    const float *hidden = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN));
    const float *value = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_VALUE));
    const u8 *key_norm = SEISMIC_PTR(SEISMIC_BUFFER_KEY_NORM);
    const u8 *query_norm = SEISMIC_PTR(SEISMIC_BUFFER_QUERY_NORM);
    const u8 *kn_base = key_norm + stream * SEISMIC_KEY_NORM_STRIDE_0 * KeyNorm::bytes;
    const u8 *qn_base = query_norm + stream * SEISMIC_QUERY_NORM_STRIDE_0 * QueryNorm::bytes;
    const bool norm_vec = SEISMIC_KEY_NORM_STRIDE_1 == 1ull &&
                          (reinterpret_cast<u64>(kn_base) & (4ull * KeyNorm::bytes - 1ull)) == 0ull &&
                          (SEISMIC_DIM_PR == 1 ||
                           (SEISMIC_QUERY_NORM_STRIDE_1 == 1ull &&
                            (reinterpret_cast<u64>(qn_base) & (4ull * QueryNorm::bytes - 1ull)) == 0ull));
    const auto scales = SEISMIC_RESULT_0_OUT;
    for (u64 row = blockIdx.y; row < SEISMIC_DIM_M; row += gridDim.y) {
        const float *k = key + row * SEISMIC_KEY_STRIDE_0 + stream * SEISMIC_KEY_STRIDE_1;
        const float *x = hidden + row * SEISMIC_HIDDEN_STRIDE_0 + stream * SEISMIC_HIDDEN_STRIDE_1;
        const float *v = value + row * SEISMIC_VALUE_STRIDE_0;
        const bool vec_ok = norm_vec &&
                            SEISMIC_KEY_STRIDE_2 == 1ull && (reinterpret_cast<u64>(k) & 15ull) == 0ull &&
                            SEISMIC_HIDDEN_STRIDE_2 == 1ull && (reinterpret_cast<u64>(x) & 15ull) == 0ull &&
                            SEISMIC_VALUE_STRIDE_1 == 1ull && (reinterpret_cast<u64>(v) & 15ull) == 0ull;
        float keys = 0.0f;
        float queries = 0.0f;
        float dot = 0.0f;
        float values = 0.0f;
        if (vec_ok) {
            const u64 step4 = static_cast<u64>(blockDim.x) * 4ull;
            for (u64 i = static_cast<u64>(threadIdx.x) * 4ull; i < width4; i += step4) {
                const float4 kv = *reinterpret_cast<const float4 *>(k + i);
                const float4 xv = *reinterpret_cast<const float4 *>(x + i);
                const float4 vv = *reinterpret_cast<const float4 *>(v + i);
                float kn[4];
                element::span<KeyNorm, 4, false>(kn_base, i, kn);
                float qn[4] = {1.0f, 1.0f, 1.0f, 1.0f};
                if (SEISMIC_DIM_PR == 0)
                    element::span<QueryNorm, 4, false>(qn_base, i, qn);
                keys = seismic_fma_rn(kv.x, kv.x, keys);
                keys = seismic_fma_rn(kv.y, kv.y, keys);
                keys = seismic_fma_rn(kv.z, kv.z, keys);
                keys = seismic_fma_rn(kv.w, kv.w, keys);
                queries = seismic_fma_rn(xv.x, xv.x, queries);
                queries = seismic_fma_rn(xv.y, xv.y, queries);
                queries = seismic_fma_rn(xv.z, xv.z, queries);
                queries = seismic_fma_rn(xv.w, xv.w, queries);
                dot += kv.x * kn[0] * xv.x * qn[0];
                dot += kv.y * kn[1] * xv.y * qn[1];
                dot += kv.z * kn[2] * xv.z * qn[2];
                dot += kv.w * kn[3] * xv.w * qn[3];
                values = seismic_fma_rn(vv.x, vv.x, values);
                values = seismic_fma_rn(vv.y, vv.y, values);
                values = seismic_fma_rn(vv.z, vv.z, values);
                values = seismic_fma_rn(vv.w, vv.w, values);
            }
            for (u64 i = width4 + threadIdx.x; i < width; i += blockDim.x) {
                const float kv = k[i];
                const float xv = x[i];
                const float vv = v[i];
                keys = seismic_fma_rn(kv, kv, keys);
                queries = seismic_fma_rn(xv, xv, queries);
                const float qn = SEISMIC_DIM_PR == 1 ? 1.0f : element::at<QueryNorm>(qn_base, i);
                dot += kv * element::at<KeyNorm>(kn_base, i) * xv * qn;
                values = seismic_fma_rn(vv, vv, values);
            }
        } else {
            for (u64 i = threadIdx.x; i < width; i += blockDim.x) {
                const float kv = k[i * SEISMIC_KEY_STRIDE_2];
                const float xv = x[i * SEISMIC_HIDDEN_STRIDE_2];
                const float vv = v[i * SEISMIC_VALUE_STRIDE_1];
                keys = seismic_fma_rn(kv, kv, keys);
                queries = seismic_fma_rn(xv, xv, queries);
                const float qn = SEISMIC_DIM_PR == 1 ? 1.0f
                    : element::at<QueryNorm>(query_norm, stream * SEISMIC_QUERY_NORM_STRIDE_0 + i * SEISMIC_QUERY_NORM_STRIDE_1);
                dot += kv * element::at<KeyNorm>(key_norm, stream * SEISMIC_KEY_NORM_STRIDE_0 + i * SEISMIC_KEY_NORM_STRIDE_1) *
                       xv * qn;
                values = seismic_fma_rn(vv, vv, values);
            }
        }
        float sums[4] = {keys, queries, dot, values};
        reduce::group_sum_n<4>(sums, partials);
        keys = sums[0];
        queries = sums[1];
        dot = sums[2];
        values = sums[3];
        if (threadIdx.x == 0) {
            const float g = dot * rsqrtf(keys / (float)width + epsilon) * rsqrtf(queries / (float)width + epsilon) /
                            sqrtf((float)width);
            const float root = sqrtf(fmaxf(fabsf(g), 1e-6f));
            const float signed_root = g > 0.0f ? root : (g < 0.0f ? -root : 0.0f);
            const float gate = 1.0f / (1.0f + expf(-signed_root));
            const auto out = scales.offset(row * SEISMIC_RESULT_0_STRIDE_0);
            out.put(stream * SEISMIC_RESULT_0_STRIDE_1, gate);
            out.put((SEISMIC_DIM_S + stream) * SEISMIC_RESULT_0_STRIDE_1,
                    rsqrtf(gate * gate * (values / (float)width) + epsilon));
        }
    }
}
