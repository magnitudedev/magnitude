// ngram_gate (contract and portable body in ngram.seismic): each row's
// per-stream gate sigmoid(g) and the inverse RMS of the gated value. One
// block per (stream, row), the y grid visiting the rows with a stride:
// the stream's key and query square sums, their normalized product, and the
// value row's square sum.
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

using element::u64;
using element::u8;

extern "C" __global__ void ngram_gate(SEISMIC_KERNEL_PARAMS) {
    using KeyNorm = ELEMENT_OF(SEISMIC_KEY_NORM);
    using QueryNorm = ELEMENT_OF(SEISMIC_QUERY_NORM);
    __shared__ float partials[32];
    const u64 width = SEISMIC_DIM_H;
    const u64 stream = blockIdx.x;
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const float *key = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_KEY));
    const float *hidden = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN));
    const float *value = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_VALUE));
    const u8 *key_norm = SEISMIC_PTR(SEISMIC_BUFFER_KEY_NORM);
    const u8 *query_norm = SEISMIC_PTR(SEISMIC_BUFFER_QUERY_NORM);
    float *scales = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    for (u64 row = blockIdx.y; row < SEISMIC_DIM_M; row += gridDim.y) {
        const float *k = key + row * SEISMIC_KEY_STRIDE_0 + stream * SEISMIC_KEY_STRIDE_1;
        const float *x = hidden + row * SEISMIC_HIDDEN_STRIDE_0 + stream * SEISMIC_HIDDEN_STRIDE_1;
        const float *v = value + row * SEISMIC_VALUE_STRIDE_0;
        float keys = 0.0f;
        float queries = 0.0f;
        float dot = 0.0f;
        float values = 0.0f;
        for (u64 i = threadIdx.x; i < width; i += blockDim.x) {
            const float kv = k[i * SEISMIC_KEY_STRIDE_2];
            const float xv = x[i * SEISMIC_HIDDEN_STRIDE_2];
            const float vv = v[i * SEISMIC_VALUE_STRIDE_1];
            keys = seismic_fma_rn(kv, kv, keys);
            queries = seismic_fma_rn(xv, xv, queries);
            dot += kv * element::at<KeyNorm>(key_norm, stream * SEISMIC_KEY_NORM_STRIDE_0 + i * SEISMIC_KEY_NORM_STRIDE_1) *
                   xv * element::at<QueryNorm>(query_norm, stream * SEISMIC_QUERY_NORM_STRIDE_0 + i * SEISMIC_QUERY_NORM_STRIDE_1);
            values = seismic_fma_rn(vv, vv, values);
        }
        keys = reduce::group_sum(keys, partials);
        queries = reduce::group_sum(queries, partials);
        dot = reduce::group_sum(dot, partials);
        values = reduce::group_sum(values, partials);
        if (threadIdx.x == 0) {
            const float g = dot * rsqrtf(keys / (float)width + epsilon) * rsqrtf(queries / (float)width + epsilon) /
                            sqrtf((float)width);
            const float root = sqrtf(fmaxf(fabsf(g), 1e-6f));
            const float signed_root = g > 0.0f ? root : (g < 0.0f ? -root : 0.0f);
            const float gate = 1.0f / (1.0f + expf(-signed_root));
            float *out = scales + row * SEISMIC_RESULT_0_STRIDE_0;
            out[stream * SEISMIC_RESULT_0_STRIDE_1] = gate;
            out[(SEISMIC_DIM_S + stream) * SEISMIC_RESULT_0_STRIDE_1] =
                rsqrtf(gate * gate * (values / (float)width) + epsilon);
        }
    }
}
