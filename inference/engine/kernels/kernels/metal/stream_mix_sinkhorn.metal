// stream_mix_sinkhorn (contract and portable body in streams.seismic): one
// threadgroup of 256 threads per row. Each thread accumulates, over its
// strided share of the row's S * H flattened stream values, their squares
// and every function row's products; the threadgroup reduces them in fixed
// order, and the first thread forms and publishes pre, post and the
// Sinkhorn comb from the reduced mixes. The threads mix the streams by the
// previous site's pre column by column (`streams::collapse`).
#include "lib/core/functions.h"
#include "lib/core/reduce.h"
#include "lib/core/streams.h"

constant constexpr uint STREAMS = uint(SEISMIC_DIM_S);
constant constexpr uint MIXES = (2u + STREAMS) * STREAMS;
constant constexpr uint THREADS = 256u;

kernel void stream_mix_sinkhorn(
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const float *previous [[buffer(SEISMIC_BUFFER_PREVIOUS)]],
    device const float *function [[buffer(SEISMIC_BUFFER_FUNCTION)]],
    device const float *base [[buffer(SEISMIC_BUFFER_BASE)]],
    device const float *scale [[buffer(SEISMIC_BUFFER_SCALE)]],
    device float *input [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *pre [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    device float *join [[buffer(SEISMIC_RESULT_2_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[THREADS / 32u];
    const uint h = uint(SEISMIC_DIM_H), width = STREAMS * h;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float norm_epsilon = as_type<float>(uint(SEISMIC_PARAM_NORM_EPSILON));
    const uint iterations = uint(SEISMIC_PARAM_ITERATIONS);
    device const float *x = hidden + ulong(row) * SEISMIC_HIDDEN_STRIDE_0;
    float squares = 0.0f;
    float mixes[MIXES];
    for (uint j = 0; j < MIXES; ++j)
        mixes[j] = 0.0f;
    for (uint at = thread_index; at < width; at += THREADS) {
        const uint stream = at / h, column = at % h;
        const float v = x[ulong(stream) * SEISMIC_HIDDEN_STRIDE_1 + ulong(column) * SEISMIC_HIDDEN_STRIDE_2];
        squares = metal::fma(v, v, squares);
        for (uint j = 0; j < MIXES; ++j)
            mixes[j] = metal::fma(function[ulong(j) * SEISMIC_FUNCTION_STRIDE_0 + ulong(at) * SEISMIC_FUNCTION_STRIDE_1],
                v, mixes[j]);
    }
    const float inverse = metal::rsqrt(reduce::group_sum<THREADS / 32u>(squares, partials, sg, lane) / float(width)
        + norm_epsilon);
    for (uint j = 0; j < MIXES; ++j)
        mixes[j] = reduce::group_sum<THREADS / 32u>(mixes[j], partials, sg, lane) * inverse;
    if (thread_index == 0u) {
        // The softmax rows of comb, then its Sinkhorn normalizations.
        float comb[STREAMS][STREAMS];
        for (uint source = 0; source < STREAMS; ++source) {
            float maximum = -INFINITY;
            for (uint target = 0; target < STREAMS; ++target) {
                const uint at = 2u * STREAMS + source * STREAMS + target;
                comb[source][target] = mixes[at] * scale[2u * SEISMIC_SCALE_STRIDE_0] + base[ulong(at) * SEISMIC_BASE_STRIDE_0];
                maximum = metal::max(maximum, comb[source][target]);
            }
            float total = 0.0f;
            for (uint target = 0; target < STREAMS; ++target) {
                comb[source][target] = metal::exp(comb[source][target] - maximum);
                total += comb[source][target];
            }
            for (uint target = 0; target < STREAMS; ++target)
                comb[source][target] = comb[source][target] / total + epsilon;
        }
        for (uint sweep = 0; sweep < iterations; ++sweep) {
            if (sweep > 0u) {
                for (uint source = 0; source < STREAMS; ++source) {
                    float sum = 0.0f;
                    for (uint target = 0; target < STREAMS; ++target)
                        sum += comb[source][target];
                    for (uint target = 0; target < STREAMS; ++target)
                        comb[source][target] /= sum + epsilon;
                }
            }
            for (uint target = 0; target < STREAMS; ++target) {
                float sum = 0.0f;
                for (uint source = 0; source < STREAMS; ++source)
                    sum += comb[source][target];
                for (uint source = 0; source < STREAMS; ++source)
                    comb[source][target] /= sum + epsilon;
            }
        }
        device float *own = pre + ulong(row) * SEISMIC_RESULT_1_STRIDE_0;
        device float *joined = join + ulong(row) * SEISMIC_RESULT_2_STRIDE_0;
        for (uint stream = 0; stream < STREAMS; ++stream) {
            own[ulong(stream) * SEISMIC_RESULT_1_STRIDE_1] =
                functions::sigmoid(mixes[stream] * scale[0] + base[ulong(stream) * SEISMIC_BASE_STRIDE_0]) + epsilon;
            joined[ulong(stream) * SEISMIC_RESULT_2_STRIDE_1] = 2.0f
                * functions::sigmoid(mixes[STREAMS + stream] * scale[SEISMIC_SCALE_STRIDE_0]
                    + base[ulong(STREAMS + stream) * SEISMIC_BASE_STRIDE_0]);
        }
        for (uint source = 0; source < STREAMS; ++source)
            for (uint target = 0; target < STREAMS; ++target)
                joined[ulong(STREAMS + source * STREAMS + target) * SEISMIC_RESULT_2_STRIDE_1] = comb[source][target];
    }
    // The sublayer's input: the streams mixed by the previous site's pre.
    streams::collapse<STREAMS>(x, SEISMIC_HIDDEN_STRIDE_1, SEISMIC_HIDDEN_STRIDE_2,
        previous + ulong(row) * SEISMIC_PREVIOUS_STRIDE_0, SEISMIC_PREVIOUS_STRIDE_1,
        input + ulong(row) * SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, h, thread_index, THREADS);
}
