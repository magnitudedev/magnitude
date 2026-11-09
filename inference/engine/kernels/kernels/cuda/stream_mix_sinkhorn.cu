// stream_mix_sinkhorn (contract and portable body in streams.seismic): one
// block of 256 threads per row. Each thread accumulates, over its strided
// share of the row's S * H flattened stream values, their squares and every
// function row's products; the block reduces them in fixed order, and the
// first thread forms and publishes pre, post and the Sinkhorn comb from the
// reduced mixes. The threads mix the streams by the previous site's pre
// column by column.
#include "lib/core/functions.cuh"
#include "lib/core/reduce.cuh"
#include "lib/core/streams.cuh"

constexpr unsigned STREAMS = SEISMIC_DIM_S;
constexpr unsigned MIXES = (2u + STREAMS) * STREAMS;
constexpr unsigned THREADS = 256u;

extern "C" __global__ void stream_mix_sinkhorn(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    __shared__ float partials[THREADS / 32u];
    const float *hidden = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN));
    const float *previous = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PREVIOUS));
    const float *function = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_FUNCTION));
    const float *base = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BASE));
    const float *scale = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALE));
    float *input = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    float *pre = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER));
    float *join = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_2_BUFFER));
    const unsigned row = blockIdx.x;
    const unsigned h = SEISMIC_DIM_H, width = STREAMS * h;
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const float norm_epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_NORM_EPSILON);
    const unsigned iterations = (unsigned)SEISMIC_PARAM_ITERATIONS;
    const float *x = hidden + (u64)row * SEISMIC_HIDDEN_STRIDE_0;
    float squares = 0.0f;
    float mixes[MIXES];
#pragma unroll
    for (unsigned j = 0; j < MIXES; ++j)
        mixes[j] = 0.0f;
    for (unsigned at = threadIdx.x; at < width; at += THREADS) {
        const unsigned stream = at / h, column = at % h;
        const float v = x[(u64)stream * SEISMIC_HIDDEN_STRIDE_1 + (u64)column * SEISMIC_HIDDEN_STRIDE_2];
        squares = __fmaf_rn(v, v, squares);
#pragma unroll
        for (unsigned j = 0; j < MIXES; ++j)
            mixes[j] = __fmaf_rn(function[(u64)j * SEISMIC_FUNCTION_STRIDE_0 + (u64)at * SEISMIC_FUNCTION_STRIDE_1],
                                 v, mixes[j]);
    }
    const float inverse = rsqrtf(reduce::group_sum(squares, partials) / (float)width + norm_epsilon);
#pragma unroll
    for (unsigned j = 0; j < MIXES; ++j)
        mixes[j] = reduce::group_sum(mixes[j], partials) * inverse;
    if (threadIdx.x == 0u) {
        // The softmax rows of comb, then its Sinkhorn normalizations.
        float comb[STREAMS][STREAMS];
        for (unsigned source = 0; source < STREAMS; ++source) {
            float maximum = -__int_as_float(0x7f800000);
            for (unsigned target = 0; target < STREAMS; ++target) {
                const unsigned at = 2u * STREAMS + source * STREAMS + target;
                comb[source][target] = mixes[at] * scale[2u * SEISMIC_SCALE_STRIDE_0] + base[(u64)at * SEISMIC_BASE_STRIDE_0];
                maximum = fmaxf(maximum, comb[source][target]);
            }
            float total = 0.0f;
            for (unsigned target = 0; target < STREAMS; ++target) {
                comb[source][target] = expf(comb[source][target] - maximum);
                total += comb[source][target];
            }
            for (unsigned target = 0; target < STREAMS; ++target)
                comb[source][target] = comb[source][target] / total + epsilon;
        }
        for (unsigned sweep = 0; sweep < iterations; ++sweep) {
            if (sweep > 0u) {
                for (unsigned source = 0; source < STREAMS; ++source) {
                    float sum = 0.0f;
                    for (unsigned target = 0; target < STREAMS; ++target)
                        sum += comb[source][target];
                    for (unsigned target = 0; target < STREAMS; ++target)
                        comb[source][target] /= sum + epsilon;
                }
            }
            for (unsigned target = 0; target < STREAMS; ++target) {
                float sum = 0.0f;
                for (unsigned source = 0; source < STREAMS; ++source)
                    sum += comb[source][target];
                for (unsigned source = 0; source < STREAMS; ++source)
                    comb[source][target] /= sum + epsilon;
            }
        }
        float *own = pre + (u64)row * SEISMIC_RESULT_1_STRIDE_0;
        float *joined = join + (u64)row * SEISMIC_RESULT_2_STRIDE_0;
        for (unsigned stream = 0; stream < STREAMS; ++stream) {
            own[(u64)stream * SEISMIC_RESULT_1_STRIDE_1] =
                functions::sigmoid(mixes[stream] * scale[0] + base[(u64)stream * SEISMIC_BASE_STRIDE_0]) + epsilon;
            joined[(u64)stream * SEISMIC_RESULT_2_STRIDE_1] =
                2.0f * functions::sigmoid(mixes[STREAMS + stream] * scale[SEISMIC_SCALE_STRIDE_0] +
                               base[(u64)(STREAMS + stream) * SEISMIC_BASE_STRIDE_0]);
        }
        for (unsigned source = 0; source < STREAMS; ++source)
            for (unsigned target = 0; target < STREAMS; ++target)
                joined[(u64)(STREAMS + source * STREAMS + target) * SEISMIC_RESULT_2_STRIDE_1] = comb[source][target];
    }
    // The sublayer's input: the streams mixed by the previous site's pre.
    streams::collapse<STREAMS>(x, SEISMIC_HIDDEN_STRIDE_1, SEISMIC_HIDDEN_STRIDE_2,
                               previous + (u64)row * SEISMIC_PREVIOUS_STRIDE_0, SEISMIC_PREVIOUS_STRIDE_1,
                               input + (u64)row * SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, h, THREADS);
}
