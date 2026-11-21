// corrector_features: one block of 256 threads per proposing row reduces the
// row's softmax normalizer (its maximum plus the log of the shifted sum), then
// forms each of its K candidates' five features and their layer norm (every
// thread, in registers) and the H outputs silu(up · normed + up_bias), a
// thread per column, rounded once to A.
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

extern "C" __global__ void corrector_features(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    __shared__ reduce::OnlineSoftmax partials[32];
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    const float *unary = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_UNARY));
    const float *norm_weight = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_NORM_WEIGHT));
    const float *norm_bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_NORM_BIAS));
    const float *up_weight = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_UP_WEIGHT));
    const float *up_bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_UP_BIAS));
    const auto result = SEISMIC_RESULT_0_OUT;
    const u64 row = blockIdx.x;
    const u64 vocabulary = SEISMIC_DIM_V, count = SEISMIC_DIM_K, width = SEISMIC_DIM_H;
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const float *line = logits + row * SEISMIC_LOGITS_STRIDE_0;

    reduce::OnlineSoftmax state = reduce::OnlineSoftmax::empty();
    if (SEISMIC_LOGITS_STRIDE_1 == 1ull && (reinterpret_cast<u64>(line) & 3ull) == 0ull) {
        const u64 misalign = (reinterpret_cast<u64>(line) & 15ull) >> 2;
        const u64 head = misalign == 0ull ? 0ull : min(4ull - misalign, vocabulary);
        if (threadIdx.x < head)
            state.add(line[threadIdx.x]);
        const float *aligned = line + head;
        const u64 vec_count = (vocabulary - head) & ~3ull;
        const u64 step4 = static_cast<u64>(blockDim.x) * 4ull;
        for (u64 i = static_cast<u64>(threadIdx.x) * 4ull; i < vec_count; i += step4) {
            const float4 v = *reinterpret_cast<const float4 *>(aligned + i);
            state.add(v.x);
            state.add(v.y);
            state.add(v.z);
            state.add(v.w);
        }
        for (u64 i = head + vec_count + threadIdx.x; i < vocabulary; i += blockDim.x)
            state.add(line[i]);
    } else {
        for (u64 i = threadIdx.x; i < vocabulary; i += blockDim.x)
            state.add(line[i * SEISMIC_LOGITS_STRIDE_1]);
    }
    const reduce::OnlineSoftmax softmax = reduce::group_online_softmax(state, partials);
    const float normalizer = softmax.max + logf(softmax.sum);

    const float best = unary[row * SEISMIC_UNARY_STRIDE_0];
    for (u64 rank = 0; rank < count; ++rank) {
        const float logit = unary[row * SEISMIC_UNARY_STRIDE_0 + rank * SEISMIC_UNARY_STRIDE_1];
        const float lp = logit - normalizer;
        const float features[5] = {
            lp,
            expf(lp),
            logit - best,
            count > 1 ? float(rank) / float(count - 1) : 0.0f,
            rank == 0 ? 1.0f : 0.0f,
        };
        const float mean = (features[0] + features[1] + features[2] + features[3] + features[4]) / 5.0f;
        float squares = 0.0f;
#pragma unroll
        for (int f = 0; f < 5; ++f)
            squares += (features[f] - mean) * (features[f] - mean);
        const float inverse = 1.0f / sqrtf(squares / 5.0f + epsilon);
        float normed[5];
#pragma unroll
        for (int f = 0; f < 5; ++f)
            normed[f] = (features[f] - mean) * inverse * norm_weight[f * SEISMIC_NORM_WEIGHT_STRIDE_0] +
                        norm_bias[f * SEISMIC_NORM_BIAS_STRIDE_0];
        const u64 out = row * count + rank;
        for (u64 column = threadIdx.x; column < width; column += blockDim.x) {
            float value = up_bias[column * SEISMIC_UP_BIAS_STRIDE_0];
#pragma unroll
            for (int f = 0; f < 5; ++f)
                value = seismic_fma_rn(
                    up_weight[column * SEISMIC_UP_WEIGHT_STRIDE_0 + f * SEISMIC_UP_WEIGHT_STRIDE_1], normed[f], value);
            result.put(out * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1,
                       value / (1.0f + expf(-value)));
        }
    }
}
