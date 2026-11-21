// corrector_attention: one block of 256 threads per (node, head). Each warp
// scores a strided set of the block's keys (its lanes split the head's
// columns of the biased query and key, then a warp sum), plus the head's
// additive bias; the block reduces their maximum and softmax denominator in
// parallel, keeping the weights in shared memory, and a thread per column
// accumulates the biased values in key order, multiplies by the inverse total,
// and rounds once to A. Dynamic shared memory: the N scores followed by 64
// bytes of reduction partials.
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

extern "C" __global__ void corrector_attention(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *qkv = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_QKV));
    const float *input_bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INPUT_BIAS));
    const float *bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BIAS));
    const auto result = SEISMIC_RESULT_0_OUT;
    const u64 nodes = SEISMIC_DIM_N, width = SEISMIC_DIM_W;
    extern __shared__ float shared[];
    float *scores = shared;           // [N]
    float *partials = shared + nodes; // [16] (64 bytes, >= warps)
    const u64 block = blockIdx.x / nodes, query = blockIdx.x % nodes, head = blockIdx.y;
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
    // Element (node, part, column) of the head's biased projections.
    auto projected = [&](u64 node, u64 part, u64 column) {
        return qkv[block * SEISMIC_QKV_STRIDE_0 + node * SEISMIC_QKV_STRIDE_1 + part * SEISMIC_QKV_STRIDE_2 +
                   head * SEISMIC_QKV_STRIDE_3 + column * SEISMIC_QKV_STRIDE_4] +
               input_bias[part * SEISMIC_INPUT_BIAS_STRIDE_0 + head * SEISMIC_INPUT_BIAS_STRIDE_1 +
                          column * SEISMIC_INPUT_BIAS_STRIDE_2];
    };
    const float scale = 1.0f / sqrtf(float(width));
    for (u64 key = warp; key < nodes; key += warps) {
        float dot = 0.0f;
        for (u64 column = lane; column < width; column += 32)
            dot = seismic_fma_rn(projected(query, 0, column), projected(key, 1, column), dot);
        dot = seismic_warp_sum_f32(dot);
        if (lane == 0)
            scores[key] = dot * scale + bias[head * SEISMIC_BIAS_STRIDE_0 + query * SEISMIC_BIAS_STRIDE_1 +
                                             key * SEISMIC_BIAS_STRIDE_2];
    }
    __syncthreads();
    float local_max = -__int_as_float(0x7f800000);
    for (u64 key = threadIdx.x; key < nodes; key += blockDim.x)
        local_max = fmaxf(local_max, scores[key]);
    const float maximum = reduce::group_max(local_max, partials);
    float local_sum = 0.0f;
    for (u64 key = threadIdx.x; key < nodes; key += blockDim.x) {
        const float weight = expf(scores[key] - maximum);
        scores[key] = weight;
        local_sum += weight;
    }
    // The sum's barriers also publish every weight to the block.
    const float total = reduce::group_sum(local_sum, partials);
    const float inv = total > 0.0f ? 1.0f / total : 0.0f;
    const float *values = qkv + block * SEISMIC_QKV_STRIDE_0 + 2 * SEISMIC_QKV_STRIDE_2 + head * SEISMIC_QKV_STRIDE_3;
    const float *value_bias = input_bias + 2 * SEISMIC_INPUT_BIAS_STRIDE_0 + head * SEISMIC_INPUT_BIAS_STRIDE_1;
    const u64 out_base =
        block * SEISMIC_RESULT_0_STRIDE_0 + query * SEISMIC_RESULT_0_STRIDE_1 + head * SEISMIC_RESULT_0_STRIDE_2;
    for (u64 column = threadIdx.x; column < width; column += blockDim.x) {
        const float vb = value_bias[column * SEISMIC_INPUT_BIAS_STRIDE_2];
        const float *v_col = values + column * SEISMIC_QKV_STRIDE_4;
        float out = 0.0f;
        for (u64 key = 0; key < nodes; ++key)
            out = seismic_fma_rn(scores[key], v_col[key * SEISMIC_QKV_STRIDE_1] + vb, out);
        result.put(out_base + column * SEISMIC_RESULT_0_STRIDE_3, out * inv);
    }
}
