// corrector_attention: one block of 256 threads per (node, head). Each warp
// scores a strided set of the block's keys (its lanes split the head's
// columns of the biased query and key, then a warp sum), plus the head's
// additive bias; the softmax weights stay in shared memory, and a thread per
// column accumulates the biased values in key order, divides by the total and
// rounds once to A. Dynamic shared memory: the N scores.
#include "lib/core/activation.cuh"

extern "C" __global__ void corrector_attention(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    extern __shared__ float shared[];
    float *scores = shared;  // [N]
    const float *qkv = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_QKV));
    const float *input_bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INPUT_BIAS));
    const float *bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BIAS));
    element::u8 *result = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    const u64 nodes = SEISMIC_DIM_N, width = SEISMIC_DIM_W;
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
    float maximum = -__int_as_float(0x7f800000);
    for (u64 key = 0; key < nodes; ++key)
        maximum = fmaxf(maximum, scores[key]);
    __syncthreads();
    for (u64 key = threadIdx.x; key < nodes; key += blockDim.x)
        scores[key] = expf(scores[key] - maximum);
    __syncthreads();
    float total = 0.0f;
    for (u64 key = 0; key < nodes; ++key)
        total += scores[key];
    for (u64 column = threadIdx.x; column < width; column += blockDim.x) {
        float out = 0.0f;
        for (u64 key = 0; key < nodes; ++key)
            out = seismic_fma_rn(scores[key], projected(key, 2, column), out);
        element::put<element::Act>(result,
                                   block * SEISMIC_RESULT_0_STRIDE_0 + query * SEISMIC_RESULT_0_STRIDE_1 +
                                       head * SEISMIC_RESULT_0_STRIDE_2 + column * SEISMIC_RESULT_0_STRIDE_3,
                                   out / total);
    }
}
