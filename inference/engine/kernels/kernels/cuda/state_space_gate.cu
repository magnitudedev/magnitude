// state_space_gate (contract and portable body in state_space.seismic): the
// state-space output rows, y * SiLU(z) RMS-normalized over each group of U
// heads and scaled by the state norm. The CUDA form of
// `metal/state_space_gate.metal`: a block of 256 threads per (row, group),
// strided lane chains, a warp reduction, then the warp sums in order.

#include "lib/core/activation.cuh"

namespace state_space_gate_detail {

typedef unsigned long long u64;

// The gated value of `channel` (in the row's flat head-major order).
__device__ __forceinline__ float gated_value(const element::u8 *mixed, const element::u8 *projection, u64 row,
                                             u64 channel, const seismic_words_t &seismic_words_value) {
    const u64 head = channel / SEISMIC_DIM_P;
    const float gate = element::at<element::Act>(projection, row * SEISMIC_PROJECTION_STRIDE_0 +
                                                                 channel * SEISMIC_PROJECTION_STRIDE_1);
    const float value = element::at<element::Act>(mixed, row * SEISMIC_MIXED_STRIDE_0 + head * SEISMIC_MIXED_STRIDE_1 +
                                                             (channel % SEISMIC_DIM_P) * SEISMIC_MIXED_STRIDE_2);
    return seismic_mul_rn(value, gate / (1.0f + expf(-gate)));
}

} // namespace state_space_gate_detail

extern "C" __global__ void state_space_gate(SEISMIC_KERNEL_PARAMS) {
    using namespace state_space_gate_detail;
    const element::u8 *mixed = SEISMIC_PTR(SEISMIC_BUFFER_MIXED);
    const element::u8 *projection = SEISMIC_PTR(SEISMIC_BUFFER_PROJECTION);
    const float *state_norm = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_STATE_NORM));
    element::u8 *normalized = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    __shared__ float partials[32];
    const u64 row = blockIdx.x;
    const u64 width = SEISMIC_DIM_U * SEISMIC_DIM_P;
    const u64 first = static_cast<u64>(blockIdx.y) * width;
    const float epsilon = element::word_f32(SEISMIC_PARAM_EPSILON);
    float squares = 0.0f;
    for (u64 column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = gated_value(mixed, projection, row, first + column, seismic_words_value);
        squares = seismic_fma_rn(value, value, squares);
    }
    const float total = seismic_warp_sum_f32(squares);
    if (threadIdx.x % 32 == 0)
        partials[threadIdx.x / 32] = total;
    __syncthreads();
    float sum = 0.0f;
    for (unsigned index = 0; index < blockDim.x / 32; ++index) sum = seismic_add_rn(sum, partials[index]);
    const float inverse = rsqrtf(sum / static_cast<float>(width) + epsilon);
    for (u64 column = threadIdx.x; column < width; column += blockDim.x) {
        const u64 channel = first + column;
        const float weight = state_norm[static_cast<u64>(blockIdx.y) * SEISMIC_STATE_NORM_STRIDE_0 +
                                        (column / SEISMIC_DIM_P) * SEISMIC_STATE_NORM_STRIDE_1 +
                                        (column % SEISMIC_DIM_P) * SEISMIC_STATE_NORM_STRIDE_2];
        const float value = gated_value(mixed, projection, row, channel, seismic_words_value);
        element::put<element::Act>(normalized,
                                   row * SEISMIC_RESULT_0_STRIDE_0 + (channel / SEISMIC_DIM_P) * SEISMIC_RESULT_0_STRIDE_1 +
                                       (channel % SEISMIC_DIM_P) * SEISMIC_RESULT_0_STRIDE_2,
                                   seismic_mul_rn(seismic_mul_rn(value, inverse), weight));
    }
}
