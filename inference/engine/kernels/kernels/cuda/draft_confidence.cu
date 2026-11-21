// draft_confidence: one block per row reduces weight · [features ; memory] +
// bias and marks the row's selection declined (status 3) when its sigmoid is
// below the threshold.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

using Act = activation::Act;
using element::u64;
using element::u8;

__device__ __forceinline__ float dot_accumulate(
    const u8 *act_row, u64 act_stride, const float *w, u64 w_stride, u64 count, float partial) {
    const bool vec = act_stride == 1ull &&
                     w_stride == 1ull &&
                     (reinterpret_cast<u64>(act_row) & (4ull * Act::bytes - 1ull)) == 0ull &&
                     (reinterpret_cast<u64>(w) & 15ull) == 0ull;
    if (vec) {
        const u64 count4 = count & ~3ull;
        const u64 step4 = static_cast<u64>(blockDim.x) * 4ull;
        for (u64 i = static_cast<u64>(threadIdx.x) * 4ull; i < count4; i += step4) {
            float a[4];
            element::span<Act, 4, true>(act_row, i, a);
            const float4 wv = *reinterpret_cast<const float4 *>(w + i);
            partial = seismic_fma_rn(a[0], wv.x, partial);
            partial = seismic_fma_rn(a[1], wv.y, partial);
            partial = seismic_fma_rn(a[2], wv.z, partial);
            partial = seismic_fma_rn(a[3], wv.w, partial);
        }
        for (u64 i = count4 + threadIdx.x; i < count; i += blockDim.x)
            partial = seismic_fma_rn(element::at<Act>(act_row, i), w[i], partial);
    } else {
        for (u64 i = threadIdx.x; i < count; i += blockDim.x)
            partial = seismic_fma_rn(element::at<Act>(act_row, i * act_stride), w[i * w_stride], partial);
    }
    return partial;
}

extern "C" __global__ void draft_confidence(SEISMIC_KERNEL_PARAMS) {
    __shared__ float partials[32];
    const u8 *features = SEISMIC_PTR(SEISMIC_BUFFER_FEATURES);
    const u8 *memory = SEISMIC_PTR(SEISMIC_BUFFER_MEMORY);
    const float *weight = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT));
    const float *bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BIAS));
    int *selection = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTION));
    const u64 row = blockIdx.x;
    const u64 width = SEISMIC_DIM_D, rank = SEISMIC_DIM_R;
    const u8 *features_row = features + row * SEISMIC_FEATURES_STRIDE_0 * Act::bytes;
    const u8 *memory_row = memory + row * SEISMIC_MEMORY_STRIDE_0 * Act::bytes;
    float partial = 0.0f;
    partial = dot_accumulate(features_row, SEISMIC_FEATURES_STRIDE_1, weight, SEISMIC_WEIGHT_STRIDE_0, width, partial);
    partial = dot_accumulate(memory_row, SEISMIC_MEMORY_STRIDE_1, weight + width * SEISMIC_WEIGHT_STRIDE_0,
        SEISMIC_WEIGHT_STRIDE_0, rank, partial);
    const float score = reduce::group_sum(partial, partials) + bias[0];
    if (threadIdx.x == 0 && 1.0f / (1.0f + expf(-score)) < __uint_as_float((unsigned)SEISMIC_PARAM_THRESHOLD))
        selection[row * SEISMIC_SELECTION_STRIDE_0 + SEISMIC_SELECTION_STRIDE_1] = 3;
}
