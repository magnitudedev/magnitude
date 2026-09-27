// draft_confidence: one block per row reduces weight · [features ; memory] +
// bias and marks the row's selection declined (status 3) when its sigmoid is
// below the threshold.
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

extern "C" __global__ void draft_confidence(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    __shared__ float partials[32];
    const element::u8 *features = SEISMIC_PTR(SEISMIC_BUFFER_FEATURES);
    const element::u8 *memory = SEISMIC_PTR(SEISMIC_BUFFER_MEMORY);
    const float *weight = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT));
    const float *bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BIAS));
    int *selection = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTION));
    const u64 row = blockIdx.x;
    const u64 width = SEISMIC_DIM_D, rank = SEISMIC_DIM_R;
    float partial = 0.0f;
    for (u64 i = threadIdx.x; i < width; i += blockDim.x)
        partial = seismic_fma_rn(
            element::at<element::Act>(features, row * SEISMIC_FEATURES_STRIDE_0 + i * SEISMIC_FEATURES_STRIDE_1),
            weight[i * SEISMIC_WEIGHT_STRIDE_0], partial);
    for (u64 i = threadIdx.x; i < rank; i += blockDim.x)
        partial = seismic_fma_rn(
            element::at<element::Act>(memory, row * SEISMIC_MEMORY_STRIDE_0 + i * SEISMIC_MEMORY_STRIDE_1),
            weight[(width + i) * SEISMIC_WEIGHT_STRIDE_0], partial);
    const float score = reduce::group_sum(partial, partials) + bias[0];
    if (threadIdx.x == 0 && 1.0f / (1.0f + expf(-score)) < __uint_as_float((unsigned)SEISMIC_PARAM_THRESHOLD))
        selection[row * SEISMIC_SELECTION_STRIDE_0 + SEISMIC_SELECTION_STRIDE_1] = 3;
}
