// rms_concat: `[RMS(a, eps) || RMS(b, eps)]` rounded to A. One block per row.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

using element::u64;

extern "C" __global__ void rms_concat(SEISMIC_KERNEL_PARAMS) {
    __shared__ float partials[32];
    const u64 row = blockIdx.x;
    const u64 H = SEISMIC_DIM_H;
    const float eps = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const float *ar = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_A)) + row * SEISMIC_A_STRIDE_0;
    const float *br = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_B)) + row * SEISMIC_B_STRIDE_0;
    float sq_a = 0.0f;
    float sq_b = 0.0f;
    for (u64 i = threadIdx.x; i < H; i += blockDim.x) {
        const float va = ar[i * SEISMIC_A_STRIDE_1];
        const float vb = br[i * SEISMIC_B_STRIDE_1];
        sq_a = seismic_fma_rn(va, va, sq_a);
        sq_b = seismic_fma_rn(vb, vb, sq_b);
    }
    const float inv_a = rsqrtf(reduce::group_sum(sq_a, partials) / (float)H + eps);
    const float inv_b = rsqrtf(reduce::group_sum(sq_b, partials) / (float)H + eps);
    const auto result = SEISMIC_RESULT_0_OUT.offset(row * SEISMIC_RESULT_0_STRIDE_0);
    for (u64 i = threadIdx.x; i < H; i += blockDim.x) {
        result.put(i * SEISMIC_RESULT_0_STRIDE_1, ar[i * SEISMIC_A_STRIDE_1] * inv_a);
        result.put((H + i) * SEISMIC_RESULT_0_STRIDE_1, br[i * SEISMIC_B_STRIDE_1] * inv_b);
    }
}
