// moe_tail: the tail of a feed-forward with a dense branch beside the routed
// experts, all F32:
//   f = rms(dense) * dense_norm + rms(routed) * routed_norm
//   result = (residual + rms(f) * norm) * scale
// One block per row. The combined row is formed again (the same operations,
// so the same bits) by the pass that reduces it and by the pass that
// publishes, instead of being held in shared memory.
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

using element::u64;
using element::u8;
using Norm = ELEMENT_OF(SEISMIC_NORM);

// f[i] of one row: both branches normalized with their inverses and norms.
struct Combined {
    const float *dense;
    u64 dense_stride;
    const float *routed;
    u64 routed_stride;
    const u8 *dense_norm;
    u64 dense_norm_stride;
    const u8 *routed_norm;
    u64 routed_norm_stride;
    float dense_inverse, routed_inverse;
    __device__ __forceinline__ float at(u64 i) const {
        const float from_dense = dense[i * dense_stride] * dense_inverse
            * element::at<Norm>(dense_norm, i * dense_norm_stride);
        const float from_routed = routed[i * routed_stride] * routed_inverse
            * element::at<Norm>(routed_norm, i * routed_norm_stride);
        return from_dense + from_routed;
    }
};

extern "C" __global__ void moe_tail(SEISMIC_KERNEL_PARAMS) {
    __shared__ float partials[32];
    const u64 row = blockIdx.x;
    const u64 width = SEISMIC_DIM_D;
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const float scale = __uint_as_float((unsigned)SEISMIC_PARAM_SCALE);
    const float *d = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_DENSE)) + row * SEISMIC_DENSE_STRIDE_0;
    const float *r = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTED)) + row * SEISMIC_ROUTED_STRIDE_0;
    float dense_squares = 0.0f, routed_squares = 0.0f;
    for (u64 i = threadIdx.x; i < width; i += blockDim.x) {
        const float dv = d[i * SEISMIC_DENSE_STRIDE_1], rv = r[i * SEISMIC_ROUTED_STRIDE_1];
        dense_squares = seismic_fma_rn(dv, dv, dense_squares);
        routed_squares = seismic_fma_rn(rv, rv, routed_squares);
    }
    const float dense_total = reduce::group_sum(dense_squares, partials);
    const float routed_total = reduce::group_sum(routed_squares, partials);
    const Combined combined{d, SEISMIC_DENSE_STRIDE_1, r, SEISMIC_ROUTED_STRIDE_1,
        SEISMIC_PTR(SEISMIC_BUFFER_DENSE_NORM), SEISMIC_DENSE_NORM_STRIDE_0,
        SEISMIC_PTR(SEISMIC_BUFFER_ROUTED_NORM), SEISMIC_ROUTED_NORM_STRIDE_0,
        rsqrtf(dense_total / (float)width + epsilon), rsqrtf(routed_total / (float)width + epsilon)};
    float squares = 0.0f;
    for (u64 i = threadIdx.x; i < width; i += blockDim.x) {
        const float f = combined.at(i);
        squares = seismic_fma_rn(f, f, squares);
    }
    const float inverse = rsqrtf(reduce::group_sum(squares, partials) / (float)width + epsilon);
    const u8 *norm = SEISMIC_PTR(SEISMIC_BUFFER_NORM);
    const float *base = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)) + row * SEISMIC_RESIDUAL_STRIDE_0;
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)) + row * SEISMIC_RESULT_0_STRIDE_0;
    for (u64 i = threadIdx.x; i < width; i += blockDim.x) {
        const float normalized = combined.at(i) * inverse * element::at<Norm>(norm, i * SEISMIC_NORM_STRIDE_0);
        result[i * SEISMIC_RESULT_0_STRIDE_1] = (base[i * SEISMIC_RESIDUAL_STRIDE_1] + normalized) * scale;
    }
}
