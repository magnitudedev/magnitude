// post_norm_residual: a sandwich-norm tail over the `out_rows` rows,
// (residual + rms(projected) * norm) * scale, all F32. One block per output
// row: the square sum of the projected row, then the update.
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

using element::u64;
using element::u8;

extern "C" __global__ void post_norm_residual(SEISMIC_KERNEL_PARAMS) {
    using Norm = ELEMENT_OF(SEISMIC_NORM);
    __shared__ float partials[32];
    const u64 row = blockIdx.x;
    const u64 width = SEISMIC_DIM_D;
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const float scale = __uint_as_float((unsigned)SEISMIC_PARAM_SCALE);
    const float *p = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PROJECTED)) + row * SEISMIC_PROJECTED_STRIDE_0;
    float squares = 0.0f;
    for (u64 i = threadIdx.x; i < width; i += blockDim.x) {
        const float v = p[i * SEISMIC_PROJECTED_STRIDE_1];
        squares = seismic_fma_rn(v, v, squares);
    }
    const float total = reduce::group_sum(squares, partials);
    const float inverse = rsqrtf(total / (float)width + epsilon);
    const int *out_rows = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT_ROWS));
    const float *r = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL))
        + (u64)out_rows[row * SEISMIC_OUT_ROWS_STRIDE_0] * SEISMIC_RESIDUAL_STRIDE_0;
    const u8 *norm = SEISMIC_PTR(SEISMIC_BUFFER_NORM);
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)) + row * SEISMIC_RESULT_0_STRIDE_0;
    for (u64 i = threadIdx.x; i < width; i += blockDim.x) {
        const float normalized = p[i * SEISMIC_PROJECTED_STRIDE_1] * inverse
            * element::at<Norm>(norm, i * SEISMIC_NORM_STRIDE_0);
        result[i * SEISMIC_RESULT_0_STRIDE_1] = (r[i * SEISMIC_RESIDUAL_STRIDE_1] + normalized) * scale;
    }
}
