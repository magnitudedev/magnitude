// corrector_rows: one block of 256 threads per row: values + bias, then by
// form the identity, silu or unit length (the row's sum of squares reduced
// over the block), plus the residual row when R = 1, published to Y.
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

typedef ELEMENT_OF(SEISMIC_ELEMENT_Y) Y;

extern "C" __global__ void corrector_rows(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    __shared__ float partials[32];
    const float *values = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_VALUES));
    const float *bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BIAS));
    const float *residual = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    element::u8 *result = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    const u64 row = blockIdx.x;
    const u64 width = SEISMIC_DIM_N;
    const int form = (int)SEISMIC_PARAM_FORM;
    auto biased = [&](u64 column) {
        return values[row * SEISMIC_VALUES_STRIDE_0 + column * SEISMIC_VALUES_STRIDE_1] +
               bias[column * SEISMIC_BIAS_STRIDE_0];
    };
    // Form 2's length, max(sqrt(sum of squares), epsilon).
    float length = 1.0f;
    if (form == 2) {
        float squares = 0.0f;
        for (u64 column = threadIdx.x; column < width; column += blockDim.x) {
            const float value = biased(column);
            squares = seismic_fma_rn(value, value, squares);
        }
        length = fmaxf(sqrtf(reduce::group_sum(squares, partials)), __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON));
    }
    for (u64 column = threadIdx.x; column < width; column += blockDim.x) {
        float value = biased(column);
        if (form == 1)
            value = value / (1.0f + expf(-value));
        else if (form == 2)
            value = value / length;
        if (SEISMIC_DIM_R == 1)
            value += residual[row * SEISMIC_RESIDUAL_STRIDE_1 + column * SEISMIC_RESIDUAL_STRIDE_2];
        element::put<Y>(result, row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1, value);
    }
}
