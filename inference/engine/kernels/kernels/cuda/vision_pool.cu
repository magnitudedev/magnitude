// vision_pool: one thread per output value: the G member rows times
// `weight`, summed in order, times `scale`, then standardized with the NS
// rows.
#include <seismic/element.cuh>

using element::u64;
typedef ELEMENT_OF(SEISMIC_STANDARD_BIAS) StandardBias;
typedef ELEMENT_OF(SEISMIC_STANDARD_SCALE) StandardScale;

extern "C" __global__ void vision_pool(SEISMIC_KERNEL_PARAMS) {
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 columns = SEISMIC_DIM_H;
    if (index >= SEISMIC_DIM_M * columns)
        return;
    const u64 row = index / columns, column = index % columns;
    const float *source = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    const float weight = __uint_as_float((unsigned)SEISMIC_PARAM_WEIGHT);
    float sum = 0.0f;
    for (u64 member = 0; member < SEISMIC_DIM_G; ++member)
        sum = __fadd_rn(sum, __fmul_rn(source[row * SEISMIC_SOURCE_STRIDE_0 + member * SEISMIC_SOURCE_STRIDE_1 + column],
                                       weight));
    float value = sum * __uint_as_float((unsigned)SEISMIC_PARAM_SCALE);
    if (SEISMIC_DIM_NS == 1)
        value = (value - element::at<StandardBias>(SEISMIC_PTR(SEISMIC_BUFFER_STANDARD_BIAS), column))
                * element::at<StandardScale>(SEISMIC_PTR(SEISMIC_BUFFER_STANDARD_SCALE), column);
    reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER))[row * SEISMIC_RESULT_0_STRIDE_0 + column] = value;
}
