// vision_clamp: one thread per element, A(clamp(x, minimum, maximum)).
#include "lib/core/activation.cuh"

typedef element::Act A;

extern "C" __global__ void vision_clamp(SEISMIC_KERNEL_PARAMS) {
    const element::u64 index = (element::u64)blockIdx.x * blockDim.x + threadIdx.x;
    const element::u64 columns = SEISMIC_DIM_N;
    if (index >= SEISMIC_DIM_M * columns)
        return;
    const element::u64 row = index / columns, column = index % columns;
    const float minimum = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_MINIMUM))[0];
    const float maximum = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_MAXIMUM))[0];
    const float value = element::at<A>(SEISMIC_PTR(SEISMIC_BUFFER_X), row * SEISMIC_X_STRIDE_0 + column);
    element::put<A>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), row * SEISMIC_RESULT_0_STRIDE_0 + column,
                    fminf(fmaxf(value, minimum), maximum));
}
