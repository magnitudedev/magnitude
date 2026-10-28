// vision_gated_convolution: one thread per value, the taps summed in tap
// order in F32 over the neighbours present, published to A, then times the
// gate and published again.
#include "lib/core/activation.cuh"

using element::u64;
typedef element::Act A;
typedef ELEMENT_OF(SEISMIC_WEIGHT) Weight;

extern "C" __global__ void vision_gated_convolution(SEISMIC_KERNEL_PARAMS) {
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 columns = SEISMIC_DIM_H;
    if (index >= SEISMIC_DIM_M * columns)
        return;
    const u64 row = index / columns, channel = index % columns;
    const int *neighbors = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_NEIGHBORS));
    float sum = 0.0f;
#pragma unroll
    for (u64 tap = 0; tap < SEISMIC_DIM_T; ++tap) {
        const int source = neighbors[row * SEISMIC_NEIGHBORS_STRIDE_0 + tap];
        if (source >= 0)
            sum = fmaf(element::at<A>(SEISMIC_PTR(SEISMIC_BUFFER_VALUE), (u64)source * SEISMIC_VALUE_STRIDE_0 + channel),
                       element::at<Weight>(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT), channel * SEISMIC_WEIGHT_STRIDE_0 + tap),
                       sum);
    }
    const float convolved = A::round(sum);
    element::put<A>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), row * SEISMIC_RESULT_0_STRIDE_0 + channel,
                    convolved * element::at<A>(SEISMIC_PTR(SEISMIC_BUFFER_GATE), row * SEISMIC_GATE_STRIDE_0 + channel));
}
