// vision_norm: one block per result row r = cell * G + member, which
// normalizes source row r (or order[r] when NO = 1) and publishes it to Y at
// member * H (side by side) or, with `interleave`, channel by channel at
// stride G. Body in lib/vision/vision.cuh.
#include "lib/vision/vision.cuh"

using vision::u32;
using vision::u64;
typedef ELEMENT_OF(SEISMIC_ELEMENT_Y) Y;
typedef ELEMENT_OF(SEISMIC_WEIGHT) WeightElement;
typedef ELEMENT_OF(SEISMIC_BIAS) BiasElement;

extern "C" __global__ void vision_norm(SEISMIC_KERNEL_PARAMS) {
    __shared__ float partials[32];
    const u32 g = SEISMIC_DIM_G, h = SEISMIC_DIM_H;
    const u32 row = blockIdx.x;
    const int *order = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ORDER));
    const u32 from = SEISMIC_DIM_NO == 1 ? (u32)order[row] : row;
    const u32 cell = row / g, member = row % g;
    const bool interleave = (int)SEISMIC_PARAM_INTERLEAVE != 0;
    const u64 first = (u64)cell * SEISMIC_RESULT_0_STRIDE_0 + (interleave ? member : (u64)member * h);
    const float *source = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    vision::norm<Y, WeightElement, BiasElement, SEISMIC_DIM_NW == 1, SEISMIC_DIM_NB == 1>(
        source + (u64)from * h, SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER) + first * Y::bytes, interleave ? g : 1,
        SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT), SEISMIC_PTR(SEISMIC_BUFFER_BIAS), h, (int)SEISMIC_PARAM_CENTERED != 0,
        __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON), partials);
}
