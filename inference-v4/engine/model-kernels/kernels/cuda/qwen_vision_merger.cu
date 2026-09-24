// qwen_vision_merger: layer norm of each of the M * G F32 patch rows (width
// H) to A, then the up GEMM over G concatenated rows (+ bias, erf GELU) and
// the down GEMM to the decoder width D (+ bias), published in F32 as the image
// features. Bodies in common/vision.cuh.
#include "common/vision.cuh"

using vision::u32;
using vision::u64;
typedef element::Act A;
typedef ELEMENT_OF(SEISMIC_UP_WEIGHT) UpWeight;
typedef ELEMENT_OF(SEISMIC_DOWN_WEIGHT) DownWeight;
typedef ELEMENT_OF(SEISMIC_UP_BIAS) UpBias;
typedef ELEMENT_OF(SEISMIC_DOWN_BIAS) DownBias;
typedef ELEMENT_OF(SEISMIC_NORM_WEIGHT) NormWeight;
typedef ELEMENT_OF(SEISMIC_NORM_BIAS) NormBias;

constexpr u32 WIDTH = SEISMIC_DIM_G * SEISMIC_DIM_H;
#define NORMALIZED SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_NORMALIZED)
#define ACTIVATED SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ACTIVATED)

extern "C" __global__ void qwen_vision_merger_norm(SEISMIC_KERNEL_PARAMS) {
    __shared__ float partials[32];
    const float *hidden = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN));
    vision::layer_norm<A, NormWeight, NormBias>(hidden + (u64)blockIdx.x * SEISMIC_DIM_H,
                                                NORMALIZED + (u64)blockIdx.x * SEISMIC_DIM_H * A::bytes,
                                                SEISMIC_PTR(SEISMIC_BUFFER_NORM_WEIGHT),
                                                SEISMIC_PTR(SEISMIC_BUFFER_NORM_BIAS), (u32)SEISMIC_DIM_H,
                                                __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON), partials);
}

extern "C" __global__ void qwen_vision_merger_up(SEISMIC_KERNEL_PARAMS) {
    __shared__ vision::GemmShared shared;
    const vision::BiasGelu<A, UpBias, true> epilogue{ACTIVATED, WIDTH, SEISMIC_PTR(SEISMIC_BUFFER_UP_BIAS)};
    vision::gemm<A, UpWeight>(shared, NORMALIZED, WIDTH, SEISMIC_PTR(SEISMIC_BUFFER_UP_WEIGHT), (u32)SEISMIC_DIM_M,
                              WIDTH, WIDTH, epilogue);
}

extern "C" __global__ void qwen_vision_merger_down(SEISMIC_KERNEL_PARAMS) {
    __shared__ vision::GemmShared shared;
    const vision::BiasF32<DownBias> epilogue{reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)),
                                             SEISMIC_RESULT_0_STRIDE_0, SEISMIC_PTR(SEISMIC_BUFFER_DOWN_BIAS),
                                             nullptr, 0};
    vision::gemm<A, DownWeight>(shared, ACTIVATED, WIDTH, SEISMIC_PTR(SEISMIC_BUFFER_DOWN_WEIGHT),
                                (u32)SEISMIC_DIM_M, (u32)SEISMIC_DIM_D, WIDTH, epilogue);
}
