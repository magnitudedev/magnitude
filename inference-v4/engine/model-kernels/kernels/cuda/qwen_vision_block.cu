// qwen_vision_block: one Qwen3-VL vision transformer block over M patch rows
// of D = H * W (W = 4P). Eight launches (bodies in lib/vision/vision.cuh): layer
// norm, QKV GEMM (+ bias), 2D rotation of the queries and keys in place, full
// attention, output GEMM (+ bias, + the block input), layer norm, up GEMM
// (+ bias, tanh GELU), down GEMM (+ bias, + the attention residual).
#include "lib/vision/vision.cuh"

using vision::u32;
using vision::u64;
using vision::u8;
typedef element::Act A;
typedef ELEMENT_OF(SEISMIC_QKV_WEIGHT) QkvWeight;
typedef ELEMENT_OF(SEISMIC_PROJECTION_WEIGHT) ProjectionWeight;
typedef ELEMENT_OF(SEISMIC_UP_WEIGHT) UpWeight;
typedef ELEMENT_OF(SEISMIC_DOWN_WEIGHT) DownWeight;
typedef ELEMENT_OF(SEISMIC_QKV_BIAS) QkvBias;
typedef ELEMENT_OF(SEISMIC_PROJECTION_BIAS) ProjectionBias;
typedef ELEMENT_OF(SEISMIC_UP_BIAS) UpBias;
typedef ELEMENT_OF(SEISMIC_DOWN_BIAS) DownBias;
typedef ELEMENT_OF(SEISMIC_NORM1_WEIGHT) Norm1Weight;
typedef ELEMENT_OF(SEISMIC_NORM1_BIAS) Norm1Bias;
typedef ELEMENT_OF(SEISMIC_NORM2_WEIGHT) Norm2Weight;
typedef ELEMENT_OF(SEISMIC_NORM2_BIAS) Norm2Bias;

constexpr u32 W = 4 * SEISMIC_DIM_P;
constexpr u32 D = SEISMIC_DIM_H * W;
constexpr u32 F = SEISMIC_DIM_F;

#define EPSILON __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON)
#define HIDDEN reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN))
#define NORMALIZED SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_NORMALIZED)
#define PROJECTED SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PROJECTED)
#define ATTENDED SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ATTENDED)
#define RESIDUAL reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_RESIDUAL))
#define ACTIVATED SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ACTIVATED)

extern "C" __global__ void qwen_vision_block_norm1(SEISMIC_KERNEL_PARAMS) {
    __shared__ float partials[32];
    vision::layer_norm<A, Norm1Weight, Norm1Bias>(HIDDEN + (u64)blockIdx.x * D,
                                                  NORMALIZED + (u64)blockIdx.x * D * A::bytes,
                                                  SEISMIC_PTR(SEISMIC_BUFFER_NORM1_WEIGHT),
                                                  SEISMIC_PTR(SEISMIC_BUFFER_NORM1_BIAS), D, EPSILON, partials);
}

extern "C" __global__ void qwen_vision_block_qkv(SEISMIC_KERNEL_PARAMS) {
    __shared__ vision::GemmShared shared;
    const vision::Bias<A, QkvBias> epilogue{PROJECTED, 3 * D, SEISMIC_PTR(SEISMIC_BUFFER_QKV_BIAS)};
    vision::gemm<A, QkvWeight>(shared, NORMALIZED, D, SEISMIC_PTR(SEISMIC_BUFFER_QKV_WEIGHT), (u32)SEISMIC_DIM_M,
                               3 * D, D, epilogue);
}

// One warp per (row, query or key, head).
extern "C" __global__ void qwen_vision_block_rotate(SEISMIC_KERNEL_PARAMS) {
    const u64 item = (u64)blockIdx.x * 8 + threadIdx.x / 32;
    const u64 row = item / (2 * SEISMIC_DIM_H);
    if (row >= SEISMIC_DIM_M)
        return;
    const u64 part = item % (2 * SEISMIC_DIM_H); // query heads, then key heads
    const int *coordinates = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES));
    vision::rotate<A, W>(PROJECTED + (row * 3 * D + part * W) * A::bytes, coordinates + row * 2, threadIdx.x % 32);
}

extern "C" __global__ void qwen_vision_block_attend(SEISMIC_KERNEL_PARAMS) {
    __shared__ vision::AttendShared<W> shared;
    vision::attend<A, W>(shared, PROJECTED, ATTENDED, (u32)SEISMIC_DIM_M, SEISMIC_DIM_H, rsqrtf(float(W)));
}

extern "C" __global__ void qwen_vision_block_output(SEISMIC_KERNEL_PARAMS) {
    __shared__ vision::GemmShared shared;
    const vision::BiasF32<ProjectionBias> epilogue{RESIDUAL, D, SEISMIC_PTR(SEISMIC_BUFFER_PROJECTION_BIAS),
                                                   HIDDEN, D};
    vision::gemm<A, ProjectionWeight>(shared, ATTENDED, D, SEISMIC_PTR(SEISMIC_BUFFER_PROJECTION_WEIGHT),
                                      (u32)SEISMIC_DIM_M, D, D, epilogue);
}

extern "C" __global__ void qwen_vision_block_norm2(SEISMIC_KERNEL_PARAMS) {
    __shared__ float partials[32];
    vision::layer_norm<A, Norm2Weight, Norm2Bias>(RESIDUAL + (u64)blockIdx.x * D,
                                                  NORMALIZED + (u64)blockIdx.x * D * A::bytes,
                                                  SEISMIC_PTR(SEISMIC_BUFFER_NORM2_WEIGHT),
                                                  SEISMIC_PTR(SEISMIC_BUFFER_NORM2_BIAS), D, EPSILON, partials);
}

extern "C" __global__ void qwen_vision_block_up(SEISMIC_KERNEL_PARAMS) {
    __shared__ vision::GemmShared shared;
    const vision::BiasGelu<A, UpBias, false> epilogue{ACTIVATED, F, SEISMIC_PTR(SEISMIC_BUFFER_UP_BIAS)};
    vision::gemm<A, UpWeight>(shared, NORMALIZED, D, SEISMIC_PTR(SEISMIC_BUFFER_UP_WEIGHT), (u32)SEISMIC_DIM_M, F, D,
                              epilogue);
}

extern "C" __global__ void qwen_vision_block_down(SEISMIC_KERNEL_PARAMS) {
    __shared__ vision::GemmShared shared;
    const vision::BiasF32<DownBias> epilogue{reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)),
                                             SEISMIC_RESULT_0_STRIDE_0, SEISMIC_PTR(SEISMIC_BUFFER_DOWN_BIAS),
                                             RESIDUAL, D};
    vision::gemm<A, DownWeight>(shared, ACTIVATED, F, SEISMIC_PTR(SEISMIC_BUFFER_DOWN_WEIGHT), (u32)SEISMIC_DIM_M, D,
                                F, epilogue);
}
