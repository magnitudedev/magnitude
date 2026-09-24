// qwen_vision_stem: the patch projection as two GEMMs, one per temporal frame
// (K = C * P * P each). A first launch converts the F32 pixels [M, C, 2, P, P]
// to the weights' 16-bit element in frame-major order [M, 2, C * P * P] (a
// relative rounding of 2^-11 for f16); the first GEMM stores its F32 product,
// the second adds it, the bias and the bilinear position-table blend,
// publishing the F32 residual stream. Bodies in common/vision.cuh.
#include "common/vision.cuh"

using vision::u32;
using vision::u64;
using vision::u8;
typedef ELEMENT_OF(SEISMIC_TEMPORAL_WEIGHT_0) Weight;
typedef ELEMENT_OF(SEISMIC_BIAS) BiasElement;
typedef ELEMENT_OF(SEISMIC_TABLE) TableElement;
static_assert(vision::Same<Weight, ELEMENT_OF(SEISMIC_TEMPORAL_WEIGHT_1)>::value,
              "both temporal patch weights share one element");

constexpr u64 AREA = (u64)SEISMIC_DIM_P * SEISMIC_DIM_P;
constexpr u64 K = SEISMIC_DIM_C * AREA; // one frame's GEMM depth
constexpr u32 H = SEISMIC_DIM_H;
#define CONVERTED SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_CONVERTED)
#define PARTIAL reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL))

extern "C" __global__ void qwen_vision_stem_convert(SEISMIC_KERNEL_PARAMS) {
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= SEISMIC_DIM_M * 2 * K)
        return;
    const u64 row = index / (2 * K), frame = (index / K) % 2, column = index % K;
    const u64 channel = column / AREA, within = column % AREA;
    const float *pixels = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PIXELS));
    element::put<Weight>(CONVERTED, index, pixels[row * SEISMIC_PIXELS_STRIDE_0 + (channel * 2 + frame) * AREA + within]);
}

// The first frame's F32 product.
struct Partial {
    float *partial;
    u64 columns;
    __device__ __forceinline__ void operator()(u32 m, u32 n, float first, float second) const {
        *reinterpret_cast<float2 *>(partial + (u64)m * columns + n) = make_float2(first, second);
    }
};

// (partial + product + bias) + the four table corners scaled by their
// coefficients, summed in order.
struct Stem {
    float *y;
    u64 stride;
    const float *partial;
    const u8 *bias;
    const u8 *table;
    const int *indices;
    const float *coefficients;
    u64 columns;
    __device__ __forceinline__ float value(u32 m, u32 n, float acc) const {
        const float projected = partial[(u64)m * columns + n] + acc + element::at<BiasElement>(bias, n);
        float blended = 0.0f;
#pragma unroll
        for (u32 i = 0; i < 4; ++i)
            blended += element::at<TableElement>(table, (u64)indices[m * 4 + i] * columns + n) * coefficients[m * 4 + i];
        return projected + blended;
    }
    __device__ __forceinline__ void operator()(u32 m, u32 n, float first, float second) const {
        *reinterpret_cast<float2 *>(y + (u64)m * stride + n) = make_float2(value(m, n, first), value(m, n + 1, second));
    }
};

extern "C" __global__ void qwen_vision_stem_first(SEISMIC_KERNEL_PARAMS) {
    __shared__ vision::GemmShared shared;
    vision::gemm<Weight, Weight>(shared, CONVERTED, 2 * K, SEISMIC_PTR(SEISMIC_BUFFER_TEMPORAL_WEIGHT_0),
                                 (u32)SEISMIC_DIM_M, H, (u32)K, Partial{PARTIAL, H});
}

extern "C" __global__ void qwen_vision_stem_second(SEISMIC_KERNEL_PARAMS) {
    __shared__ vision::GemmShared shared;
    const Stem epilogue{reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)),
                        SEISMIC_RESULT_0_STRIDE_0,
                        PARTIAL,
                        SEISMIC_PTR(SEISMIC_BUFFER_BIAS),
                        SEISMIC_PTR(SEISMIC_BUFFER_TABLE),
                        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_INDICES)),
                        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENTS)),
                        H};
    vision::gemm<Weight, Weight>(shared, CONVERTED + K * Weight::bytes, 2 * K,
                                 SEISMIC_PTR(SEISMIC_BUFFER_TEMPORAL_WEIGHT_1), (u32)SEISMIC_DIM_M, H, (u32)K, epilogue);
}
