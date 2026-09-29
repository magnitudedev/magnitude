// vision_patch_stem: the patch projection as one GEMM per frame (K = C * P * P
// each). A first launch converts the F32 pixels [M, C, 1 + S, P, P] to the
// staged element (the weights' 16-bit element, or f16 for F32 weights; a
// relative rounding of 2^-11 for f16) in frame-major rows [M, 1 + S, Kp]
// (Kp = K rounded up to whole 16-byte chunks, the pad zero); with two frames
// the first GEMM stores its F32 product; the last adds it, the bias and the
// bilinear position-table blend, publishing the F32 residual stream. Bodies in
// lib/vision/vision.cuh.
#include "lib/vision/vision.cuh"

using vision::u32;
using vision::u64;
using vision::u8;
typedef ELEMENT_OF(SEISMIC_FRAME_WEIGHT) Weight;
typedef ELEMENT_OF(SEISMIC_BIAS) BiasElement;
typedef ELEMENT_OF(SEISMIC_TABLE) TableElement;
typedef typename vision::Staged<element::F16, Weight>::type Pixel;
static_assert(SEISMIC_DIM_S == 0 || vision::Same<Weight, ELEMENT_OF(SEISMIC_NEXT_FRAME_WEIGHT)>::value,
              "every frame weight shares one element");

constexpr u64 FRAMES = 1 + SEISMIC_DIM_S;
constexpr u64 AREA = (u64)SEISMIC_DIM_P * SEISMIC_DIM_P;
constexpr u64 K = SEISMIC_DIM_C * AREA;      // one frame's GEMM depth
constexpr u64 KP = (K + 7) / 8 * 8;          // its staged row
constexpr u32 H = SEISMIC_DIM_H;
#define CONVERTED SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_CONVERTED)
#define PARTIAL reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL))

extern "C" __global__ void vision_patch_stem_convert(SEISMIC_KERNEL_PARAMS) {
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= SEISMIC_DIM_M * FRAMES * KP)
        return;
    const u64 row = index / (FRAMES * KP), frame = (index / KP) % FRAMES, column = index % KP;
    const u64 channel = column / AREA, within = column % AREA;
    const float *pixels = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PIXELS));
    element::put<Pixel>(CONVERTED, index,
                        column < K ? pixels[row * SEISMIC_PIXELS_STRIDE_0 + (channel * FRAMES + frame) * AREA + within]
                                   : 0.0f);
}

// The first frame's F32 product.
struct Partial {
    float *partial;
    u64 columns;
    __device__ __forceinline__ void operator()(u32 m, u32 n, float first, float second) const {
        *reinterpret_cast<float2 *>(partial + (u64)m * columns + n) = make_float2(first, second);
    }
};

// ((partial) + product (+ bias)) + the four table corners scaled by their
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
        float projected = acc;
        if constexpr (SEISMIC_DIM_S == 1)
            projected = partial[(u64)m * columns + n] + acc;
        if constexpr (SEISMIC_DIM_NB == 1)
            projected = projected + element::at<BiasElement>(bias, n);
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

extern "C" __global__ void vision_patch_stem_first(SEISMIC_KERNEL_PARAMS) {
    __shared__ vision::GemmShared shared;
    vision::gemm<Pixel, Weight>(shared, CONVERTED, FRAMES * KP, SEISMIC_PTR(SEISMIC_BUFFER_FRAME_WEIGHT),
                                (u32)SEISMIC_DIM_M, H, (u32)K, Partial{PARTIAL, H});
}

extern "C" __global__ void vision_patch_stem_last(SEISMIC_KERNEL_PARAMS) {
    __shared__ vision::GemmShared shared;
    const Stem epilogue{reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)),
                        SEISMIC_RESULT_0_STRIDE_0,
                        PARTIAL,
                        SEISMIC_PTR(SEISMIC_BUFFER_BIAS),
                        SEISMIC_PTR(SEISMIC_BUFFER_TABLE),
                        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_INDICES)),
                        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENTS)),
                        H};
    const u8 *weight = SEISMIC_DIM_S == 1 ? SEISMIC_PTR(SEISMIC_BUFFER_NEXT_FRAME_WEIGHT)
                                          : SEISMIC_PTR(SEISMIC_BUFFER_FRAME_WEIGHT);
    vision::gemm<Pixel, Weight>(shared, CONVERTED + SEISMIC_DIM_S * KP * Pixel::bytes, FRAMES * KP, weight,
                                (u32)SEISMIC_DIM_M, H, (u32)K, epilogue);
}
