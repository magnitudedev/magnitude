// vision_patch_stem: the patch projection as one GEMM per frame (K = C * P * P
// each). A first launch converts the F32 pixels [M, C, 1 + S, P, P] to f16
// (a relative rounding of 2^-11, whatever the weights' element: BF16 pixels
// would round at 2^-9 ahead of every block) in frame-major rows [M, 1 + S, Kp]
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
typedef element::F16 Pixel;
static_assert(SEISMIC_DIM_S == 0 || vision::Same<Weight, ELEMENT_OF(SEISMIC_NEXT_FRAME_WEIGHT)>::value,
              "every frame weight shares one element");

constexpr u64 FRAMES = 1 + SEISMIC_DIM_S;
constexpr u64 AREA = (u64)SEISMIC_DIM_P * SEISMIC_DIM_P;
constexpr u64 K = SEISMIC_DIM_C * AREA;      // one frame's GEMM depth
constexpr u64 KP = (K + 7) / 8 * 8;          // its staged row
constexpr u32 H = SEISMIC_DIM_H;
#define CONVERTED SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_CONVERTED)
#define PARTIAL reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL))

template <class E> __device__ __forceinline__ float2 load2(const u8 *base, u64 index) {
    if constexpr (E::bytes == 4) {
        const u8 *address = base + index * 4;
        if ((reinterpret_cast<u64>(address) & 7u) == 0)
            return *reinterpret_cast<const float2 *>(address);
    } else {
        const u8 *address = base + index * 2;
        if ((reinterpret_cast<u64>(address) & 3u) == 0)
            return E::unpack2(*reinterpret_cast<const u32 *>(address));
    }
    return make_float2(element::at<E>(base, index), element::at<E>(base, index + 1));
}

extern "C" __global__ void vision_patch_stem_convert(SEISMIC_KERNEL_PARAMS) {
    const u32 row = blockIdx.x;
    const u32 frame = blockIdx.y;
    const u32 col0 = (blockIdx.z * blockDim.x + threadIdx.x) * 4u;
    if (col0 >= (u32)KP)
        return;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (col0 < (u32)K) {
        const float *row_pixels =
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PIXELS)) + (u64)row * SEISMIC_PIXELS_STRIDE_0;
        u32 channel = col0 / (u32)AREA;
        u32 within = col0 % (u32)AREA;
        if (within + 4u <= (u32)AREA) {
            const float *src = row_pixels + ((u64)channel * FRAMES + frame) * AREA + within;
            if ((reinterpret_cast<u64>(src) & 15u) == 0) {
                const float4 f = *reinterpret_cast<const float4 *>(src);
                v[0] = f.x;
                v[1] = f.y;
                v[2] = f.z;
                v[3] = f.w;
            } else {
                v[0] = src[0];
                v[1] = src[1];
                v[2] = src[2];
                v[3] = src[3];
            }
        } else {
            const u32 count = min(4u, (u32)K - col0);
#pragma unroll
            for (u32 i = 0; i < 4u; ++i) {
                if (i < count) {
                    v[i] = row_pixels[((u64)channel * FRAMES + frame) * AREA + within];
                    if (++within == (u32)AREA) {
                        within = 0u;
                        ++channel;
                    }
                }
            }
        }
    }
    const u64 out_index = ((u64)row * FRAMES + frame) * KP + col0;
    *reinterpret_cast<uint2 *>(CONVERTED + out_index * Pixel::bytes) =
        make_uint2(Pixel::pack2(v[0], v[1]), Pixel::pack2(v[2], v[3]));
}

// The first frame's F32 product.
struct Partial {
    float *partial;
    u64 columns;
    __device__ __forceinline__ void operator()(u32 m, u32 n, float first, float second) const {
        *reinterpret_cast<float2 *>(partial + (u64)m * columns + n) = make_float2(first, second);
    }
};

// ((partial) + product (+ bias)) + (with NT = 1) the four table corners
// scaled by their coefficients, summed in order.
struct Stem {
    element::Out<element::F32> y;
    u64 stride;
    const float *partial;
    const u8 *bias;
    const u8 *table;
    const int *indices;
    const float *coefficients;
    u64 columns;
    __device__ __forceinline__ void operator()(u32 m, u32 n, float first, float second) const {
        float p0 = first, p1 = second;
        if constexpr (SEISMIC_DIM_S == 1) {
            const float2 p = *reinterpret_cast<const float2 *>(partial + (u64)m * columns + n);
            p0 = p.x + first;
            p1 = p.y + second;
        }
        if constexpr (SEISMIC_DIM_NB == 1) {
            const float2 b = load2<BiasElement>(bias, n);
            p0 = p0 + b.x;
            p1 = p1 + b.y;
        }
        float b0 = 0.0f, b1 = 0.0f;
        if constexpr (SEISMIC_DIM_NT == 1) {
            const int *idx_ptr = indices + (u64)m * 4;
            const float *coef_ptr = coefficients + (u64)m * 4;
            int idx[4];
            float coef[4];
            if ((reinterpret_cast<u64>(idx_ptr) & 15u) == 0) {
                const int4 v = *reinterpret_cast<const int4 *>(idx_ptr);
                idx[0] = v.x;
                idx[1] = v.y;
                idx[2] = v.z;
                idx[3] = v.w;
            } else {
#pragma unroll
                for (u32 i = 0; i < 4; ++i)
                    idx[i] = idx_ptr[i];
            }
            if ((reinterpret_cast<u64>(coef_ptr) & 15u) == 0) {
                const float4 v = *reinterpret_cast<const float4 *>(coef_ptr);
                coef[0] = v.x;
                coef[1] = v.y;
                coef[2] = v.z;
                coef[3] = v.w;
            } else {
#pragma unroll
                for (u32 i = 0; i < 4; ++i)
                    coef[i] = coef_ptr[i];
            }
#pragma unroll
            for (u32 i = 0; i < 4; ++i) {
                const float2 t = load2<TableElement>(table, (u64)idx[i] * columns + n);
                b0 += t.x * coef[i];
                b1 += t.y * coef[i];
            }
        }
        y.put2((u64)m * stride + n, p0 + b0, p1 + b1);
    }
};

extern "C" __global__ void vision_patch_stem_first(SEISMIC_KERNEL_PARAMS) {
    __shared__ vision::GemmShared shared;
    vision::gemm<Pixel, Weight>(shared, CONVERTED, FRAMES * KP, SEISMIC_PTR(SEISMIC_BUFFER_FRAME_WEIGHT),
                                (u32)SEISMIC_DIM_M, H, (u32)K, Partial{PARTIAL, H});
}

extern "C" __global__ void vision_patch_stem_last(SEISMIC_KERNEL_PARAMS) {
    __shared__ vision::GemmShared shared;
    const Stem epilogue{SEISMIC_RESULT_0_OUT,
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
