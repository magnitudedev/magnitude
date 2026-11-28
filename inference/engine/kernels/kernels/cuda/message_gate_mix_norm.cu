// message_gate_mix_norm: fused cross-pass message gate + RMSNorm to A:
//   y[m, h]   = (1 - tanh(forget)) * dest + tanh(gain) * msg
//   inv[m]    = rsqrt(mean_h(y[m, h]^2) + epsilon)
//   out[m, h] = A((y[m, h] * inv[m]) * norm[h])
// One 256-thread block per row.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

using element::u64;
using element::u8;

extern "C" __global__ void message_gate_mix_norm(SEISMIC_KERNEL_PARAMS) {
    using Norm = ELEMENT_OF(SEISMIC_NORM);
    __shared__ float partials[32];

    const u64 row = blockIdx.x;
    const u64 H = SEISMIC_DIM_H;
    const float eps = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const float *dest = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_DEST))
        + row * SEISMIC_DEST_STRIDE_0;
    const float *msg = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_MSG))
        + row * SEISMIC_MSG_STRIDE_0;
    const float *gain = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_GAIN))
        + row * SEISMIC_GAIN_STRIDE_0;
    const float *forget = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_FORGET))
        + row * SEISMIC_FORGET_STRIDE_0;
    const u8 *norm = SEISMIC_PTR(SEISMIC_BUFFER_NORM);
    const auto result = SEISMIC_RESULT_0_OUT.offset(row * SEISMIC_RESULT_0_STRIDE_0);

    constexpr u64 CACHE = 32ull;
    const bool can_cache = H <= CACHE * 256ull;
    float cached[CACHE];
    float sq = 0.0f;

    if (can_cache) {
#pragma unroll
        for (u64 slot = 0; slot < CACHE; ++slot) {
            const u64 i = (u64)threadIdx.x + slot * 256ull;
            if (i < H) {
                const float d = dest[i * SEISMIC_DEST_STRIDE_1];
                const float m = msg[i * SEISMIC_MSG_STRIDE_1];
                const float g = gain[i * SEISMIC_GAIN_STRIDE_1];
                const float f = forget[i * SEISMIC_FORGET_STRIDE_1];
                const float tf = 2.0f / (1.0f + expf(-2.0f * f)) - 1.0f;
                const float tg = 2.0f / (1.0f + expf(-2.0f * g)) - 1.0f;
                const float y = (1.0f - tf) * d + tg * m;
                cached[slot] = y;
                sq = seismic_fma_rn(y, y, sq);
            }
        }
        const float inv = rsqrtf(reduce::group_sum(sq, partials) / (float)H + eps);
#pragma unroll
        for (u64 slot = 0; slot < CACHE; ++slot) {
            const u64 i = (u64)threadIdx.x + slot * 256ull;
            if (i < H) {
                const float nv = element::at<Norm>(norm, i * SEISMIC_NORM_STRIDE_0);
                result.put(i * SEISMIC_RESULT_0_STRIDE_1, (cached[slot] * inv) * nv);
            }
        }
    } else {
        for (u64 i = threadIdx.x; i < H; i += blockDim.x) {
            const float d = dest[i * SEISMIC_DEST_STRIDE_1];
            const float m = msg[i * SEISMIC_MSG_STRIDE_1];
            const float g = gain[i * SEISMIC_GAIN_STRIDE_1];
            const float f = forget[i * SEISMIC_FORGET_STRIDE_1];
            const float tf = 2.0f / (1.0f + expf(-2.0f * f)) - 1.0f;
            const float tg = 2.0f / (1.0f + expf(-2.0f * g)) - 1.0f;
            const float y = (1.0f - tf) * d + tg * m;
            sq = seismic_fma_rn(y, y, sq);
        }
        const float inv = rsqrtf(reduce::group_sum(sq, partials) / (float)H + eps);
        for (u64 i = threadIdx.x; i < H; i += blockDim.x) {
            const float d = dest[i * SEISMIC_DEST_STRIDE_1];
            const float m = msg[i * SEISMIC_MSG_STRIDE_1];
            const float g = gain[i * SEISMIC_GAIN_STRIDE_1];
            const float f = forget[i * SEISMIC_FORGET_STRIDE_1];
            const float tf = 2.0f / (1.0f + expf(-2.0f * f)) - 1.0f;
            const float tg = 2.0f / (1.0f + expf(-2.0f * g)) - 1.0f;
            const float y = (1.0f - tf) * d + tg * m;
            const float nv = element::at<Norm>(norm, i * SEISMIC_NORM_STRIDE_0);
            result.put(i * SEISMIC_RESULT_0_STRIDE_1, (y * inv) * nv);
        }
    }
}
