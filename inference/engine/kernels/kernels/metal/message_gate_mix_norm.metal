// message_gate_mix_norm: fused cross-pass message gate + RMSNorm to A:
//   y[m, h]   = (1 - tanh(forget)) * dest + tanh(gain) * msg
//   inv[m]    = rsqrt(mean_h(y[m, h]^2) + epsilon)
//   out[m, h] = A((y[m, h] * inv[m]) * norm[h])
// One 256-thread threadgroup per row.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"
#include "lib/core/reduce.h"

typedef ELEMENT_OF(SEISMIC_NORM) Norm;

kernel void message_gate_mix_norm(
    device const float *dest [[buffer(SEISMIC_BUFFER_DEST)]],
    device const float *msg [[buffer(SEISMIC_BUFFER_MSG)]],
    device const float *gain [[buffer(SEISMIC_BUFFER_GAIN)]],
    device const float *forget [[buffer(SEISMIC_BUFFER_FORGET)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[8];

    const ulong H = SEISMIC_DIM_H;
    const float eps = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    device const float *dr = dest + ulong(row) * SEISMIC_DEST_STRIDE_0;
    device const float *mr = msg + ulong(row) * SEISMIC_MSG_STRIDE_0;
    device const float *gr = gain + ulong(row) * SEISMIC_GAIN_STRIDE_0;
    device const float *fr = forget + ulong(row) * SEISMIC_FORGET_STRIDE_0;
    const element::Out<activation::Act> result(result_cells);
    const ulong out_base = ulong(row) * SEISMIC_RESULT_0_STRIDE_0;

    constexpr ulong CACHE = 32ul;
    const bool can_cache = H <= CACHE * 256ul;
    float cached[CACHE];
    float sq = 0.0f;

    if (can_cache) {
        for (ulong slot = 0ul; slot < CACHE; ++slot) {
            const ulong i = ulong(thread_index) + slot * 256ul;
            if (i < H) {
                const float d = dr[i * SEISMIC_DEST_STRIDE_1];
                const float m = mr[i * SEISMIC_MSG_STRIDE_1];
                const float g = gr[i * SEISMIC_GAIN_STRIDE_1];
                const float f = fr[i * SEISMIC_FORGET_STRIDE_1];
                const float tf = 2.0f / (1.0f + metal::exp(-2.0f * f)) - 1.0f;
                const float tg = 2.0f / (1.0f + metal::exp(-2.0f * g)) - 1.0f;
                const float y = (1.0f - tf) * d + tg * m;
                cached[slot] = y;
                sq = metal::fma(y, y, sq);
            }
        }
        const float inv = metal::rsqrt(reduce::group_sum<8>(sq, partials, sg, lane) / float(H) + eps);
        for (ulong slot = 0ul; slot < CACHE; ++slot) {
            const ulong i = ulong(thread_index) + slot * 256ul;
            if (i < H) {
                const float nv = element::at<Norm>(norm, i * SEISMIC_NORM_STRIDE_0);
                result.put(out_base + i * SEISMIC_RESULT_0_STRIDE_1, (cached[slot] * inv) * nv);
            }
        }
    } else {
        for (ulong i = thread_index; i < H; i += 256u) {
            const float d = dr[i * SEISMIC_DEST_STRIDE_1];
            const float m = mr[i * SEISMIC_MSG_STRIDE_1];
            const float g = gr[i * SEISMIC_GAIN_STRIDE_1];
            const float f = fr[i * SEISMIC_FORGET_STRIDE_1];
            const float tf = 2.0f / (1.0f + metal::exp(-2.0f * f)) - 1.0f;
            const float tg = 2.0f / (1.0f + metal::exp(-2.0f * g)) - 1.0f;
            const float y = (1.0f - tf) * d + tg * m;
            sq = metal::fma(y, y, sq);
        }
        const float inv = metal::rsqrt(reduce::group_sum<8>(sq, partials, sg, lane) / float(H) + eps);
        for (ulong i = thread_index; i < H; i += 256u) {
            const float d = dr[i * SEISMIC_DEST_STRIDE_1];
            const float m = mr[i * SEISMIC_MSG_STRIDE_1];
            const float g = gr[i * SEISMIC_GAIN_STRIDE_1];
            const float f = fr[i * SEISMIC_FORGET_STRIDE_1];
            const float tf = 2.0f / (1.0f + metal::exp(-2.0f * f)) - 1.0f;
            const float tg = 2.0f / (1.0f + metal::exp(-2.0f * g)) - 1.0f;
            const float y = (1.0f - tf) * d + tg * m;
            const float nv = element::at<Norm>(norm, i * SEISMIC_NORM_STRIDE_0);
            result.put(out_base + i * SEISMIC_RESULT_0_STRIDE_1, (y * inv) * nv);
        }
    }
}
