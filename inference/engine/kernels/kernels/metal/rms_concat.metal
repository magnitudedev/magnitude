// rms_concat: `[RMS(a, eps) || RMS(b, eps)]` rounded to A. One threadgroup per row.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"
#include "lib/core/reduce.h"

kernel void rms_concat(
    device const float *a [[buffer(SEISMIC_BUFFER_A)]],
    device const float *b [[buffer(SEISMIC_BUFFER_B)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[8];
    const ulong H = SEISMIC_DIM_H;
    const float eps = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    device const float *ar = a + ulong(row) * SEISMIC_A_STRIDE_0;
    device const float *br = b + ulong(row) * SEISMIC_B_STRIDE_0;
    float sq_a = 0.0f;
    float sq_b = 0.0f;
    for (ulong i = thread_index; i < H; i += 256u) {
        const float va = ar[i * SEISMIC_A_STRIDE_1];
        const float vb = br[i * SEISMIC_B_STRIDE_1];
        sq_a = metal::fma(va, va, sq_a);
        sq_b = metal::fma(vb, vb, sq_b);
    }
    const float inv_a = metal::rsqrt(reduce::group_sum<8>(sq_a, partials, sg, lane) / float(H) + eps);
    const float inv_b = metal::rsqrt(reduce::group_sum<8>(sq_b, partials, sg, lane) / float(H) + eps);
    const element::Out<activation::Act> result(result_cells);
    const ulong base = ulong(row) * SEISMIC_RESULT_0_STRIDE_0;
    for (ulong i = thread_index; i < H; i += 256u) {
        result.put(base + i * SEISMIC_RESULT_0_STRIDE_1, ar[i * SEISMIC_A_STRIDE_1] * inv_a);
        result.put(base + (H + i) * SEISMIC_RESULT_0_STRIDE_1, br[i * SEISMIC_B_STRIDE_1] * inv_b);
    }
}
