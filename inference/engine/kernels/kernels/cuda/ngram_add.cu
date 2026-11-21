// ngram_add (contract and portable body in ngram.seismic): each stream's
// gated value row added into its streams. A thread owns 4 contiguous columns
// across all S streams so `value` is read once per row; the y grid visits the
// rows with a stride (it is capped at 65535).

#include <seismic/element.cuh>

namespace {

typedef unsigned long long u64;

__device__ __forceinline__ void load4_f32(const float *base, u64 stride, float (&out)[4]) {
    if (stride == 1 && ((reinterpret_cast<u64>(base) & 15u) == 0)) {
        element::f32_span<4>(base, out);
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            out[i] = base[static_cast<u64>(i) * stride];
    }
}

__device__ __forceinline__ void store4_f32(const element::Out<element::F32> &out, u64 index, u64 stride,
                                           const float (&values)[4]) {
    if (stride == 1 && ((reinterpret_cast<u64>(out.stored(index)) & 15u) == 0)) {
        out.put_span<4>(index, values);
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            out.put(index + static_cast<u64>(i) * stride, values[i]);
    }
}

} // namespace

extern "C" __global__ void ngram_add(SEISMIC_KERNEL_PARAMS) {
    const float *hidden = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN));
    const float *scales = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALES));
    const float *value = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_VALUE));
    const auto result = SEISMIC_RESULT_0_OUT;
    const u64 col0 = (static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (col0 >= SEISMIC_DIM_H)
        return;
    const bool full4 = (SEISMIC_DIM_H % 4 == 0) || (col0 + 4 <= SEISMIC_DIM_H);
    for (u64 row = blockIdx.y; row < SEISMIC_DIM_M; row += gridDim.y) {
        const u64 scale_row = row * SEISMIC_SCALES_STRIDE_0;
        const u64 hidden_row = row * SEISMIC_HIDDEN_STRIDE_0;
        const u64 result_row = row * SEISMIC_RESULT_0_STRIDE_0;
        const float *value_base = value + row * SEISMIC_VALUE_STRIDE_0 + col0 * SEISMIC_VALUE_STRIDE_1;
        if (full4) {
            float v[4];
            load4_f32(value_base, SEISMIC_VALUE_STRIDE_1, v);
            for (u64 stream = 0; stream < SEISMIC_DIM_S; ++stream) {
                const float gate = scales[scale_row + stream * SEISMIC_SCALES_STRIDE_1];
                const u64 chan0 = stream * SEISMIC_DIM_H + col0;
                float h[4];
                load4_f32(hidden + hidden_row + chan0 * SEISMIC_HIDDEN_STRIDE_1, SEISMIC_HIDDEN_STRIDE_1, h);
                float out[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    out[i] = __fadd_rn(h[i], __fmul_rn(gate, v[i]));
                store4_f32(result, result_row + chan0 * SEISMIC_RESULT_0_STRIDE_1, SEISMIC_RESULT_0_STRIDE_1, out);
            }
        } else {
            const u64 count = SEISMIC_DIM_H - col0;
            float v[3];
            for (u64 i = 0; i < count; ++i)
                v[i] = value_base[i * SEISMIC_VALUE_STRIDE_1];
            for (u64 stream = 0; stream < SEISMIC_DIM_S; ++stream) {
                const float gate = scales[scale_row + stream * SEISMIC_SCALES_STRIDE_1];
                const u64 chan0 = stream * SEISMIC_DIM_H + col0;
                for (u64 i = 0; i < count; ++i) {
                    const u64 chan = chan0 + i;
                    const float gated = __fmul_rn(gate, v[i]);
                    result.put(result_row + chan * SEISMIC_RESULT_0_STRIDE_1,
                               __fadd_rn(hidden[hidden_row + chan * SEISMIC_HIDDEN_STRIDE_1], gated));
                }
            }
        }
    }
}
