// ngram_add (contract and portable body in ngram.seismic): each stream's
// gated value row added into its streams. A thread owns 4 contiguous columns
// of one row across all S streams so `value` is read once.
#include <seismic/element.h>

namespace ngram_add_ops {

inline float4 load4_f32(device const float *base, ulong stride) {
    if (stride == 1)
        return float4(*reinterpret_cast<device const packed_float4 *>(base));
    return float4(base[0], base[stride], base[2 * stride], base[3 * stride]);
}

inline void store4_f32(element::Out<element::F32> out, ulong index, ulong stride, float4 values) {
    if (stride == 1) {
        out.put4(index, values);
    } else {
        out.put(index, values.x);
        out.put(index + stride, values.y);
        out.put(index + 2 * stride, values.z);
        out.put(index + 3 * stride, values.w);
    }
}

} // namespace ngram_add_ops

kernel void ngram_add(
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const float *scales [[buffer(SEISMIC_BUFFER_SCALES)]],
    device const float *value [[buffer(SEISMIC_BUFFER_VALUE)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    const ulong col0 = (ulong(group.x) * threadgroup_shape.x + thread_index) * 4;
    if (col0 >= SEISMIC_DIM_H)
        return;
    const ulong row = group.y;
    const ulong scale_row = row * SEISMIC_SCALES_STRIDE_0;
    const ulong hidden_row = row * SEISMIC_HIDDEN_STRIDE_0;
    const ulong result_row = row * SEISMIC_RESULT_0_STRIDE_0;
    device const float *value_base = value + row * SEISMIC_VALUE_STRIDE_0 + col0 * SEISMIC_VALUE_STRIDE_1;
    const element::Out<element::F32> result(result_cells);
    if (SEISMIC_DIM_H % 4 == 0 || col0 + 4 <= SEISMIC_DIM_H) {
        const float4 v = ngram_add_ops::load4_f32(value_base, SEISMIC_VALUE_STRIDE_1);
        for (ulong stream = 0; stream < SEISMIC_DIM_S; ++stream) {
            const float gate = scales[scale_row + stream * SEISMIC_SCALES_STRIDE_1];
            const ulong chan0 = stream * SEISMIC_DIM_H + col0;
            const float4 h = ngram_add_ops::load4_f32(
                hidden + hidden_row + chan0 * SEISMIC_HIDDEN_STRIDE_1, SEISMIC_HIDDEN_STRIDE_1);
            ngram_add_ops::store4_f32(result, result_row + chan0 * SEISMIC_RESULT_0_STRIDE_1,
                SEISMIC_RESULT_0_STRIDE_1, h + gate * v);
        }
    } else {
        const ulong count = SEISMIC_DIM_H - col0;
        float v[3];
        for (ulong i = 0; i < count; ++i)
            v[i] = value_base[i * SEISMIC_VALUE_STRIDE_1];
        for (ulong stream = 0; stream < SEISMIC_DIM_S; ++stream) {
            const float gate = scales[scale_row + stream * SEISMIC_SCALES_STRIDE_1];
            const ulong chan0 = stream * SEISMIC_DIM_H + col0;
            for (ulong i = 0; i < count; ++i) {
                const ulong chan = chan0 + i;
                result.put(result_row + chan * SEISMIC_RESULT_0_STRIDE_1,
                    hidden[hidden_row + chan * SEISMIC_HIDDEN_STRIDE_1] + gate * v[i]);
            }
        }
    }
}
