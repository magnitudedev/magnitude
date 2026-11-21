// ngram_inject (contract and portable body in ngram.seismic): the n-gram
// input's dilated causal convolution over its banked window, added with the
// gated value into the streams. A thread owns 4 contiguous columns of one
// stream (grid z) and one row: grid y < M computes row y (a copy of `hidden`
// past the slots), grid y = M + b publishes slot b's successor window. `norm`
// is hoisted once before the tap loop.
#include "lib/core/activation.h"
#include "lib/recurrent/versions.h"

typedef ELEMENT_OF(SEISMIC_NORM) Norm;

namespace ngram {

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

inline float4 load4_norm(device const uchar *base, ulong index, ulong stride) {
    if (stride == 1)
        return Norm::load4(reinterpret_cast<device const Norm::storage *>(base) + index);
    return float4(element::at<Norm>(base, index), element::at<Norm>(base, index + stride),
        element::at<Norm>(base, index + 2 * stride), element::at<Norm>(base, index + 3 * stride));
}

inline float4 u4(device const float *scales_buf, device const float *value_buf, ulong row, ulong stream, ulong col0,
    float4 norm4, thread float4 &gated, constant ulong *seismic_words) {
    device const float *scales = scales_buf + row * SEISMIC_SCALES_STRIDE_0;
    const float s0 = scales[stream * SEISMIC_SCALES_STRIDE_1];
    const float s1 = scales[(SEISMIC_DIM_S + stream) * SEISMIC_SCALES_STRIDE_1];
    const float4 v = load4_f32(value_buf + row * SEISMIC_VALUE_STRIDE_0 + col0 * SEISMIC_VALUE_STRIDE_1,
        SEISMIC_VALUE_STRIDE_1);
    gated = s0 * v;
    return (gated * s1) * norm4;
}

} // namespace ngram

kernel void ngram_inject(
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const float *scales [[buffer(SEISMIC_BUFFER_SCALES)]],
    device const float *value [[buffer(SEISMIC_BUFFER_VALUE)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const float *convolution [[buffer(SEISMIC_BUFFER_CONVOLUTION)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    SEISMIC_OUT_WINDOW_PARAM(window),
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    const ulong col0 = (ulong(group.x) * threadgroup_shape.x + thread_index) * 4;
    if (col0 >= SEISMIC_DIM_H)
        return;
    const ulong stream = ulong(group.z);
    const ulong chan0 = stream * SEISMIC_DIM_H + col0;
    const bool full4 = (SEISMIC_DIM_H % 4 == 0) || (col0 + 4 <= SEISMIC_DIM_H);
    const long dilation = long(int(uint(SEISMIC_PARAM_DILATION)));
    const long reach = dilation * (long(SEISMIC_DIM_K) - 1);
    const versions::Slots slots{segments, stop, previous_bank, previous_tape, following_bank};
    const ulong rows = SEISMIC_DIM_M;
    const ulong norm_idx = stream * SEISMIC_NORM_STRIDE_0 + col0 * SEISMIC_NORM_STRIDE_1;

    if (group.y >= rows) {
        const versions::Slot slot = versions::slot_of(slots, group.y - rows, seismic_words);
        if (slot.lo == slot.hi) return;
        const element::Out<element::F32> source = versions::bank(window, slot.source, SEISMIC_WINDOW_STRIDE_0,
            seismic_words);
        const element::Out<element::F32> target = versions::bank(window, slot.target, SEISMIC_WINDOW_STRIDE_0,
            seismic_words);
        // The tape holds T = R - reach rows.
        const long published = reach + versions::tape_rows(slot, long(SEISMIC_DIM_R) - reach);
        if (full4) {
            const float4 norm4 = ngram::load4_norm(norm, norm_idx, SEISMIC_NORM_STRIDE_1);
            for (long tap = 0; tap < published; ++tap) {
                const long position = slot.stop + tap - reach;
                const ulong to = ulong(tap) * SEISMIC_WINDOW_STRIDE_1 + chan0 * SEISMIC_WINDOW_STRIDE_2;
                // The source window's rows are the owner's: moved. A launch row's
                // u is formed: stored.
                if (position < 0) {
                    const ulong from = ulong(slot.taped + slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1
                        + chan0 * SEISMIC_WINDOW_STRIDE_2;
                    if (SEISMIC_WINDOW_STRIDE_2 == 1) {
                        target.move_stored(to, source, from, 4);
                    } else {
                        for (ulong i = 0; i < 4; ++i)
                            target.move_stored(to + i * SEISMIC_WINDOW_STRIDE_2, source,
                                from + i * SEISMIC_WINDOW_STRIDE_2, 1);
                    }
                } else {
                    float4 gated;
                    const float4 u_val = ngram::u4(scales, value, ulong(slot.lo + position), stream, col0, norm4,
                        gated, seismic_words);
                    ngram::store4_f32(target, to, SEISMIC_WINDOW_STRIDE_2, u_val);
                }
            }
        } else {
            const ulong count = SEISMIC_DIM_H - col0;
            float norm_tail[3];
            for (ulong i = 0; i < count; ++i)
                norm_tail[i] = element::at<Norm>(norm, norm_idx + i * SEISMIC_NORM_STRIDE_1);
            for (long tap = 0; tap < published; ++tap) {
                const long position = slot.stop + tap - reach;
                const ulong to = ulong(tap) * SEISMIC_WINDOW_STRIDE_1 + chan0 * SEISMIC_WINDOW_STRIDE_2;
                if (position < 0) {
                    const ulong from = ulong(slot.taped + slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1
                        + chan0 * SEISMIC_WINDOW_STRIDE_2;
                    for (ulong i = 0; i < count; ++i)
                        target.move_stored(to + i * SEISMIC_WINDOW_STRIDE_2, source,
                            from + i * SEISMIC_WINDOW_STRIDE_2, 1);
                } else {
                    const ulong src_row = ulong(slot.lo + position);
                    device const float *row_scales = scales + src_row * SEISMIC_SCALES_STRIDE_0;
                    const float s0 = row_scales[stream * SEISMIC_SCALES_STRIDE_1];
                    const float s1 = row_scales[(SEISMIC_DIM_S + stream) * SEISMIC_SCALES_STRIDE_1];
                    device const float *val_base = value + src_row * SEISMIC_VALUE_STRIDE_0 + col0 * SEISMIC_VALUE_STRIDE_1;
                    for (ulong i = 0; i < count; ++i) {
                        const float g = s0 * val_base[i * SEISMIC_VALUE_STRIDE_1];
                        target.put(to + i * SEISMIC_WINDOW_STRIDE_2, (g * s1) * norm_tail[i]);
                    }
                }
            }
        }
        return;
    }

    const long row = long(group.y);
    device const float *hidden_ptr = hidden + ulong(row) * SEISMIC_HIDDEN_STRIDE_0 + chan0 * SEISMIC_HIDDEN_STRIDE_1;
    const element::Out<element::F32> result(result_cells);
    const ulong out_idx = ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + chan0 * SEISMIC_RESULT_0_STRIDE_1;
    const ulong slot_index = versions::slot_index_of_row(slots, row, seismic_words);
    if (slot_index == SEISMIC_DIM_B) {
        if (full4) {
            const float4 residual = ngram::load4_f32(hidden_ptr, SEISMIC_HIDDEN_STRIDE_1);
            ngram::store4_f32(result, out_idx, SEISMIC_RESULT_0_STRIDE_1, residual);
        } else {
            const ulong count = SEISMIC_DIM_H - col0;
            for (ulong i = 0; i < count; ++i)
                result.put(out_idx + i * SEISMIC_RESULT_0_STRIDE_1, hidden_ptr[i * SEISMIC_HIDDEN_STRIDE_1]);
        }
        return;
    }

    const versions::Slot slot = versions::slot_of(slots, slot_index, seismic_words);
    device const float *source = versions::bank_view(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words);
    const long local = row - slot.lo;
    if (full4) {
        const float4 norm4 = ngram::load4_norm(norm, norm_idx, SEISMIC_NORM_STRIDE_1);
        float4 sum = float4(0.0f);
        float4 gated = float4(0.0f);
        for (long tap = 0; tap < long(SEISMIC_DIM_K); ++tap) {
            const long position = local + dilation * tap - reach;
            const float4 input = position < 0
                ? ngram::load4_f32(
                    source + ulong(slot.taped + reach + position) * SEISMIC_WINDOW_STRIDE_1
                        + chan0 * SEISMIC_WINDOW_STRIDE_2,
                    SEISMIC_WINDOW_STRIDE_2)
                : ngram::u4(scales, value, ulong(slot.lo + position), stream, col0, norm4, gated, seismic_words);
            const ulong tap_off = ulong(tap) * SEISMIC_CONVOLUTION_STRIDE_1;
            const float4 weight = float4(
                convolution[chan0 * SEISMIC_CONVOLUTION_STRIDE_0 + tap_off],
                convolution[(chan0 + 1) * SEISMIC_CONVOLUTION_STRIDE_0 + tap_off],
                convolution[(chan0 + 2) * SEISMIC_CONVOLUTION_STRIDE_0 + tap_off],
                convolution[(chan0 + 3) * SEISMIC_CONVOLUTION_STRIDE_0 + tap_off]);
            sum = metal::fma(weight, input, sum);
        }
        // tap == K - 1 has position == local >= 0 and slot.lo + position == row, so
        // `gated` already holds scales[row, stream] * value[row, col0..col0+4].
        const float4 residual = ngram::load4_f32(hidden_ptr, SEISMIC_HIDDEN_STRIDE_1);
        ngram::store4_f32(result, out_idx, SEISMIC_RESULT_0_STRIDE_1,
            residual + (gated + sum / (1.0f + metal::exp(-sum))));
    } else {
        const ulong count = SEISMIC_DIM_H - col0;
        float norm_tail[3];
        for (ulong i = 0; i < count; ++i)
            norm_tail[i] = element::at<Norm>(norm, norm_idx + i * SEISMIC_NORM_STRIDE_1);
        float sum[3] = {0.0f, 0.0f, 0.0f};
        float gated[3] = {0.0f, 0.0f, 0.0f};
        for (long tap = 0; tap < long(SEISMIC_DIM_K); ++tap) {
            const long position = local + dilation * tap - reach;
            const ulong tap_off = ulong(tap) * SEISMIC_CONVOLUTION_STRIDE_1;
            if (position < 0) {
                const ulong win_base = ulong(slot.taped + reach + position) * SEISMIC_WINDOW_STRIDE_1
                    + chan0 * SEISMIC_WINDOW_STRIDE_2;
                for (ulong i = 0; i < count; ++i) {
                    const float input = source[win_base + i * SEISMIC_WINDOW_STRIDE_2];
                    sum[i] = metal::fma(convolution[(chan0 + i) * SEISMIC_CONVOLUTION_STRIDE_0 + tap_off], input, sum[i]);
                }
            } else {
                const ulong src_row = ulong(slot.lo + position);
                device const float *row_scales = scales + src_row * SEISMIC_SCALES_STRIDE_0;
                const float s0 = row_scales[stream * SEISMIC_SCALES_STRIDE_1];
                const float s1 = row_scales[(SEISMIC_DIM_S + stream) * SEISMIC_SCALES_STRIDE_1];
                device const float *val_base = value + src_row * SEISMIC_VALUE_STRIDE_0 + col0 * SEISMIC_VALUE_STRIDE_1;
                for (ulong i = 0; i < count; ++i) {
                    gated[i] = s0 * val_base[i * SEISMIC_VALUE_STRIDE_1];
                    const float input = (gated[i] * s1) * norm_tail[i];
                    sum[i] = metal::fma(convolution[(chan0 + i) * SEISMIC_CONVOLUTION_STRIDE_0 + tap_off], input, sum[i]);
                }
            }
        }
        for (ulong i = 0; i < count; ++i) {
            const float residual = hidden_ptr[i * SEISMIC_HIDDEN_STRIDE_1];
            result.put(out_idx + i * SEISMIC_RESULT_0_STRIDE_1,
                residual + (gated[i] + sum[i] / (1.0f + metal::exp(-sum[i]))));
        }
    }
}
