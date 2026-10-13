// ngram_inject (contract and portable body in ngram.seismic): the n-gram
// input's dilated causal convolution over its banked window, added with the
// gated value into the streams. A thread owns one channel of one row: grid
// y < M computes row y (a copy of `hidden` past the slots), grid y = M + b
// publishes slot b's successor window. A launch row's u value is formed from
// the scales, the value row and the norm, as the body forms it.
#include "lib/core/activation.h"
#include "lib/recurrent/versions.h"

typedef ELEMENT_OF(SEISMIC_NORM) Norm;

namespace ngram {

struct Operands {
    device const float *scales;
    device const float *value;
    device const uchar *norm;
};

// u at launch row `row`, channel (`stream`, `column`).
inline float u(const Operands in, ulong row, ulong stream, ulong column, constant ulong *seismic_words) {
    device const float *scales = in.scales + row * SEISMIC_SCALES_STRIDE_0;
    return scales[stream * SEISMIC_SCALES_STRIDE_1] * in.value[row * SEISMIC_VALUE_STRIDE_0 + column * SEISMIC_VALUE_STRIDE_1]
        * scales[(SEISMIC_DIM_S + stream) * SEISMIC_SCALES_STRIDE_1]
        * element::at<Norm>(in.norm, stream * SEISMIC_NORM_STRIDE_0 + column * SEISMIC_NORM_STRIDE_1);
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
    device const ulong *window [[buffer(SEISMIC_BUFFER_WINDOW)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    const ulong channel = ulong(group.x) * threadgroup_shape.x + thread_index;
    if (channel >= SEISMIC_DIM_S * SEISMIC_DIM_H)
        return;
    const ulong stream = channel / SEISMIC_DIM_H;
    const ulong column = channel % SEISMIC_DIM_H;
    const long dilation = long(int(uint(SEISMIC_PARAM_DILATION)));
    const long reach = dilation * (long(SEISMIC_DIM_K) - 1);
    const ngram::Operands in{scales, value, norm};
    const versions::Slots slots{segments, stop, previous_bank, previous_tape, following_bank};
    const ulong rows = SEISMIC_DIM_M;
    if (group.y >= rows) {
        const versions::Slot slot = versions::slot_of(slots, group.y - rows, seismic_words);
        device const float *source = versions::bank<float>(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words);
        device float *target = versions::bank<float>(window, slot.target, SEISMIC_WINDOW_STRIDE_0, seismic_words);
        // The tape holds T = R - reach rows.
        const long published = reach + versions::tape_rows(slot, long(SEISMIC_DIM_R) - reach);
        for (long tap = 0; tap < published; ++tap) {
            const long position = slot.stop + tap - reach;
            target[ulong(tap) * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2] = position < 0
                ? source[ulong(slot.taped + slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2]
                : ngram::u(in, ulong(slot.lo + position), stream, column, seismic_words);
        }
        return;
    }
    const long row = long(group.y);
    const float residual = hidden[ulong(row) * SEISMIC_HIDDEN_STRIDE_0 + channel * SEISMIC_HIDDEN_STRIDE_1];
    device float *out = result + ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + channel * SEISMIC_RESULT_0_STRIDE_1;
    const ulong slot_index = versions::slot_index_of_row(slots, row, seismic_words);
    if (slot_index == SEISMIC_DIM_B) {
        *out = residual;
        return;
    }
    const versions::Slot slot = versions::slot_of(slots, slot_index, seismic_words);
    device const float *source = versions::bank<float>(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words);
    const long local = row - slot.lo;
    float sum = 0.0f;
    for (long tap = 0; tap < long(SEISMIC_DIM_K); ++tap) {
        const long position = local + dilation * tap - reach;
        const float input = position < 0
            ? source[ulong(slot.taped + reach + position) * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2]
            : ngram::u(in, ulong(slot.lo + position), stream, column, seismic_words);
        sum = metal::fma(convolution[channel * SEISMIC_CONVOLUTION_STRIDE_0 + ulong(tap) * SEISMIC_CONVOLUTION_STRIDE_1],
            input, sum);
    }
    const float gated = scales[ulong(row) * SEISMIC_SCALES_STRIDE_0 + stream * SEISMIC_SCALES_STRIDE_1]
        * value[ulong(row) * SEISMIC_VALUE_STRIDE_0 + column * SEISMIC_VALUE_STRIDE_1];
    *out = residual + (gated + sum / (1.0f + metal::exp(-sum)));
}
