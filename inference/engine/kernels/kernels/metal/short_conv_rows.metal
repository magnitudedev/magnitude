// short_conv_rows (contract and portable body in short_conv.seismic): the
// gated causal depthwise convolution of one short-convolution layer. A thread
// owns one channel of one row: grid y < M computes row y (zero past the
// slots), grid y = M + b publishes slot b's successor window. Each output is
// the body's tap-ascending F32 FMA chain times the gate, so every row class
// and backend gives the same bits. The successor window is no slot's source,
// so the publishing threads run beside the row threads.

#include "lib/core/activation.h"
#include "lib/recurrent/versions.h"

kernel void short_conv_rows(
    device const float *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const float *convolution [[buffer(SEISMIC_BUFFER_CONVOLUTION)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    device const ulong *window [[buffer(SEISMIC_BUFFER_WINDOW)]],
    device element::Act::storage *gated [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    const ulong channel = ulong(group.x) * threadgroup_shape.x + thread_index;
    if (channel >= SEISMIC_DIM_CH)
        return;
    const ulong channels = SEISMIC_DIM_CH;
    const long taps = long(SEISMIC_DIM_C) - 1;
    const versions::Slots slots{segments, stop, previous_bank, previous_tape, following_bank};
    const ulong rows = SEISMIC_DIM_M;
    if (group.y >= rows) {
        const versions::Slot slot = versions::slot_of(slots, group.y - rows, seismic_words);
        device const float *source = versions::bank<float>(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words);
        device float *target = versions::bank<float>(window, slot.target, SEISMIC_WINDOW_STRIDE_0, seismic_words);
        const long published = taps + versions::tape_rows(slot, seismic_words);
        for (long tap = 0; tap < published; ++tap) {
            const long position = slot.stop + tap - taps;
            target[ulong(tap) * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2] = position < 0
                ? source[ulong(slot.taped + slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1
                    + channel * SEISMIC_WINDOW_STRIDE_2]
                : projection[ulong(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0
                    + channel * SEISMIC_PROJECTION_STRIDE_1];
        }
        return;
    }
    const long row = long(group.y);
    const ulong slot_index = versions::slot_index_of_row(slots, row, seismic_words);
    device element::Act::storage *out = gated + ulong(row) * SEISMIC_RESULT_0_STRIDE_0
        + channel * SEISMIC_RESULT_0_STRIDE_1;
    if (slot_index == SEISMIC_DIM_B) {
        *out = element::Act::store(0.0f);
        return;
    }
    const versions::Slot slot = versions::slot_of(slots, slot_index, seismic_words);
    device const float *source = versions::bank<float>(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words);
    const long local = row - slot.lo;
    float sum = 0.0f;
    for (long tap = 0; tap <= taps; ++tap) {
        const long position = local + tap - taps;
        const float value = position < 0
            ? source[ulong(slot.taped + taps + position) * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2]
            : projection[ulong(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0
                + channel * SEISMIC_PROJECTION_STRIDE_1];
        sum = metal::fma(convolution[channel * SEISMIC_CONVOLUTION_STRIDE_0 + ulong(tap) * SEISMIC_CONVOLUTION_STRIDE_1],
            value, sum);
    }
    const float gate = projection[ulong(row) * SEISMIC_PROJECTION_STRIDE_0
        + (channels + channel) * SEISMIC_PROJECTION_STRIDE_1];
    *out = element::Act::store(gate * sum);
}
