// causal_short_conv (contract and portable body in short_conv.seismic):
// generalized depthwise causal short convolution with residual add, optional
// per-head dynamic taps, and variance normalization:
//   y = residual_scale * (x + static_conv(x) + dyn_conv(x))
// A thread owns one channel of one row: grid y < M computes row y, grid y = M + b
// publishes slot b's successor window.

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"
#include "lib/recurrent/versions.h"

typedef ELEMENT_OF(SEISMIC_INPUT) InElem;
typedef ELEMENT_OF(SEISMIC_RESULT_0) OutElem;

kernel void causal_short_conv(
    device const uchar *input [[buffer(SEISMIC_BUFFER_INPUT)]],
    device const float *convolution [[buffer(SEISMIC_BUFFER_CONVOLUTION)]],
    device const float *dynamic_taps [[buffer(SEISMIC_BUFFER_DYNAMIC_TAPS)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    SEISMIC_OUT_WINDOW_PARAM(window),
    SEISMIC_RESULT_0_PARAM(output),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    const ulong channel = ulong(group.x) * threadgroup_shape.x + thread_index;
    if (channel >= SEISMIC_DIM_CH)
        return;
    const ulong channels = SEISMIC_DIM_CH;
    const long taps = long(SEISMIC_DIM_C) - 1;
    const ulong hd = SEISMIC_DIM_HD;
    const versions::Slots slots{segments, stop, previous_bank, previous_tape, following_bank};
    const ulong rows = SEISMIC_DIM_M;
    if (group.y >= rows) {
        const versions::Slot slot = versions::slot_of(slots, group.y - rows, seismic_words);
        if (slot.lo == slot.hi) return;
        const element::Out<element::F32> source = versions::bank(window, slot.source, SEISMIC_WINDOW_STRIDE_0,
            seismic_words);
        const element::Out<element::F32> target = versions::bank(window, slot.target, SEISMIC_WINDOW_STRIDE_0,
            seismic_words);
        const long published = taps + versions::tape_rows(slot, long(SEISMIC_DIM_T));
        for (long tap = 0; tap < published; ++tap) {
            const long position = slot.stop + tap - taps;
            const ulong to = ulong(tap) * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2;
            if (position < 0)
                target.move_stored(to, source, ulong(slot.taped + slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1
                    + channel * SEISMIC_WINDOW_STRIDE_2, 1);
            else
                target.put(to, element::at<InElem>(input,
                    ulong(slot.lo + position) * SEISMIC_INPUT_STRIDE_0 + channel * SEISMIC_INPUT_STRIDE_1));
        }
        return;
    }
    const long row = long(group.y);
    const ulong slot_index = versions::slot_index_of_row(slots, row, seismic_words);
    const ulong out = ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + channel * SEISMIC_RESULT_0_STRIDE_1;
    const element::Out<OutElem> result(output);
    if (slot_index == SEISMIC_DIM_B) {
        result.put(out, 0.0f);
        return;
    }
    const versions::Slot slot = versions::slot_of(slots, slot_index, seismic_words);
    device const float *source = versions::bank_view(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words);
    const long local = row - slot.lo;
    const ulong head = hd > 0 ? channel / (channels / hd) : 0;
    float static_sum = 0.0f;
    float dyn_sum = 0.0f;
    for (long tap = 0; tap <= taps; ++tap) {
        const long position = local + tap - taps;
        const float value = position < 0
            ? source[ulong(slot.taped + taps + position) * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2]
            : element::at<InElem>(input,
                ulong(slot.lo + position) * SEISMIC_INPUT_STRIDE_0 + channel * SEISMIC_INPUT_STRIDE_1);
        static_sum = metal::fma(
            convolution[channel * SEISMIC_CONVOLUTION_STRIDE_0 + ulong(tap) * SEISMIC_CONVOLUTION_STRIDE_1],
            value, static_sum);
        if (hd > 0) {
            dyn_sum = metal::fma(
                dynamic_taps[ulong(row) * SEISMIC_DYNAMIC_TAPS_STRIDE_0
                    + head * SEISMIC_DYNAMIC_TAPS_STRIDE_1 + ulong(taps - tap) * SEISMIC_DYNAMIC_TAPS_STRIDE_2],
                value, dyn_sum);
        }
    }
    const float x = element::at<InElem>(input,
        ulong(row) * SEISMIC_INPUT_STRIDE_0 + channel * SEISMIC_INPUT_STRIDE_1);
    const float residual_scale = as_type<float>(uint(SEISMIC_PARAM_RESIDUAL_SCALE));
    result.put(out, (x + static_sum + dyn_sum) * residual_scale);
}
