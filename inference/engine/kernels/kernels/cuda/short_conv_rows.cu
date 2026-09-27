// short_conv_rows (contract and portable body in short_conv.seismic): the
// gated causal depthwise convolution of one short-convolution layer. A thread
// owns one channel of one item: item y < M computes row y (zero past the
// slots), item M + b publishes slot b's successor window; the y grid visits
// the items with a stride (it is capped at 65535). Each output is the body's
// tap-ascending F32 FMA chain times the gate, so every row class and backend
// gives the same bits. The successor window is no slot's source, so the
// publishing threads run beside the row threads.

#include "lib/core/activation.cuh"
#include "lib/recurrent/versions.cuh"

namespace short_conv {

typedef versions::u64 u64;

struct Operands {
    const float *projection;
    const float *convolution;
    const versions::u32 *window;
    versions::u8 *gated;
    versions::Slots slots;
};

// Publishes channel `channel` of slot `index`'s successor window.
__device__ __forceinline__ void publish(const Operands &in, u64 index, u64 channel,
                                        const seismic_words_t &seismic_words_value) {
    const int taps = static_cast<int>(SEISMIC_DIM_C) - 1;
    const u64 bank_bytes = SEISMIC_WINDOW_STRIDE_0 * 4;
    const versions::Slot slot = versions::slot_of(in.slots, index, seismic_words_value);
    const float *source =
        reinterpret_cast<const float *>(versions::bank(in.window, slot.source, bank_bytes, seismic_words_value));
    float *target = reinterpret_cast<float *>(versions::bank(in.window, slot.target, bank_bytes, seismic_words_value));
    const int published = taps + versions::tape_rows(slot, seismic_words_value);
    for (int tap = 0; tap < published; ++tap) {
        const int position = slot.stop + tap - taps;
        target[static_cast<u64>(tap) * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2] =
            position < 0 ? source[static_cast<u64>(slot.taped + slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1 +
                                  channel * SEISMIC_WINDOW_STRIDE_2]
                         : in.projection[static_cast<u64>(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0 +
                                         channel * SEISMIC_PROJECTION_STRIDE_1];
    }
}

// Channel `channel` of row `row`'s gated convolution.
__device__ __forceinline__ void convolve(const Operands &in, int row, u64 channel,
                                         const seismic_words_t &seismic_words_value) {
    const int taps = static_cast<int>(SEISMIC_DIM_C) - 1;
    const u64 bank_bytes = SEISMIC_WINDOW_STRIDE_0 * 4;
    const u64 slot_index = versions::slot_index_of_row(in.slots, row, seismic_words_value);
    const u64 out = static_cast<u64>(row) * SEISMIC_RESULT_0_STRIDE_0 + channel * SEISMIC_RESULT_0_STRIDE_1;
    if (slot_index == SEISMIC_DIM_B) {
        element::put<element::Act>(in.gated, out, 0.0f);
        return;
    }
    const versions::Slot slot = versions::slot_of(in.slots, slot_index, seismic_words_value);
    const float *source =
        reinterpret_cast<const float *>(versions::bank(in.window, slot.source, bank_bytes, seismic_words_value));
    const int local = row - slot.lo;
    float sum = 0.0f;
    for (int tap = 0; tap <= taps; ++tap) {
        const int position = local + tap - taps;
        const float value =
            position < 0 ? source[static_cast<u64>(slot.taped + taps + position) * SEISMIC_WINDOW_STRIDE_1 +
                                  channel * SEISMIC_WINDOW_STRIDE_2]
                         : in.projection[static_cast<u64>(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0 +
                                         channel * SEISMIC_PROJECTION_STRIDE_1];
        sum = __fmaf_rn(in.convolution[channel * SEISMIC_CONVOLUTION_STRIDE_0 + tap * SEISMIC_CONVOLUTION_STRIDE_1],
                        value, sum);
    }
    const float gate = in.projection[static_cast<u64>(row) * SEISMIC_PROJECTION_STRIDE_0 +
                                     (SEISMIC_DIM_CH + channel) * SEISMIC_PROJECTION_STRIDE_1];
    element::put<element::Act>(in.gated, out, __fmul_rn(gate, sum));
}

} // namespace short_conv

extern "C" __global__ void short_conv_rows(SEISMIC_KERNEL_PARAMS) {
    typedef versions::u64 u64;
    const short_conv::Operands in{
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PROJECTION)),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_CONVOLUTION)),
        reinterpret_cast<const versions::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW)),
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), VERSIONS_SLOTS()};
    const u64 channel = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= SEISMIC_DIM_CH)
        return;
    const u64 rows = SEISMIC_DIM_M;
    for (u64 item = blockIdx.y; item < rows + SEISMIC_DIM_B; item += gridDim.y) {
        if (item >= rows)
            short_conv::publish(in, item - rows, channel, seismic_words_value);
        else
            short_conv::convolve(in, static_cast<int>(item), channel, seismic_words_value);
    }
}
