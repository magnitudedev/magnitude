// causal_short_conv (contract and portable body in short_conv.seismic):
// generalized depthwise causal short convolution with residual add, optional
// per-head dynamic taps, and variance normalization:
//   y = residual_scale * (x + static_conv(x) + dyn_conv(x))

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"
#include "lib/recurrent/versions.cuh"

namespace causal_sconv {

typedef versions::u64 u64;
using element::u8;
using InElem = ELEMENT_OF(SEISMIC_INPUT);
using OutElem = ELEMENT_OF(SEISMIC_RESULT_0);

struct Operands {
    const u8 *input;
    const float *convolution;
    const float *dynamic_taps;
    element::SlabOut<element::F32> window;
    element::Out<OutElem> output;
    versions::Slots slots;
};

__device__ __forceinline__ void publish(const Operands &in, u64 index, u64 channel,
                                        const seismic_words_t &seismic_words_value) {
    const int taps = static_cast<int>(SEISMIC_DIM_C) - 1;
    const u64 bank_elements = SEISMIC_WINDOW_STRIDE_0;
    const versions::Slot slot = versions::slot_of(in.slots, index, seismic_words_value);
    if (slot.lo == slot.hi) return;
    const element::Stored<element::F32> source = versions::bank_view(in.window, slot.source, bank_elements, seismic_words_value);
    const element::Out<element::F32> target = versions::bank(in.window, slot.target, bank_elements, seismic_words_value);
    const int published = taps + versions::tape_rows(slot, static_cast<int>(SEISMIC_DIM_T));
    for (int tap = 0; tap < published; ++tap) {
        const int position = slot.stop + tap - taps;
        const u64 to = static_cast<u64>(tap) * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2;
        if (position < 0)
            target.move_stored(to,
                               source.offset(static_cast<u64>(slot.taped + slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1 +
                                             channel * SEISMIC_WINDOW_STRIDE_2));
        else
            target.put(to, element::at<InElem>(in.input,
                static_cast<u64>(slot.lo + position) * SEISMIC_INPUT_STRIDE_0 + channel * SEISMIC_INPUT_STRIDE_1));
    }
}

__device__ __forceinline__ void convolve(const Operands &in, int row, u64 channel,
                                         const seismic_words_t &seismic_words_value) {
    const int taps = static_cast<int>(SEISMIC_DIM_C) - 1;
    const u64 bank_elements = SEISMIC_WINDOW_STRIDE_0;
    const u64 slot_index = versions::slot_index_of_row(in.slots, row, seismic_words_value);
    const u64 out = static_cast<u64>(row) * SEISMIC_RESULT_0_STRIDE_0 + channel * SEISMIC_RESULT_0_STRIDE_1;
    if (slot_index == SEISMIC_DIM_B) {
        in.output.put(out, 0.0f);
        return;
    }
    const versions::Slot slot = versions::slot_of(in.slots, slot_index, seismic_words_value);
    const element::Stored<element::F32> source = versions::bank_view(in.window, slot.source, bank_elements, seismic_words_value);
    const int local = row - slot.lo;
    const u64 hd = SEISMIC_DIM_HD;
    const u64 head = hd > 0 ? channel / (SEISMIC_DIM_CH / hd) : 0;
    float static_sum = 0.0f;
    float dyn_sum = 0.0f;
    for (int tap = 0; tap <= taps; ++tap) {
        const int position = local + tap - taps;
        const float value =
            position < 0 ? source.at(static_cast<u64>(slot.taped + taps + position) * SEISMIC_WINDOW_STRIDE_1 +
                                     channel * SEISMIC_WINDOW_STRIDE_2)
                         : element::at<InElem>(in.input,
                               static_cast<u64>(slot.lo + position) * SEISMIC_INPUT_STRIDE_0 +
                               channel * SEISMIC_INPUT_STRIDE_1);
        static_sum = __fmaf_rn(
            in.convolution[channel * SEISMIC_CONVOLUTION_STRIDE_0 + static_cast<u64>(tap) * SEISMIC_CONVOLUTION_STRIDE_1],
            value, static_sum);
        if (hd > 0) {
            dyn_sum = __fmaf_rn(
                in.dynamic_taps[static_cast<u64>(row) * SEISMIC_DYNAMIC_TAPS_STRIDE_0 +
                                head * SEISMIC_DYNAMIC_TAPS_STRIDE_1 +
                                static_cast<u64>(taps - tap) * SEISMIC_DYNAMIC_TAPS_STRIDE_2],
                value, dyn_sum);
        }
    }
    const float x = element::at<InElem>(in.input,
        static_cast<u64>(row) * SEISMIC_INPUT_STRIDE_0 + channel * SEISMIC_INPUT_STRIDE_1);
    const float residual_scale = __uint_as_float((unsigned)SEISMIC_PARAM_RESIDUAL_SCALE);
    in.output.put(out, __fmul_rn(x + static_sum + dyn_sum, residual_scale));
}

} // namespace causal_sconv

extern "C" __global__ void causal_short_conv(SEISMIC_KERNEL_PARAMS) {
    typedef versions::u64 u64;
    const causal_sconv::Operands in{
        SEISMIC_PTR(SEISMIC_BUFFER_INPUT),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_CONVOLUTION)),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_DYNAMIC_TAPS)),
        SEISMIC_OUT_WINDOW, SEISMIC_RESULT_0_OUT, VERSIONS_SLOTS()};
    const u64 channel = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= SEISMIC_DIM_CH)
        return;
    const u64 rows = SEISMIC_DIM_M;
    for (u64 item = blockIdx.y; item < rows + SEISMIC_DIM_B; item += gridDim.y) {
        if (item >= rows)
            causal_sconv::publish(in, item - rows, channel, seismic_words_value);
        else
            causal_sconv::convolve(in, static_cast<int>(item), channel, seismic_words_value);
    }
}
