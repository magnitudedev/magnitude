// ngram_inject (contract and portable body in ngram.seismic): the n-gram
// input's dilated causal convolution over its banked window, added with the
// gated value into the streams. A thread owns one channel of one item: item
// y < M computes row y (a copy of `hidden` past the slots), item M + b
// publishes slot b's successor window; the y grid visits the items with a
// stride (it is capped at 65535). A launch row's u value is formed from the
// scales, the value row and the norm, as the body forms it.

#include "lib/core/activation.cuh"
#include "lib/recurrent/versions.cuh"

namespace ngram {

typedef versions::u64 u64;

struct Operands {
    const float *hidden;
    const float *scales;
    const float *value;
    const versions::u8 *norm;
    const float *convolution;
    const versions::u32 *window;
    float *result;
    versions::Slots slots;
    int dilation;
};

// u at launch row `row`, channel (`stream`, `column`).
__device__ __forceinline__ float u(const Operands &in, u64 row, u64 stream, u64 column,
                                   const seismic_words_t &seismic_words_value) {
    using Norm = ELEMENT_OF(SEISMIC_NORM);
    const float *scales = in.scales + row * SEISMIC_SCALES_STRIDE_0;
    return scales[stream * SEISMIC_SCALES_STRIDE_1] * in.value[row * SEISMIC_VALUE_STRIDE_0 + column * SEISMIC_VALUE_STRIDE_1] *
           scales[(SEISMIC_DIM_S + stream) * SEISMIC_SCALES_STRIDE_1] *
           element::at<Norm>(in.norm, stream * SEISMIC_NORM_STRIDE_0 + column * SEISMIC_NORM_STRIDE_1);
}

// Publishes channel `channel` of slot `index`'s successor window.
__device__ __forceinline__ void publish(const Operands &in, u64 index, u64 channel,
                                        const seismic_words_t &seismic_words_value) {
    const int reach = in.dilation * (static_cast<int>(SEISMIC_DIM_K) - 1);
    const u64 bank_bytes = SEISMIC_WINDOW_STRIDE_0 * 4;
    const u64 stream = channel / SEISMIC_DIM_H;
    const u64 column = channel % SEISMIC_DIM_H;
    const versions::Slot slot = versions::slot_of(in.slots, index, seismic_words_value);
    const float *source =
        reinterpret_cast<const float *>(versions::bank(in.window, slot.source, bank_bytes, seismic_words_value));
    float *target = reinterpret_cast<float *>(versions::bank(in.window, slot.target, bank_bytes, seismic_words_value));
    // The tape holds T = R - reach rows.
    const int published = reach + versions::tape_rows(slot, static_cast<int>(SEISMIC_DIM_R) - reach);
    for (int tap = 0; tap < published; ++tap) {
        const int position = slot.stop + tap - reach;
        target[static_cast<u64>(tap) * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2] =
            position < 0 ? source[static_cast<u64>(slot.taped + slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1 +
                                  channel * SEISMIC_WINDOW_STRIDE_2]
                         : u(in, static_cast<u64>(slot.lo + position), stream, column, seismic_words_value);
    }
}

// Channel `channel` of row `row`.
__device__ __forceinline__ void inject(const Operands &in, int row, u64 channel,
                                       const seismic_words_t &seismic_words_value) {
    const int dilation = in.dilation;
    const int reach = dilation * (static_cast<int>(SEISMIC_DIM_K) - 1);
    const u64 bank_bytes = SEISMIC_WINDOW_STRIDE_0 * 4;
    const u64 stream = channel / SEISMIC_DIM_H;
    const u64 column = channel % SEISMIC_DIM_H;
    const float residual = in.hidden[static_cast<u64>(row) * SEISMIC_HIDDEN_STRIDE_0 + channel * SEISMIC_HIDDEN_STRIDE_1];
    float *out = in.result + static_cast<u64>(row) * SEISMIC_RESULT_0_STRIDE_0 + channel * SEISMIC_RESULT_0_STRIDE_1;
    const u64 slot_index = versions::slot_index_of_row(in.slots, row, seismic_words_value);
    if (slot_index == SEISMIC_DIM_B) {
        *out = residual;
        return;
    }
    const versions::Slot slot = versions::slot_of(in.slots, slot_index, seismic_words_value);
    const float *source =
        reinterpret_cast<const float *>(versions::bank(in.window, slot.source, bank_bytes, seismic_words_value));
    const int local = row - slot.lo;
    float sum = 0.0f;
    for (int tap = 0; tap < static_cast<int>(SEISMIC_DIM_K); ++tap) {
        const int position = local + dilation * tap - reach;
        const float input =
            position < 0 ? source[static_cast<u64>(slot.taped + reach + position) * SEISMIC_WINDOW_STRIDE_1 +
                                  channel * SEISMIC_WINDOW_STRIDE_2]
                         : u(in, static_cast<u64>(slot.lo + position), stream, column, seismic_words_value);
        sum = __fmaf_rn(in.convolution[channel * SEISMIC_CONVOLUTION_STRIDE_0 + tap * SEISMIC_CONVOLUTION_STRIDE_1],
                        input, sum);
    }
    const float gated = in.scales[static_cast<u64>(row) * SEISMIC_SCALES_STRIDE_0 + stream * SEISMIC_SCALES_STRIDE_1] *
                        in.value[static_cast<u64>(row) * SEISMIC_VALUE_STRIDE_0 + column * SEISMIC_VALUE_STRIDE_1];
    *out = residual + (gated + sum / (1.0f + expf(-sum)));
}

} // namespace ngram

extern "C" __global__ void ngram_inject(SEISMIC_KERNEL_PARAMS) {
    typedef versions::u64 u64;
    const ngram::Operands in{
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALES)),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_VALUE)),
        SEISMIC_PTR(SEISMIC_BUFFER_NORM),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_CONVOLUTION)),
        reinterpret_cast<const versions::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW)),
        reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)), VERSIONS_SLOTS(),
        static_cast<int>(static_cast<unsigned>(SEISMIC_PARAM_DILATION))};
    const u64 channel = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= SEISMIC_DIM_S * SEISMIC_DIM_H)
        return;
    const u64 rows = SEISMIC_DIM_M;
    for (u64 item = blockIdx.y; item < rows + SEISMIC_DIM_B; item += gridDim.y) {
        if (item >= rows)
            ngram::publish(in, item - rows, channel, seismic_words_value);
        else
            ngram::inject(in, static_cast<int>(item), channel, seismic_words_value);
    }
}
