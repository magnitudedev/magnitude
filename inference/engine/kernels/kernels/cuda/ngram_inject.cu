// ngram_inject (contract and portable body in ngram.seismic): the n-gram
// input's dilated causal convolution over its banked window, added with the
// gated value into the streams. A thread owns 4 contiguous columns of one
// stream (grid z) and one item: item y < M computes row y (a copy of `hidden`
// past the slots), item M + b publishes slot b's successor window; the y grid
// visits the items with a stride (it is capped at 65535). `norm` is hoisted
// once per thread before the item and tap loops.

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
    element::SlabOut<element::F32> window;
    element::Out<element::F32> result;
    versions::Slots slots;
    int dilation;
};

__device__ __forceinline__ void load4_f32(const float *base, u64 stride, float (&out)[4]) {
    if (stride == 1 && ((reinterpret_cast<u64>(base) & 15u) == 0)) {
        element::f32_span<4>(base, out);
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            out[i] = base[static_cast<u64>(i) * stride];
    }
}

__device__ __forceinline__ void load4_stored(const element::Stored<element::F32> &src, u64 index, u64 stride,
                                             float (&out)[4]) {
    if (stride == 1 && ((reinterpret_cast<u64>(src.bytes(index)) & 15u) == 0)) {
        src.span<4>(index, out);
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            out[i] = src.at(index + static_cast<u64>(i) * stride);
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

__device__ __forceinline__ void load4_norm(const versions::u8 *base, u64 index, u64 stride, float (&out)[4]) {
    using Norm = ELEMENT_OF(SEISMIC_NORM);
    constexpr u64 align_mask = static_cast<u64>(4 * Norm::bytes - 1);
    if (stride == 1 && ((reinterpret_cast<u64>(base + index * Norm::bytes) & align_mask) == 0)) {
        element::span<Norm, 4, true>(base, index, out);
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            out[i] = element::at<Norm>(base, index + static_cast<u64>(i) * stride);
    }
}

// u at launch row `row`, stream `stream`, columns `col0..col0+4`, using hoisted `norm4`.
__device__ __forceinline__ void u4(const Operands &in, u64 row, u64 stream, u64 col0, const float (&norm4)[4],
                                   float (&gated4)[4], float (&out)[4],
                                   const seismic_words_t &seismic_words_value) {
    const float *scales = in.scales + row * SEISMIC_SCALES_STRIDE_0;
    const float s0 = scales[stream * SEISMIC_SCALES_STRIDE_1];
    const float s1 = scales[(SEISMIC_DIM_S + stream) * SEISMIC_SCALES_STRIDE_1];
    float v[4];
    load4_f32(in.value + row * SEISMIC_VALUE_STRIDE_0 + col0 * SEISMIC_VALUE_STRIDE_1, SEISMIC_VALUE_STRIDE_1, v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        gated4[i] = s0 * v[i];
        out[i] = (gated4[i] * s1) * norm4[i];
    }
}

// Publishes 4 channels of slot `index`'s successor window.
__device__ __forceinline__ void publish4(const Operands &in, u64 index, u64 stream, u64 col0,
                                         const float (&norm4)[4], const seismic_words_t &seismic_words_value) {
    const int reach = in.dilation * (static_cast<int>(SEISMIC_DIM_K) - 1);
    const u64 bank_elements = SEISMIC_WINDOW_STRIDE_0;
    const u64 chan0 = stream * SEISMIC_DIM_H + col0;
    const versions::Slot slot = versions::slot_of(in.slots, index, seismic_words_value);
    if (slot.lo == slot.hi) return;
    const element::Stored<element::F32> source = versions::bank_view(in.window, slot.source, bank_elements, seismic_words_value);
    const element::Out<element::F32> target = versions::bank(in.window, slot.target, bank_elements, seismic_words_value);
    // The tape holds T = R - reach rows.
    const int published = reach + versions::tape_rows(slot, static_cast<int>(SEISMIC_DIM_R) - reach);
    for (int tap = 0; tap < published; ++tap) {
        const int position = slot.stop + tap - reach;
        const u64 to = static_cast<u64>(tap) * SEISMIC_WINDOW_STRIDE_1 + chan0 * SEISMIC_WINDOW_STRIDE_2;
        if (position < 0) {
            const u64 from = static_cast<u64>(slot.taped + slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1 +
                             chan0 * SEISMIC_WINDOW_STRIDE_2;
            if (SEISMIC_WINDOW_STRIDE_2 == 1) {
                target.move_stored(to, source.offset(from), 4);
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    target.move_stored(to + static_cast<u64>(i) * SEISMIC_WINDOW_STRIDE_2,
                                       source.offset(from + static_cast<u64>(i) * SEISMIC_WINDOW_STRIDE_2), 1);
            }
        } else {
            float gated[4];
            float u_val[4];
            u4(in, static_cast<u64>(slot.lo + position), stream, col0, norm4, gated, u_val, seismic_words_value);
            store4_f32(target, to, SEISMIC_WINDOW_STRIDE_2, u_val);
        }
    }
}

// Publishes tail channels (1..3) of slot `index`'s successor window.
__device__ __forceinline__ void publish_tail(const Operands &in, u64 index, u64 stream, u64 col0, u64 count,
                                             const float (&norm4)[4], const seismic_words_t &seismic_words_value) {
    const int reach = in.dilation * (static_cast<int>(SEISMIC_DIM_K) - 1);
    const u64 bank_elements = SEISMIC_WINDOW_STRIDE_0;
    const u64 chan0 = stream * SEISMIC_DIM_H + col0;
    const versions::Slot slot = versions::slot_of(in.slots, index, seismic_words_value);
    if (slot.lo == slot.hi) return;
    const element::Stored<element::F32> source = versions::bank_view(in.window, slot.source, bank_elements, seismic_words_value);
    const element::Out<element::F32> target = versions::bank(in.window, slot.target, bank_elements, seismic_words_value);
    const int published = reach + versions::tape_rows(slot, static_cast<int>(SEISMIC_DIM_R) - reach);
    for (int tap = 0; tap < published; ++tap) {
        const int position = slot.stop + tap - reach;
        const u64 to = static_cast<u64>(tap) * SEISMIC_WINDOW_STRIDE_1 + chan0 * SEISMIC_WINDOW_STRIDE_2;
        if (position < 0) {
            const u64 from = static_cast<u64>(slot.taped + slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1 +
                             chan0 * SEISMIC_WINDOW_STRIDE_2;
            for (u64 i = 0; i < count; ++i)
                target.move_stored(to + i * SEISMIC_WINDOW_STRIDE_2,
                                   source.offset(from + i * SEISMIC_WINDOW_STRIDE_2), 1);
        } else {
            const u64 src_row = static_cast<u64>(slot.lo + position);
            const float *scales = in.scales + src_row * SEISMIC_SCALES_STRIDE_0;
            const float s0 = scales[stream * SEISMIC_SCALES_STRIDE_1];
            const float s1 = scales[(SEISMIC_DIM_S + stream) * SEISMIC_SCALES_STRIDE_1];
            const float *val_base = in.value + src_row * SEISMIC_VALUE_STRIDE_0 + col0 * SEISMIC_VALUE_STRIDE_1;
            for (u64 i = 0; i < count; ++i) {
                const float g = s0 * val_base[i * SEISMIC_VALUE_STRIDE_1];
                target.put(to + i * SEISMIC_WINDOW_STRIDE_2, (g * s1) * norm4[i]);
            }
        }
    }
}

// 4 channels of row `row`.
__device__ __forceinline__ void inject4(const Operands &in, int row, u64 stream, u64 col0,
                                        const float (&norm4)[4], const seismic_words_t &seismic_words_value) {
    const int dilation = in.dilation;
    const int reach = dilation * (static_cast<int>(SEISMIC_DIM_K) - 1);
    const u64 bank_elements = SEISMIC_WINDOW_STRIDE_0;
    const u64 chan0 = stream * SEISMIC_DIM_H + col0;
    const float *hidden_ptr = in.hidden + static_cast<u64>(row) * SEISMIC_HIDDEN_STRIDE_0 + chan0 * SEISMIC_HIDDEN_STRIDE_1;
    const u64 out_idx = static_cast<u64>(row) * SEISMIC_RESULT_0_STRIDE_0 + chan0 * SEISMIC_RESULT_0_STRIDE_1;
    const u64 slot_index = versions::slot_index_of_row(in.slots, row, seismic_words_value);
    if (slot_index == SEISMIC_DIM_B) {
        float residual[4];
        load4_f32(hidden_ptr, SEISMIC_HIDDEN_STRIDE_1, residual);
        store4_f32(in.result, out_idx, SEISMIC_RESULT_0_STRIDE_1, residual);
        return;
    }
    const versions::Slot slot = versions::slot_of(in.slots, slot_index, seismic_words_value);
    const element::Stored<element::F32> source = versions::bank_view(in.window, slot.source, bank_elements, seismic_words_value);
    const int local = row - slot.lo;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float gated[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int tap = 0; tap < static_cast<int>(SEISMIC_DIM_K); ++tap) {
        const int position = local + dilation * tap - reach;
        float input[4];
        if (position < 0) {
            load4_stored(source,
                         static_cast<u64>(slot.taped + reach + position) * SEISMIC_WINDOW_STRIDE_1 +
                             chan0 * SEISMIC_WINDOW_STRIDE_2,
                         SEISMIC_WINDOW_STRIDE_2, input);
        } else {
            u4(in, static_cast<u64>(slot.lo + position), stream, col0, norm4, gated, input, seismic_words_value);
        }
        const u64 tap_off = static_cast<u64>(tap) * SEISMIC_CONVOLUTION_STRIDE_1;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            sum[i] = __fmaf_rn(in.convolution[(chan0 + static_cast<u64>(i)) * SEISMIC_CONVOLUTION_STRIDE_0 + tap_off],
                               input[i], sum[i]);
    }
    // tap == K - 1 has position == local >= 0 and slot.lo + position == row, so
    // `gated` already holds scales[row, stream] * value[row, col0..col0+4].
    float residual[4];
    load4_f32(hidden_ptr, SEISMIC_HIDDEN_STRIDE_1, residual);
    float out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        out[i] = residual[i] + (gated[i] + sum[i] / (1.0f + expf(-sum[i])));
    store4_f32(in.result, out_idx, SEISMIC_RESULT_0_STRIDE_1, out);
}

// Tail channels (1..3) of row `row`.
__device__ __forceinline__ void inject_tail(const Operands &in, int row, u64 stream, u64 col0, u64 count,
                                            const float (&norm4)[4], const seismic_words_t &seismic_words_value) {
    const int dilation = in.dilation;
    const int reach = dilation * (static_cast<int>(SEISMIC_DIM_K) - 1);
    const u64 bank_elements = SEISMIC_WINDOW_STRIDE_0;
    const u64 chan0 = stream * SEISMIC_DIM_H + col0;
    const float *hidden_ptr = in.hidden + static_cast<u64>(row) * SEISMIC_HIDDEN_STRIDE_0 + chan0 * SEISMIC_HIDDEN_STRIDE_1;
    const u64 out_idx = static_cast<u64>(row) * SEISMIC_RESULT_0_STRIDE_0 + chan0 * SEISMIC_RESULT_0_STRIDE_1;
    const u64 slot_index = versions::slot_index_of_row(in.slots, row, seismic_words_value);
    if (slot_index == SEISMIC_DIM_B) {
        for (u64 i = 0; i < count; ++i)
            in.result.put(out_idx + i * SEISMIC_RESULT_0_STRIDE_1, hidden_ptr[i * SEISMIC_HIDDEN_STRIDE_1]);
        return;
    }
    const versions::Slot slot = versions::slot_of(in.slots, slot_index, seismic_words_value);
    const element::Stored<element::F32> source = versions::bank_view(in.window, slot.source, bank_elements, seismic_words_value);
    const int local = row - slot.lo;
    float sum[3] = {0.0f, 0.0f, 0.0f};
    float gated[3] = {0.0f, 0.0f, 0.0f};
    for (int tap = 0; tap < static_cast<int>(SEISMIC_DIM_K); ++tap) {
        const int position = local + dilation * tap - reach;
        const u64 tap_off = static_cast<u64>(tap) * SEISMIC_CONVOLUTION_STRIDE_1;
        if (position < 0) {
            const u64 win_base = static_cast<u64>(slot.taped + reach + position) * SEISMIC_WINDOW_STRIDE_1 +
                                 chan0 * SEISMIC_WINDOW_STRIDE_2;
            for (u64 i = 0; i < count; ++i) {
                const float input = source.at(win_base + i * SEISMIC_WINDOW_STRIDE_2);
                sum[i] = __fmaf_rn(in.convolution[(chan0 + i) * SEISMIC_CONVOLUTION_STRIDE_0 + tap_off], input, sum[i]);
            }
        } else {
            const u64 src_row = static_cast<u64>(slot.lo + position);
            const float *scales = in.scales + src_row * SEISMIC_SCALES_STRIDE_0;
            const float s0 = scales[stream * SEISMIC_SCALES_STRIDE_1];
            const float s1 = scales[(SEISMIC_DIM_S + stream) * SEISMIC_SCALES_STRIDE_1];
            const float *val_base = in.value + src_row * SEISMIC_VALUE_STRIDE_0 + col0 * SEISMIC_VALUE_STRIDE_1;
            for (u64 i = 0; i < count; ++i) {
                gated[i] = s0 * val_base[i * SEISMIC_VALUE_STRIDE_1];
                const float input = (gated[i] * s1) * norm4[i];
                sum[i] = __fmaf_rn(in.convolution[(chan0 + i) * SEISMIC_CONVOLUTION_STRIDE_0 + tap_off], input, sum[i]);
            }
        }
    }
    for (u64 i = 0; i < count; ++i) {
        const float residual = hidden_ptr[i * SEISMIC_HIDDEN_STRIDE_1];
        in.result.put(out_idx + i * SEISMIC_RESULT_0_STRIDE_1,
                      residual + (gated[i] + sum[i] / (1.0f + expf(-sum[i]))));
    }
}

} // namespace ngram

extern "C" __global__ void ngram_inject(SEISMIC_KERNEL_PARAMS) {
    typedef versions::u64 u64;
    using Norm = ELEMENT_OF(SEISMIC_NORM);
    const u64 col0 = (static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (col0 >= SEISMIC_DIM_H)
        return;
    const u64 stream = static_cast<u64>(blockIdx.z);
    const ngram::Operands in{
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN)),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCALES)),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_VALUE)),
        SEISMIC_PTR(SEISMIC_BUFFER_NORM),
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_CONVOLUTION)),
        SEISMIC_OUT_WINDOW, SEISMIC_RESULT_0_OUT, VERSIONS_SLOTS(),
        static_cast<int>(static_cast<unsigned>(SEISMIC_PARAM_DILATION))};
    const bool full4 = (SEISMIC_DIM_H % 4 == 0) || (col0 + 4 <= SEISMIC_DIM_H);
    const u64 norm_idx = stream * SEISMIC_NORM_STRIDE_0 + col0 * SEISMIC_NORM_STRIDE_1;
    float norm4[4];
    if (full4) {
        ngram::load4_norm(in.norm, norm_idx, SEISMIC_NORM_STRIDE_1, norm4);
    } else {
        const u64 count = SEISMIC_DIM_H - col0;
        for (u64 i = 0; i < count; ++i)
            norm4[i] = element::at<Norm>(in.norm, norm_idx + i * SEISMIC_NORM_STRIDE_1);
    }
    const u64 rows = SEISMIC_DIM_M;
    for (u64 item = blockIdx.y; item < rows + SEISMIC_DIM_B; item += gridDim.y) {
        if (item >= rows) {
            if (full4)
                ngram::publish4(in, item - rows, stream, col0, norm4, seismic_words_value);
            else
                ngram::publish_tail(in, item - rows, stream, col0, SEISMIC_DIM_H - col0, norm4, seismic_words_value);
        } else {
            if (full4)
                ngram::inject4(in, static_cast<int>(item), stream, col0, norm4, seismic_words_value);
            else
                ngram::inject_tail(in, static_cast<int>(item), stream, col0, SEISMIC_DIM_H - col0, norm4, seismic_words_value);
        }
    }
}
