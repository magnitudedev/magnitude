// The causal depthwise convolution of the Metal gated-delta entries that bind
// the window arena and the convolution taps (`gated_delta_step`,
// `gated_delta_chunk`; contracts in recurrent.seismic): the convolution input
// rows, the convolution with SiLU, the `advance_rows` input that convolves,
// and the successor window publication. Channels of the projection and window
// rows are contiguous (unit stride).

#include "recurrent.h"

namespace recurrent {

#define RECURRENT_TAPS SEISMIC_DIM_C

// The input rows of taps 0..C. A struct, so that it is passed by a thread
// reference: Metal 4.1 refuses a reference to an array of device pointers
// that spells both address spaces on one declarator.
struct Taps {
    device const Storage *rows[RECURRENT_TAPS];
};

// The convolution input rows of slot-local row `local` for taps 0..C: the
// raw input row at slot-local `position` is the source version's window row
// before the slot, read through its const view, and the projection's after.
inline Taps taps(device const Storage *projection, device const slab::Table<activation::Act> *window, Slot slot,
    long local, constant ulong *seismic_words) {
    device const Storage *source = versions::bank_view(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words);
    Taps taps;
    RECURRENT_UNROLL for (uint tap = 0; tap < RECURRENT_TAPS; ++tap) {
        const long position = local + long(tap) - long(RECURRENT_TAPS - 1);
        taps.rows[tap] = position < 0
            ? source + ulong(slot.taped + long(RECURRENT_TAPS) - 1 + position) * SEISMIC_WINDOW_STRIDE_1
            : projection + ulong(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0;
    }
    return taps;
}

// SiLU of the causal depthwise convolution of `channel` over `rows`.
inline float convolve(device const float *convolution, thread const Taps &taps,
    ulong channel, constant ulong *seismic_words) {
    float sum = 0.0f;
    RECURRENT_UNROLL for (uint tap = 0; tap < RECURRENT_TAPS; ++tap) {
        sum = metal::fma(convolution[channel * SEISMIC_CONVOLUTION_STRIDE_0 + tap * SEISMIC_CONVOLUTION_STRIDE_1],
            activation::Act::load(taps.rows[tap][channel]), sum);
    }
    return sum / (1.0f + metal::exp(-sum));
}

// The same for the four channels `channel`..`channel + 3`.
inline float4 convolve4(device const float *convolution, thread const Taps &taps,
    ulong channel, constant ulong *seismic_words) {
    float4 sum = 0.0f;
    RECURRENT_UNROLL for (uint tap = 0; tap < RECURRENT_TAPS; ++tap) {
        float4 weights;
        RECURRENT_UNROLL for (uint e = 0; e < 4; ++e) {
            weights[e] = convolution[(channel + e) * SEISMIC_CONVOLUTION_STRIDE_0 + tap * SEISMIC_CONVOLUTION_STRIDE_1];
        }
        sum = metal::fma(weights, activation::Act::load4(taps.rows[tap] + channel), sum);
    }
    return sum / (1.0f + metal::exp(-sum));
}

// The `advance_rows` input that forms a row's channels from the projection
// and the source window (`convolve`).
struct Convolving {
    device const Storage *projection;
    device const slab::Table<activation::Act> *window;
    device const float *convolution;
    Slot slot;
    struct Row {
        Taps taps;
        device const float *convolution;
        float at(ulong channel, constant ulong *seismic_words) const {
            // A member reached through `this` has no stated address space
            // under Metal 4.1; the copy is a thread object.
            const Taps held = taps;
            return convolve(convolution, held, channel, seismic_words);
        }
    };
    Row row(long row, constant ulong *seismic_words) const {
        return Row{taps(projection, window, slot, row - slot.lo, seismic_words), convolution};
    }
};

// Channel index for segment `seg` (0 = value head, 1 + 2*k = q head, 2 + 2*k = k head)
// at element `offset` within the head.
inline ulong window_channel(ulong head, ulong seg, ulong offset, ulong key_heads, ulong value_heads, ulong width) {
    if (seg == 0) {
        return (2 * key_heads + head) * width + offset;
    }
    const ulong key_seg = seg - 1;
    const ulong owner = head + (key_seg >> 1) * value_heads;
    return ((key_seg & 1) ? key_heads + owner : owner) * width + offset;
}

// Publishes value head `head`'s share of the slot's successor window (the
// C - 1 raw rows before the publication row, then the raw rows of its tape):
// its value channels and the q/k channels of the key heads congruent to it.
// Thread `thread_index` of `threads` copies an even share in 16-byte chunks
// when channels are contiguous, falling back to scalar copies otherwise.
inline void publish_window(device const Storage *projection, device const slab::Table<activation::Act> *window,
    Slot slot, ulong head, uint thread_index, uint threads, constant ulong *seismic_words) {
    constexpr ulong WIDTH = SEISMIC_DIM_W;
    constexpr ulong VEC = 16 / activation::Act::bytes;
    constexpr ulong CHUNKS = WIDTH / VEC;
    static_assert(WIDTH % VEC == 0, "head width divides into 16-byte chunks");
    const ulong key_heads = SEISMIC_DIM_NK;
    const ulong value_heads = SEISMIC_DIM_NV;
    const long taps = long(RECURRENT_TAPS) - 1;
    const long rows = taps + versions::tape_rows(slot, long(SEISMIC_DIM_T));
    if (rows <= 0) {
        return;
    }
    const element::Out<activation::Act> target_window =
        versions::bank(window, slot.target, SEISMIC_WINDOW_STRIDE_0, seismic_words);
    const element::Out<activation::Act> source_window =
        versions::bank(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words);
    const ulong owned = key_heads > head ? (key_heads - head - 1) / value_heads + 1 : 0;
    const ulong segments = 1 + 2 * owned;
    const ulong total_segments = ulong(rows) * segments;
    if (SEISMIC_WINDOW_STRIDE_2 == 1 && SEISMIC_PROJECTION_STRIDE_1 == 1) {
        const ulong total = total_segments * CHUNKS;
        for (ulong item = thread_index; item < total; item += threads) {
            const ulong seg_id = item / CHUNKS;
            const ulong offset = (item % CHUNKS) * VEC;
            const ulong tap = owned == 0 ? seg_id : (owned == 1 ? seg_id / 3 : seg_id / segments);
            const ulong seg = owned == 0 ? 0 : seg_id - tap * segments;
            const ulong channel = window_channel(head, seg, offset, key_heads, value_heads, WIDTH);
            const long position = slot.stop + long(tap) - taps;
            const ulong to = tap * SEISMIC_WINDOW_STRIDE_1 + channel;
            // Both the source window and the projection are stored in `Act`: moved without decoding.
            if (position < 0)
                target_window.move_stored(to, source_window,
                    ulong(slot.taped + slot.stop + long(tap)) * SEISMIC_WINDOW_STRIDE_1 + channel, VEC);
            else
                target_window.copy_stored(to, projection,
                    ulong(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0 + channel, VEC);
        }
        return;
    }
    const ulong total = total_segments * WIDTH;
    for (ulong item = thread_index; item < total; item += threads) {
        const ulong seg_id = item / WIDTH;
        const ulong offset = item % WIDTH;
        const ulong tap = owned == 0 ? seg_id : (owned == 1 ? seg_id / 3 : seg_id / segments);
        const ulong seg = owned == 0 ? 0 : seg_id - tap * segments;
        const ulong channel = window_channel(head, seg, offset, key_heads, value_heads, WIDTH);
        const long position = slot.stop + long(tap) - taps;
        const ulong to = tap * SEISMIC_WINDOW_STRIDE_1 + channel * SEISMIC_WINDOW_STRIDE_2;
        if (position < 0)
            target_window.move_stored(to, source_window,
                ulong(slot.taped + slot.stop + long(tap)) * SEISMIC_WINDOW_STRIDE_1
                + channel * SEISMIC_WINDOW_STRIDE_2, 1);
        else
            target_window.copy_stored(to, projection,
                ulong(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0
                + channel * SEISMIC_PROJECTION_STRIDE_1, 1);
    }
}

} // namespace recurrent
