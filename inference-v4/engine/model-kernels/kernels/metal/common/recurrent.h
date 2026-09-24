// Shared pieces of the Metal gated-delta entries (`qwen_recurrent_step`,
// `qwen_recurrent_chunk`; contracts in recurrent_stages.seismic): slot and
// bank lookup, the convolution taps and the causal convolution with SiLU,
// the gates, piece splitting and the successor window publication. The
// kernels keep their own loop structure (the step convolves per lane column,
// the chunk per channel quad) and call these. Channels of the projection and
// window rows are contiguous (unit stride).

#include "common/element.h"

namespace recurrent {

typedef element::Act::storage Storage;

#define RECURRENT_TAPS SEISMIC_DIM_C
#define RECURRENT_UNROLL _Pragma("clang loop unroll(full)")

// One slot's rows [lo, hi), its publication row count and its banks.
struct Slot {
    long lo;
    long hi;
    long stop;
    ulong source;
    ulong target;
};

inline Slot slot_of(device const int *segments, device const int *stop, device const int *previous_bank,
    device const int *following_bank, ulong slot, constant ulong *seismic_words) {
    Slot result;
    result.lo = segments[slot * SEISMIC_SEGMENTS_STRIDE_0];
    result.hi = segments[slot * SEISMIC_SEGMENTS_STRIDE_0 + SEISMIC_SEGMENTS_STRIDE_1];
    result.stop = stop[slot * SEISMIC_STOP_STRIDE_0];
    result.source = ulong(previous_bank[slot * SEISMIC_PREVIOUS_BANK_STRIDE_0]);
    result.target = ulong(following_bank[slot * SEISMIC_FOLLOWING_BANK_STRIDE_0]);
    return result;
}

// The key head of value head `head`.
inline ulong key_head(ulong head, constant ulong *seismic_words) {
    return SEISMIC_PARAM_GROUPED != 0 ? head * SEISMIC_DIM_NK / SEISMIC_DIM_NV : head % SEISMIC_DIM_NK;
}

// The convolution input rows of `row` (slot-local `local`) for taps 0..C:
// the window of bank `source` before the slot, the projection after.
inline void taps(device const Storage *projection, device const Storage *window, ulong source, long row,
    long local, thread device const Storage *(&rows)[RECURRENT_TAPS], constant ulong *seismic_words) {
    RECURRENT_UNROLL for (uint tap = 0; tap < RECURRENT_TAPS; ++tap) {
        const long position = local + long(tap) - long(RECURRENT_TAPS - 1);
        rows[tap] = position < 0
            ? window + source * SEISMIC_WINDOW_STRIDE_0 + ulong(local + long(tap)) * SEISMIC_WINDOW_STRIDE_1
            : projection + ulong(row + long(tap) - long(RECURRENT_TAPS - 1)) * SEISMIC_PROJECTION_STRIDE_0;
    }
}

// SiLU of the causal depthwise convolution of `channel` over `rows`.
inline float convolve(device const float *convolution, thread device const Storage *const (&rows)[RECURRENT_TAPS],
    ulong channel, constant ulong *seismic_words) {
    float sum = 0.0f;
    RECURRENT_UNROLL for (uint tap = 0; tap < RECURRENT_TAPS; ++tap) {
        sum = metal::fma(convolution[channel * SEISMIC_CONVOLUTION_STRIDE_0 + tap * SEISMIC_CONVOLUTION_STRIDE_1],
            element::Act::load(rows[tap][channel]), sum);
    }
    return sum / (1.0f + metal::exp(-sum));
}

// The same for the four channels `channel`..`channel + 3`.
inline float4 convolve4(device const float *convolution, thread device const Storage *const (&rows)[RECURRENT_TAPS],
    ulong channel, constant ulong *seismic_words) {
    float4 sum = 0.0f;
    RECURRENT_UNROLL for (uint tap = 0; tap < RECURRENT_TAPS; ++tap) {
        float4 weights;
        RECURRENT_UNROLL for (uint e = 0; e < 4; ++e) {
            weights[e] = convolution[(channel + e) * SEISMIC_CONVOLUTION_STRIDE_0 + tap * SEISMIC_CONVOLUTION_STRIDE_1];
        }
        sum = metal::fma(weights, element::Act::load4(rows[tap] + channel), sum);
    }
    return sum / (1.0f + metal::exp(-sum));
}

// beta = sigmoid(b) and the log decay rate * softplus(alpha + time_bias).
struct Gates {
    float beta;
    float log_decay;
};

inline Gates gates(float alpha, float beta_input, float rate, float time_bias) {
    const float shifted = alpha + time_bias;
    const float softplus = metal::max(shifted, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(shifted)));
    Gates result;
    result.beta = 1.0f / (1.0f + metal::exp(-beta_input));
    result.log_decay = rate * softplus;
    return result;
}

// A slot's pieces: its rows split into runs of at most PIECE rows, with a run
// boundary at the publication row lo + stop. The pieces before that row:
template <long PIECE>
inline ulong pieces_before(Slot slot) {
    return ulong(slot.stop + PIECE - 1) / PIECE;
}

template <long PIECE>
inline ulong piece_count(Slot slot) {
    return pieces_before<PIECE>(slot) + ulong(slot.hi - slot.lo - slot.stop + PIECE - 1) / PIECE;
}

// Rows [first, first + length) of the slot's `piece`-th piece.
template <long PIECE>
inline void piece_of(Slot slot, ulong piece, thread long &first, thread long &length) {
    const ulong before = pieces_before<PIECE>(slot);
    if (piece < before) {
        first = slot.lo + long(piece) * PIECE;
        length = metal::min(PIECE, slot.stop - long(piece) * PIECE);
    } else {
        const long later = long(piece - before) * PIECE;
        first = slot.lo + slot.stop + later;
        length = metal::min(PIECE, slot.hi - slot.lo - slot.stop - later);
    }
}

// Publishes value head `head`'s share of the slot's successor window (the
// C - 1 raw rows before the publication row): its value channels and the q/k
// channels of the key heads congruent to it. Thread `thread_index` of
// `threads` copies an even share.
inline void publish_window(device const Storage *projection, device Storage *window, Slot slot, ulong head,
    uint thread_index, uint threads, constant ulong *seismic_words) {
    const ulong width = SEISMIC_DIM_W;
    const ulong key_heads = SEISMIC_DIM_NK;
    const ulong value_heads = SEISMIC_DIM_NV;
    const long taps = long(RECURRENT_TAPS) - 1;
    const ulong owned = key_heads > head ? (key_heads - head - 1) / value_heads + 1 : 0;
    const ulong per_tap = width + 2 * width * owned;
    for (ulong item = thread_index; item < ulong(taps) * per_tap; item += threads) {
        const long tap = long(item / per_tap);
        const ulong offset = item % per_tap;
        ulong channel;
        if (offset < width) {
            channel = (2 * key_heads + head) * width + offset;
        } else {
            const ulong key_offset = offset - width;
            const ulong owner = head + (key_offset / (2 * width)) * value_heads;
            const ulong within = key_offset % (2 * width);
            channel = within < width ? owner * width + within : (key_heads + owner) * width + within - width;
        }
        const long position = slot.stop + tap - taps;
        window[slot.target * SEISMIC_WINDOW_STRIDE_0 + ulong(tap) * SEISMIC_WINDOW_STRIDE_1
            + channel * SEISMIC_WINDOW_STRIDE_2] = position < 0
            ? window[slot.source * SEISMIC_WINDOW_STRIDE_0 + ulong(slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1
                + channel * SEISMIC_WINDOW_STRIDE_2]
            : projection[ulong(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0
                + channel * SEISMIC_PROJECTION_STRIDE_1];
    }
}

} // namespace recurrent
