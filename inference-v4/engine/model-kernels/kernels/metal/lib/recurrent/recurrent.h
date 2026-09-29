// Shared pieces of the Metal gated-delta entries (`gated_delta_step`,
// `gated_delta_chunk`; contracts in recurrent.seismic): slot, version and
// tape lookup, the convolution taps and the causal convolution with SiLU,
// the gates, piece splitting, the successor window publication, and the
// row-sequential advance both entries run for slots of at most
// RECURRENT_SEQUENTIAL_ROWS rows and for the rows after a stop row. Channels
// of the projection and window rows and tape rows are contiguous (unit
// stride).

#include "../core/activation.h"

namespace recurrent {

typedef element::Act::storage Storage;

#define RECURRENT_TAPS SEISMIC_DIM_C
#define RECURRENT_UNROLL _Pragma("clang loop unroll(full)")

// A tape row: the innovations u [NV, W], the normalized keys k [NK, W], the
// decays d [NV].
#define RECURRENT_TAPE_U 0
#define RECURRENT_TAPE_K (SEISMIC_DIM_NV * SEISMIC_DIM_W)
#define RECURRENT_TAPE_D ((SEISMIC_DIM_NV + SEISMIC_DIM_NK) * SEISMIC_DIM_W)

// One slot's rows [lo, hi), its publication row count, the version it reads
// (bank `source` advanced by its first `taped` tape rows) and its successor.
struct Slot {
    long lo;
    long hi;
    long stop;
    ulong source;
    long taped;
    ulong target;
};

inline Slot slot_of(device const int *segments, device const int *stop, device const int *previous_bank,
    device const int *previous_tape, device const int *following_bank, ulong slot, constant ulong *seismic_words) {
    Slot result;
    result.lo = segments[slot * SEISMIC_SEGMENTS_STRIDE_0];
    result.hi = segments[slot * SEISMIC_SEGMENTS_STRIDE_0 + SEISMIC_SEGMENTS_STRIDE_1];
    result.stop = stop[slot * SEISMIC_STOP_STRIDE_0];
    result.source = ulong(previous_bank[slot * SEISMIC_PREVIOUS_BANK_STRIDE_0]);
    result.taped = previous_tape[slot * SEISMIC_PREVIOUS_TAPE_STRIDE_0];
    result.target = ulong(following_bank[slot * SEISMIC_FOLLOWING_BANK_STRIDE_0]);
    return result;
}

// Rows the slot records in its successor's tape: those after the stop row, at
// most T.
inline long tape_rows(Slot slot, constant ulong *seismic_words) {
    return metal::min(long(SEISMIC_DIM_T), slot.hi - slot.lo - slot.stop);
}

// Tape row `entry` of `bank`.
inline device float *tape_row(device float *tape, ulong bank, long entry, constant ulong *seismic_words) {
    return tape + bank * SEISMIC_TAPE_STRIDE_0 + ulong(entry) * SEISMIC_TAPE_STRIDE_1;
}

// The one value head of each key head that records the key in a tape row.
inline bool records_key(ulong head, constant ulong *seismic_words) {
    return SEISMIC_PARAM_GROUPED != 0 ? head % (SEISMIC_DIM_NV / SEISMIC_DIM_NK) == 0 : head < SEISMIC_DIM_NK;
}

// The key head of value head `head`.
inline ulong key_head(ulong head, constant ulong *seismic_words) {
    return SEISMIC_PARAM_GROUPED != 0 ? head * SEISMIC_DIM_NK / SEISMIC_DIM_NV : head % SEISMIC_DIM_NK;
}

// The raw input row at slot-local `position`: the source version's window rows
// before the slot, the projection after.
inline device const Storage *raw_row(device const Storage *projection, device const Storage *window, Slot slot,
    long position, constant ulong *seismic_words) {
    return position < 0
        ? window + slot.source * SEISMIC_WINDOW_STRIDE_0
            + ulong(slot.taped + long(RECURRENT_TAPS) - 1 + position) * SEISMIC_WINDOW_STRIDE_1
        : projection + ulong(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0;
}

// The convolution input rows of slot-local row `local` for taps 0..C.
inline void taps(device const Storage *projection, device const Storage *window, Slot slot, long local,
    thread device const Storage *(&rows)[RECURRENT_TAPS], constant ulong *seismic_words) {
    RECURRENT_UNROLL for (uint tap = 0; tap < RECURRENT_TAPS; ++tap) {
        rows[tap] = raw_row(projection, window, slot, local + long(tap) - long(RECURRENT_TAPS - 1), seismic_words);
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

// The chunked rows of a slot, the ones before its publication row lo + stop,
// split into pieces of at most PIECE rows:
template <long PIECE>
inline ulong pieces_before(Slot slot) {
    return ulong(slot.stop + PIECE - 1) / PIECE;
}

// Rows [first, first + length) of the slot's `piece`-th piece (< pieces_before).
template <long PIECE>
inline void piece_of(Slot slot, ulong piece, thread long &first, thread long &length) {
    first = slot.lo + long(piece) * PIECE;
    length = metal::min(PIECE, slot.stop - long(piece) * PIECE);
}

// Publishes value head `head`'s share of the slot's successor window (the
// C - 1 raw rows before the publication row, then the raw rows of its tape):
// its value channels and the q/k channels of the key heads congruent to it.
// Thread `thread_index` of `threads` copies an even share.
inline void publish_window(device const Storage *projection, device Storage *window, Slot slot, ulong head,
    uint thread_index, uint threads, constant ulong *seismic_words) {
    const ulong width = SEISMIC_DIM_W;
    const ulong key_heads = SEISMIC_DIM_NK;
    const ulong value_heads = SEISMIC_DIM_NV;
    const long taps = long(RECURRENT_TAPS) - 1;
    const long rows = taps + tape_rows(slot, seismic_words);
    const ulong owned = key_heads > head ? (key_heads - head - 1) / value_heads + 1 : 0;
    const ulong per_tap = width + 2 * width * owned;
    for (ulong item = thread_index; item < ulong(rows) * per_tap; item += threads) {
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
            ? window[slot.source * SEISMIC_WINDOW_STRIDE_0 + ulong(slot.taped + slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1
                + channel * SEISMIC_WINDOW_STRIDE_2]
            : projection[ulong(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0
                + channel * SEISMIC_PROJECTION_STRIDE_1];
    }
}

// Slots of at most this many rows advance row-sequentially in either entry,
// so a request's recurrent bits never depend on its row class or its peers
// (an MTP verify slot gets the bits of one-row decode).
#define RECURRENT_SEQUENTIAL_ROWS 16

// The row-sequential gated delta rule over the slot's rows [begin, hi), the
// arithmetic of `gated_delta_step` (a threadgroup's shape never changes
// bits). Every thread of the threadgroup calls it. The threadgroup owns state
// rows [block_row0, block_row0 + BLOCK_ROWS) of value head `head`; simdgroup
// `simdgroup` owns LANE_ROWS of them from `row0`, W / 32 contiguous key columns
// per lane. From `begin` = lo it reads them from the slot's source version
// (the bank's state advanced by its tape rows with the step's update) and
// publishes them after the slot's first `stop` rows; from `begin` = lo + stop
// it reads the state already published there. Rows after the stop row are
// recorded in the successor's tape. For each span of up to SPAN rows the
// threadgroup computes the prologue into threadgroup memory (q and k rows of
// QK_STRIDE floats, v rows of V_STRIDE floats, the gates): a simdgroup
// convolves (causal convolution over the window, SiLU) and L2-normalizes a
// whole q or k row with one simd_sum, threads convolve the value channels of
// its state rows; after one barrier the rows advance in order.
template <uint LANE_ROWS, uint BLOCK_ROWS, uint SPAN, uint QK_STRIDE, uint V_STRIDE>
inline void advance_rows(device const Storage *projection, device const float *convolution,
    device const float *rate, device const float *time_bias, device const Storage *window,
    device float *delta, device float *tape, device Storage *mixed, Slot slot, long begin, ulong head,
    ulong block_row0, ulong row0, threadgroup float *query_block, threadgroup float *key_block,
    threadgroup float *value_block, threadgroup float *beta_block, threadgroup float *decay_block,
    uint thread_index, uint threads, uint simdgroup, uint lane, constant ulong *seismic_words) {
    constexpr uint COLUMNS = SEISMIC_DIM_W / 32;
    const uint simdgroups = threads / 32;
    const ulong width = SEISMIC_DIM_W;
    const ulong key_heads = SEISMIC_DIM_NK;
    const ulong value_heads = SEISMIC_DIM_NV;
    const ulong key = key_head(head, seismic_words);
    const long lo = slot.lo;
    const long hi = slot.hi;
    const long publish = lo + slot.stop;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_NORM_EPSILON));
    const float query_scale = metal::rsqrt(float(width));
    const float head_rate = rate[head * SEISMIC_RATE_STRIDE_0];
    const float head_bias = time_bias[head * SEISMIC_TIME_BIAS_STRIDE_0];
    const ulong first_column = ulong(lane) * COLUMNS;

    const long recorded = tape_rows(slot, seismic_words);
    // The simdgroup that owns state row 0 records the head's decay, and the
    // key when the head records its key head's key.
    const bool records = row0 == 0;
    const bool records_keys = records && records_key(head, seismic_words);

    // This simdgroup's state rows, loaded first so their traffic overlaps the
    // prologue.
    device const float *initial = delta + (begin == lo ? slot.source : slot.target) * SEISMIC_DELTA_STRIDE_0
        + head * SEISMIC_DELTA_STRIDE_1 + row0 * SEISMIC_DELTA_STRIDE_2 + first_column;
    device float *published = delta + slot.target * SEISMIC_DELTA_STRIDE_0
        + head * SEISMIC_DELTA_STRIDE_1 + row0 * SEISMIC_DELTA_STRIDE_2 + first_column;
    float state[LANE_ROWS][COLUMNS];
    RECURRENT_UNROLL for (uint r = 0; r < LANE_ROWS; ++r) {
        RECURRENT_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
            state[r][j] = initial[r * SEISMIC_DELTA_STRIDE_2 + j];
        }
    }
    for (long entry = 0; begin == lo && entry < slot.taped; ++entry) {
        device const float *row = tape_row(tape, slot.source, entry, seismic_words);
        const float factor = row[RECURRENT_TAPE_D + head];
        RECURRENT_UNROLL for (uint r = 0; r < LANE_ROWS; ++r) {
            const float innovation = row[RECURRENT_TAPE_U + head * width + row0 + r];
            RECURRENT_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                state[r][j] *= factor;
                state[r][j] = metal::fma(innovation, row[RECURRENT_TAPE_K + key * width + first_column + j],
                    state[r][j]);
            }
        }
    }
    if (begin == lo && publish == lo) {
        RECURRENT_UNROLL for (uint r = 0; r < LANE_ROWS; ++r) {
            RECURRENT_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                published[r * SEISMIC_DELTA_STRIDE_2 + j] = state[r][j];
            }
        }
    }
    const ulong alpha_column = (2 * key_heads + 2 * value_heads) * width + head;
    const ulong value_channel = (2 * key_heads + head) * width + block_row0;
    for (long first = begin; first < hi; first += SPAN) {
        const ulong rows = ulong(metal::min(long(SPAN), hi - first));
        // A simdgroup convolves and L2-normalizes a whole q or k row.
        for (ulong task = simdgroup; task < rows * 2; task += simdgroups) {
            const ulong i = task / 2;
            const bool is_key = task % 2 != 0;
            const long row = first + long(i);
            device const Storage *row_taps[RECURRENT_TAPS];
            taps(projection, window, slot, row - lo, row_taps, seismic_words);
            const ulong channel0 = (is_key ? key_heads + key : key) * width + first_column;
            float values[COLUMNS];
            float squares = 0.0f;
            RECURRENT_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                values[j] = convolve(convolution, row_taps, channel0 + j, seismic_words);
                squares = metal::fma(values[j], values[j], squares);
            }
            const float inverse = metal::rsqrt(simd_sum(squares) + epsilon)
                * (is_key ? 1.0f : query_scale);
            threadgroup float *destination = (is_key ? key_block : query_block) + i * QK_STRIDE
                + first_column;
            RECURRENT_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                destination[j] = values[j] * inverse;
            }
        }
        // The value channels go to the last threads, off the q/k simdgroups.
        for (ulong item = threads - 1 - thread_index; item < rows * BLOCK_ROWS; item += threads) {
            const ulong i = item / BLOCK_ROWS;
            const long row = first + long(i);
            device const Storage *row_taps[RECURRENT_TAPS];
            taps(projection, window, slot, row - lo, row_taps, seismic_words);
            value_block[i * V_STRIDE + item % BLOCK_ROWS] = convolve(convolution, row_taps,
                value_channel + item % BLOCK_ROWS, seismic_words);
        }
        for (ulong i = thread_index; i < rows; i += threads) {
            device const Storage *item = projection + (ulong(first) + i) * SEISMIC_PROJECTION_STRIDE_0;
            const Gates row_gates = gates(element::Act::load(item[alpha_column]),
                element::Act::load(item[alpha_column + value_heads]), head_rate, head_bias);
            beta_block[i] = row_gates.beta;
            decay_block[i] = metal::exp(row_gates.log_decay);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ulong i = 0; i < rows; ++i) {
            const long row = first + long(i);
            const float factor = decay_block[i];
            const float beta = beta_block[i];
            float query[COLUMNS];
            float key_values[COLUMNS];
            float remembered[LANE_ROWS];
            RECURRENT_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                query[j] = query_block[i * QK_STRIDE + first_column + j];
                key_values[j] = key_block[i * QK_STRIDE + first_column + j];
            }
            RECURRENT_UNROLL for (uint r = 0; r < LANE_ROWS; ++r) {
                remembered[r] = 0.0f;
                RECURRENT_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                    state[r][j] *= factor;
                    remembered[r] = metal::fma(state[r][j], key_values[j], remembered[r]);
                }
            }
            const ulong local_row = row0 - block_row0;
            // Rows after the stop row are recorded in the successor's tape.
            device float *entry = row >= publish && row - publish < recorded
                ? tape_row(tape, slot.target, row - publish, seismic_words) : nullptr;
            if (entry != nullptr && records_keys) {
                RECURRENT_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                    entry[RECURRENT_TAPE_K + key * width + first_column + j] = key_values[j];
                }
            }
            if (entry != nullptr && records && lane == 0) {
                entry[RECURRENT_TAPE_D + head] = factor;
            }
            float output[LANE_ROWS];
            RECURRENT_UNROLL for (uint r = 0; r < LANE_ROWS; ++r) {
                const float residual = (value_block[i * V_STRIDE + local_row + r]
                    - simd_sum(remembered[r])) * beta;
                if (entry != nullptr && lane == r) {
                    entry[RECURRENT_TAPE_U + head * width + row0 + r] = residual;
                }
                output[r] = 0.0f;
                RECURRENT_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                    state[r][j] = metal::fma(residual, key_values[j], state[r][j]);
                    output[r] = metal::fma(state[r][j], query[j], output[r]);
                }
                output[r] = simd_sum(output[r]);
            }
            if (lane < LANE_ROWS) {
                float mine = output[0];
                RECURRENT_UNROLL for (uint r = 1; r < LANE_ROWS; ++r) {
                    mine = lane == r ? output[r] : mine;
                }
                mixed[ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1
                    + (row0 + lane) * SEISMIC_RESULT_0_STRIDE_2] = element::Act::store(mine);
            }
            if (row + 1 == publish) {
                RECURRENT_UNROLL for (uint r = 0; r < LANE_ROWS; ++r) {
                    RECURRENT_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                        published[r * SEISMIC_DELTA_STRIDE_2 + j] = state[r][j];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

} // namespace recurrent
