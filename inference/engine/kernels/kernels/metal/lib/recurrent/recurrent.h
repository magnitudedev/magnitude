// Shared pieces of the Metal delta-rule entries (`gated_delta_step`,
// `gated_delta_chunk`, `gated_delta_step_convolved`, and through `kimi.h`
// `kimi_delta_step` and `kimi_delta_chunk`; contracts in recurrent.seismic):
// slot, version and tape lookup, Gated DeltaNet's gates, piece splitting, the
// row-sequential advance the entries run for slots of at most
// RECURRENT_SEQUENTIAL_ROWS rows and for the rows after a stop row, and the
// gating of the raw outputs into the entries' result. The advance and the
// gating take the rule's gates (`HeadGates`, Kimi's `kimi::ChannelGates`) and
// the output gate's input (`GateInput`) as arguments. The convolution and the
// window publication, which only the entries binding the window use, are in
// `convolution.h`. Channels of the projection and convolved rows and tape
// rows are contiguous (unit stride).

#include "../core/activation.h"
#include "../core/arrive.h"
#include "../core/functions.h"
#include <seismic/slab.h>

namespace recurrent {

typedef element::Act::storage Storage;

#define RECURRENT_UNROLL _Pragma("clang loop unroll(full)")

// A tape row: the innovations u [NV, W], the normalized keys k [NK, W], the
// decay factors d [NV, DECAYS] (`HeadGates`).
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
template <typename T>
inline device T *bank(device const ulong *table, ulong index, ulong stride,
    constant ulong *seismic_words) {
    const ulong slab_banks = ulong(SEISMIC_PARAM_SLAB_BANKS);
    return slab::region<T>(table, index / slab_banks) + (index % slab_banks) * stride;
}

inline device float *tape_row(device const ulong *tape, ulong bank_index, long entry,
    constant ulong *seismic_words) {
    return bank<float>(tape, bank_index, SEISMIC_TAPE_STRIDE_0, seismic_words)
        + ulong(entry) * SEISMIC_TAPE_STRIDE_1;
}

// The offset in a tape row of value head `head`'s decay factor of key column
// `column`, DECAYS factors per value head (`HeadGates`).
template <uint DECAYS>
inline ulong tape_decay(ulong head, ulong column, constant ulong *seismic_words) {
    static_assert(DECAYS == 1 || DECAYS == SEISMIC_DIM_W, "one decay per value head or per key column");
    return RECURRENT_TAPE_D + head * DECAYS + (DECAYS == 1 ? 0 : column);
}

// The one value head of each key head that records the key in a tape row.
inline bool records_key(ulong head, constant ulong *seismic_words) {
    return SEISMIC_PARAM_GROUPED != 0 ? head % (SEISMIC_DIM_NV / SEISMIC_DIM_NK) == 0 : head < SEISMIC_DIM_NK;
}

// The key head of value head `head`.
inline ulong key_head(ulong head, constant ulong *seismic_words) {
    return SEISMIC_PARAM_GROUPED != 0 ? head * SEISMIC_DIM_NK / SEISMIC_DIM_NV : head % SEISMIC_DIM_NK;
}

// A row's q | k | v channels after the convolution and SiLU, the prologue
// input of `advance_rows`: `Convolved` loads them (rows `stride` apart) as
// `gated_delta_project_convolved` published them; `Convolving`
// (`convolution.h`) forms the same F32 values from the projection and the
// source window.
struct Convolved {
    device const float *convolved;
    ulong stride;
    struct Row {
        device const float *values;
        float at(ulong channel, constant ulong *) const { return values[channel]; }
    };
    Row row(long row, constant ulong *) const {
        Row result;
        result.values = convolved + ulong(row) * stride;
        return result;
    }
};

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

// The gates of one value head that `advance_rows` reads: `at(row, channel,
// beta)`, the log of the head's decay factor of key channel `channel`
// (< DECAYS) at `row`, setting `beta` to the row's beta for channel 0.
// DECAYS is 1 (the head's one decay scales every key column; `channel` is 0)
// or W (one per key column); a tape row records DECAYS factors per value
// head. Gated DeltaNet's are columns of the projection row (`gates`),
// computed once per row.
struct HeadGates {
    enum : uint { DECAYS = 1 };
    device const Storage *projection;
    ulong alpha_column;
    float rate;
    float time_bias;

    float at(ulong row, ulong, thread float &beta, constant ulong *seismic_words) const {
        device const Storage *item = projection + row * SEISMIC_PROJECTION_STRIDE_0;
        const Gates row_gates = gates(element::Act::load(item[alpha_column]),
            element::Act::load(item[alpha_column + SEISMIC_DIM_NV]), rate, time_bias);
        beta = row_gates.beta;
        return row_gates.log_decay;
    }
};

// A rule's inputs, which give the gates of each value head (`gates`): Gated
// DeltaNet's (`HeadRule`) or Kimi's (`kimi::ChannelRule`).
struct HeadRule {
    enum : uint { DECAYS = HeadGates::DECAYS };
    device const Storage *projection;
    device const float *rate;
    device const float *time_bias;

    HeadGates gates(ulong head, constant ulong *seismic_words) const {
        return HeadGates{projection, (2 * SEISMIC_DIM_NK + 2 * SEISMIC_DIM_NV) * SEISMIC_DIM_W + head,
            rate[head * SEISMIC_RATE_STRIDE_0], time_bias[head * SEISMIC_TIME_BIAS_STRIDE_0]};
    }
};

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

// Element (row, value head, state row) of the raw outputs `mixed`, an
// [M, NV, W] scratch in A.
inline ulong raw_index(ulong row, ulong head, ulong state_row, constant ulong *seismic_words) {
    return (row * SEISMIC_DIM_NV + head) * SEISMIC_DIM_W + state_row;
}

// Slots of at most this many rows advance row-sequentially in either entry,
// so a request's recurrent bits never depend on its row class or its peers
// (an MTP verify slot gets the bits of one-row decode).
#define RECURRENT_SEQUENTIAL_ROWS SEISMIC_CONST_SEQUENTIAL_ROWS

// The row-sequential delta rule over the slot's rows [begin, hi) with value
// head `head`'s gates `gates` (`HeadGates`): S <- S diag(d) + beta (v -
// S diag(d) k) k^T, output S q, the arithmetic of the step entries (a
// threadgroup's shape never changes bits), storing the raw outputs to
// `mixed` (`raw_index`). Every thread of the threadgroup calls it. The
// threadgroup owns state rows [block_row0, block_row0 + BLOCK_ROWS) of value
// head `head`; simdgroup `simdgroup` owns LANE_ROWS of them from `row0`,
// W / 32 contiguous key columns per lane. From `begin` = lo it reads them
// from the slot's source version (the bank's state advanced by its tape rows
// with the step's update) and publishes them after the slot's first `stop`
// rows; from `begin` = lo + stop it reads the state already published there.
// Rows after the stop row are recorded in the successor's tape. For each
// span of up to SPAN rows the threadgroup computes the prologue into
// threadgroup memory (q and k rows of QK_STRIDE floats, v rows of V_STRIDE
// floats, each row's beta in `beta_block` and its DECAYS decay factors in
// `decay_block`): a simdgroup forms a whole q or k row's convolved channels
// (`Inputs`) and L2-normalizes it with one simd_sum, threads form the value
// channels of its state rows; after one barrier the rows advance in order.
template <uint LANE_ROWS, uint BLOCK_ROWS, uint SPAN, uint QK_STRIDE, uint V_STRIDE, typename Inputs,
    typename RowGates, typename Mixed>
inline void advance_rows(Inputs inputs, RowGates gates,
    device const ulong *delta, device const ulong *tape, Mixed mixed, Slot slot, long begin, ulong head,
    ulong block_row0, ulong row0, threadgroup float *query_block, threadgroup float *key_block,
    threadgroup float *value_block, threadgroup float *beta_block, threadgroup float *decay_block,
    uint thread_index, uint threads, uint simdgroup, uint lane, constant ulong *seismic_words) {
    constexpr uint COLUMNS = SEISMIC_DIM_W / 32;
    constexpr uint DECAYS = RowGates::DECAYS;
    // A lane's distinct decay factors.
    constexpr uint LANE_DECAYS = DECAYS == 1 ? 1 : COLUMNS;
    const uint simdgroups = threads / 32;
    const ulong width = SEISMIC_DIM_W;
    const ulong key_heads = SEISMIC_DIM_NK;
    const ulong key = key_head(head, seismic_words);
    const long lo = slot.lo;
    const long hi = slot.hi;
    const long publish = lo + slot.stop;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_NORM_EPSILON));
    const float query_scale = metal::rsqrt(float(width));
    const ulong first_column = ulong(lane) * COLUMNS;
    // The key column of the lane's first decay factor.
    const ulong first_decay = DECAYS == 1 ? 0 : first_column;
    const ulong tape_decays = tape_decay<DECAYS>(head, first_column, seismic_words);

    const long recorded = tape_rows(slot, seismic_words);
    // The simdgroup that owns state row 0 records the head's decay factors,
    // and the key when the head records its key head's key.
    const bool records = row0 == 0;
    const bool records_keys = records && records_key(head, seismic_words);

    // This simdgroup's state rows, loaded first so their traffic overlaps the
    // prologue.
    device const float *initial = bank<float>(delta, begin == lo ? slot.source : slot.target,
        SEISMIC_DELTA_STRIDE_0, seismic_words)
        + head * SEISMIC_DELTA_STRIDE_1 + row0 * SEISMIC_DELTA_STRIDE_2 + first_column;
    device float *published = bank<float>(delta, slot.target, SEISMIC_DELTA_STRIDE_0, seismic_words)
        + head * SEISMIC_DELTA_STRIDE_1 + row0 * SEISMIC_DELTA_STRIDE_2 + first_column;
    float state[LANE_ROWS][COLUMNS];
    RECURRENT_UNROLL for (uint r = 0; r < LANE_ROWS; ++r) {
        RECURRENT_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
            state[r][j] = initial[r * SEISMIC_DELTA_STRIDE_2 + j];
        }
    }
    for (long entry = 0; begin == lo && entry < slot.taped; ++entry) {
        device const float *row = tape_row(tape, slot.source, entry, seismic_words);
        // One decay per value head is loaded once.
        float factor[LANE_DECAYS];
        RECURRENT_UNROLL for (uint j = 0; j < LANE_DECAYS; ++j) {
            factor[j] = row[tape_decays + j];
        }
        RECURRENT_UNROLL for (uint r = 0; r < LANE_ROWS; ++r) {
            const float innovation = row[RECURRENT_TAPE_U + head * width + row0 + r];
            RECURRENT_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                state[r][j] *= factor[j % LANE_DECAYS];
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
    const ulong value_channel = (2 * key_heads + head) * width + block_row0;
    for (long first = begin; first < hi; first += SPAN) {
        const ulong rows = ulong(metal::min(long(SPAN), hi - first));
        // A simdgroup forms and L2-normalizes a whole q or k row.
        for (ulong task = simdgroup; task < rows * 2; task += simdgroups) {
            const ulong i = task / 2;
            const bool is_key = task % 2 != 0;
            const long row = first + long(i);
            const typename Inputs::Row channels = inputs.row(row, seismic_words);
            const ulong channel0 = (is_key ? key_heads + key : key) * width + first_column;
            float values[COLUMNS];
            float squares = 0.0f;
            RECURRENT_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                values[j] = channels.at(channel0 + j, seismic_words);
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
            value_block[i * V_STRIDE + item % BLOCK_ROWS] = inputs.row(row, seismic_words).at(
                value_channel + item % BLOCK_ROWS, seismic_words);
        }
        for (ulong item = thread_index; item < rows * DECAYS; item += threads) {
            const ulong i = item / DECAYS, channel = item % DECAYS;
            float beta;
            decay_block[item] = metal::exp(gates.at(ulong(first) + i, channel, beta, seismic_words));
            if (channel == 0) {
                beta_block[i] = beta;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ulong i = 0; i < rows; ++i) {
            const long row = first + long(i);
            const float beta = beta_block[i];
            float factor[LANE_DECAYS];
            float query[COLUMNS];
            float key_values[COLUMNS];
            float remembered[LANE_ROWS];
            RECURRENT_UNROLL for (uint j = 0; j < LANE_DECAYS; ++j) {
                factor[j] = decay_block[i * DECAYS + first_decay + j];
            }
            RECURRENT_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                query[j] = query_block[i * QK_STRIDE + first_column + j];
                key_values[j] = key_block[i * QK_STRIDE + first_column + j];
            }
            RECURRENT_UNROLL for (uint r = 0; r < LANE_ROWS; ++r) {
                remembered[r] = 0.0f;
                RECURRENT_UNROLL for (uint j = 0; j < COLUMNS; ++j) {
                    state[r][j] *= factor[j % LANE_DECAYS];
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
            if (entry != nullptr && records && (DECAYS != 1 || lane == 0)) {
                RECURRENT_UNROLL for (uint j = 0; j < LANE_DECAYS; ++j) {
                    entry[tape_decays + j] = factor[j];
                }
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
                mixed[raw_index(ulong(row), head, row0 + lane, seismic_words)] = element::Act::store(mine);
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

// ---------------------------------------------------------------------------
// Gating: the result's value (row, head, i) is
//   round_A(round_A(raw * inverse * norm[i]) * round_A(gate(z)))
// with gate the output gate of `gate_function` (`functions::output_gate`)
// with raw the stored raw output, z column head * W + i of the row's
// `GateInput` and inverse = rsqrt(sum_head raw^2 / W + epsilon). The
// entries differ only in how a head's square sum is reduced.

typedef ELEMENT_OF(SEISMIC_RECURRENT_NORM) Norm;

// The output gate's input z: row `row`'s [NV, W] values from
// `first + row * stride`.
struct GateInput {
    device const Storage *first;
    ulong stride;
};

// Gated DeltaNet's z: the projection row's columns from CH.
inline GateInput projected_gate(device const Storage *projection, constant ulong *seismic_words) {
    return GateInput{projection + (2 * SEISMIC_DIM_NK + SEISMIC_DIM_NV) * SEISMIC_DIM_W, SEISMIC_PROJECTION_STRIDE_0};
}

// The inputs of columns i..i+7 of (row, head) as (even, odd): raw outputs,
// z and the norm weights.
struct Gated8 {
    float4 raw_even, raw_odd, z_even, z_odd, norm_even, norm_odd;
};

template <typename Mixed>
inline Gated8 gated_inputs8(Mixed mixed, GateInput input,
    device const uchar *norm, ulong row, ulong head, uint i, constant ulong *seismic_words) {
    const auto raw = mixed + raw_index(row, head, i, seismic_words);
    device const Storage *z = input.first + row * input.stride + head * SEISMIC_DIM_W + i;
    Gated8 v;
    RECURRENT_UNROLL for (uint j = 0; j < 4; ++j) {
        v.raw_even[j] = element::Act::load(raw[2 * j]);
        v.raw_odd[j] = element::Act::load(raw[2 * j + 1]);
        v.z_even[j] = element::Act::load(z[2 * j]);
        v.z_odd[j] = element::Act::load(z[2 * j + 1]);
        v.norm_even[j] = element::at<Norm>(norm, ulong(i + 2 * j) * SEISMIC_RECURRENT_NORM_STRIDE_0);
        v.norm_odd[j] = element::at<Norm>(norm, ulong(i + 2 * j + 1) * SEISMIC_RECURRENT_NORM_STRIDE_0);
    }
    return v;
}

// Stores the gated columns i..i+7 of (row, head) given the head's inverse.
inline void store_gated8(thread const Gated8 &v, float inverse, device Storage *gated, ulong row, ulong head,
    uint i, constant ulong *seismic_words) {
    const float4 e = v.raw_even * inverse * v.norm_even, o = v.raw_odd * inverse * v.norm_odd;
    // `functions::output_gate` over four lanes.
    const bool sigmoid = int(uint(SEISMIC_PARAM_GATE_FUNCTION)) == functions::gate_sigmoid;
    const float4 ae = (sigmoid ? 1.0f : v.z_even) / (1.0f + metal::exp(-v.z_even));
    const float4 ao = (sigmoid ? 1.0f : v.z_odd) / (1.0f + metal::exp(-v.z_odd));
    device Storage *out = gated + row * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1;
    RECURRENT_UNROLL for (uint j = 0; j < 4; ++j) {
        out[(i + 2 * j) * SEISMIC_RESULT_0_STRIDE_2] = element::Act::store(
            element::Act::round(element::Act::round(e[j]) * element::Act::round(ae[j])));
        out[(i + 2 * j + 1) * SEISMIC_RESULT_0_STRIDE_2] = element::Act::store(
            element::Act::round(element::Act::round(o[j]) * element::Act::round(ao[j])));
    }
}

// Gates every row of value head `head` (the step): W / 8 adjacent lanes per
// row, lane j owning columns 8j..8j+7, each sum the squares of their eight
// raw values in column order, and a butterfly over the lanes gives each the
// head's square sum. This is the lane grouping of the projection GEMV
// staging. All threads of the threadgroup (whole simdgroups) call it.
inline void gate_head(arrive::coherent_device<const Storage> mixed, GateInput input, device const uchar *norm,
    device Storage *gated, ulong head, uint thread_index, uint threads, constant ulong *seismic_words) {
    constexpr uint LANES = SEISMIC_DIM_W / 8;
    static_assert(SEISMIC_DIM_W % 8 == 0 && LANES <= 32 && (LANES & (LANES - 1)) == 0,
        "a head is gated over W / 8 lanes, a power of two up to 32");
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const uint items = uint(SEISMIC_DIM_M) * LANES;
    for (uint first = 0; first < items; first += threads) {
        const uint item = first + thread_index;
        // Lanes past the rows repeat the last row, so every lane of a
        // butterfly group takes part; they store nothing.
        const ulong row = metal::min(ulong(item / LANES), ulong(SEISMIC_DIM_M) - 1);
        const uint i = 8 * (item % LANES);
        const Gated8 v = gated_inputs8(mixed, input, norm, row, head, i, seismic_words);
        float squares = 0.0f;
        RECURRENT_UNROLL for (uint j = 0; j < 4; ++j) {
            squares = metal::fma(v.raw_even[j], v.raw_even[j], squares);
            squares = metal::fma(v.raw_odd[j], v.raw_odd[j], squares);
        }
        for (ushort offset = 1; offset < LANES; offset <<= 1)
            squares += simd_shuffle_xor(squares, offset);
        if (item < items)
            store_gated8(v, metal::rsqrt(squares / float(SEISMIC_DIM_W) + epsilon), gated, row, head, i,
                seismic_words);
    }
}

// Gates (row, head) with one simdgroup (the chunk): lane l sums the squares
// of columns l, l + 32, ... in order, then one simd_sum. This is the
// projection family's normalizing pre-pass order.
inline void gate_row(device const Storage *mixed, GateInput input, device const uchar *norm,
    device Storage *gated, ulong row, ulong head, uint lane, constant ulong *seismic_words) {
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    device const Storage *raw = mixed + raw_index(row, head, 0, seismic_words);
    float squares = 0.0f;
    for (uint i = lane; i < SEISMIC_DIM_W; i += 32) {
        const float value = element::Act::load(raw[i]);
        squares = metal::fma(value, value, squares);
    }
    const float inverse = metal::rsqrt(simd_sum(squares) / float(SEISMIC_DIM_W) + epsilon);
    for (uint i = 8 * lane; i < SEISMIC_DIM_W; i += 8 * 32)
        store_gated8(gated_inputs8(mixed, input, norm, row, head, i, seismic_words), inverse, gated, row,
            head, i, seismic_words);
}

// A step entry's threadgroup (value head group.x, block group.y of ROWS
// state rows, slot `slot` = group.z) after the entry stages its blocks of
// SPAN rows: it advances the slot's rows under `rule` (`advance_rows`),
// zeroes the raw rows after the last slot, and the last of the head's
// threadgroups to arrive (`arrivals[head]`, over every row block and slot)
// gates the head's rows into the result (`gate_head`).
template <uint LANE_ROWS, uint SPAN, typename Inputs, typename Rule>
inline void step(Inputs inputs, Rule rule, GateInput gate, device const uchar *norm, device Storage *gated,
    arrive::coherent_device<Storage> mixed, device atomic_uint *arrivals, device const ulong *delta,
    device const ulong *tape, Slot slot, uint3 group, threadgroup float *query_block, threadgroup float *key_block,
    threadgroup float *value_block, threadgroup float *beta_block, threadgroup float *decay_block,
    threadgroup uint *last, uint thread_index, uint threads, uint simdgroup, uint lane,
    constant ulong *seismic_words) {
    const ulong head = group.x;
    const ulong block_row0 = ulong(group.y) * SEISMIC_TUNE_ROWS;
    const ulong row0 = block_row0 + ulong(simdgroup) * LANE_ROWS;
    advance_rows<LANE_ROWS, SEISMIC_TUNE_ROWS, SPAN, SEISMIC_DIM_W, SEISMIC_TUNE_ROWS>(inputs,
        rule.gates(head, seismic_words), delta, tape, mixed, slot, slot.lo, head, block_row0, row0, query_block,
        key_block, value_block, beta_block, decay_block, thread_index, threads, simdgroup, lane, seismic_words);
    // Rows after the last slot belong to no sequence; their raw output is zero.
    if (group.z + 1 == SEISMIC_DIM_B && lane < LANE_ROWS) {
        for (ulong row = ulong(slot.hi); row < SEISMIC_DIM_M; ++row)
            mixed[raw_index(row, head, row0 + lane, seismic_words)] = element::Act::store(0.0f);
    }
    const uint expected = uint(SEISMIC_DIM_W / SEISMIC_TUNE_ROWS * SEISMIC_DIM_B);
    if (!arrive::last(arrivals + head, expected, last, thread_index))
        return;
    gate_head(mixed, gate, norm, gated, head, thread_index, threads, seismic_words);
}

} // namespace recurrent
