// The Metal chunked delta-rule entries (`gated_delta_chunk`,
// `kimi_delta_chunk`; contracts in recurrent.seismic) over the rule's gates
// (`recurrent::HeadRule`, `kimi::ChannelRule`): the chunked gated delta rule
// (WY form) over pieces of at most 8 rows, in two launches that share the
// `inputs` scratch (f32, the chunked rows only), then a third that gates the
// raw outputs the scan stores in the `mixed` scratch. The chunked rows of a
// slot are those before its stop row, starting from the slot's source
// version (the bank's state advanced by its tape rows); the rows after the
// stop row advance row-sequentially from the published state with the step's
// arithmetic (`recurrent::advance_rows`), recording the tape. A slot of at
// most RECURRENT_SEQUENTIAL_ROWS rows (an MTP verify) is not chunked at all
// and gets the step's bits. The scratch:
//   rows    [NK, M, 2W]   per key head: L2-normalized q (scaled by W^-1/2), k
//   values  [NV, M, W]    convolved v heads
//   gates   [NV, M * D + M]  per value head: log decays [M, D], then beta [M]
//                            (D = DECAYS)
//
// `rch_inputs`: one simdgroup per (row, channel head) convolves the row once,
// W / 4 channel quads over the lanes.
//
// `rch_scan`: a threadgroup owns (ROWS state rows, value head, slot); each
// simdgroup keeps the transposed state of 16 rows, two S^T [W, 8] halves, in
// registers as 8x8 matrices (each K / Q / K^T operand tile it loads serves
// both halves), and the slot's pieces advance in order. Per piece (rows past
// the piece are zero):
//   1. the piece's q/k rows, the threadgroup's value channels and the gates
//      are staged into threadgroup memory;
//   2. the rule's decayed products (`RchDecays`): A and D's undecayed or
//      decayed key-column shares and X = K S^T, Y = Q S^T;
//   3. every simdgroup sums the shares and forms
//        A_ij = beta_i d_ij k_i . k_j   (j < i),   D_ij = d_ij q_i . k_j   (j <= i),
//        T = (I + A)^-1 = (I - A)(I + A^2)(I + A^4)   (A strictly lower, A^8 = 0),
//        U = T diag(beta) (V - diag(gamma) X),  out = diag(gamma) Y + D U,
//        S^T <- diag(f) S^T + K'^T diag(tail) U,
//      with d, gamma, tail, f and K' the rule's (`RchDecays`).
//
// The gate launch: one simdgroup per (row, value head) gates the row's raw
// outputs into the result (`recurrent::gate_row`).
//
// Threadgroup row strides are 8 or 24 floats modulo 32 banks, so 8x8 loads
// are conflict-free. Decay products are exponentials of differences of
// cumulative log decays, never ratios. Channels of the projection, gate,
// decay and window rows and the columns of the delta arena's rows must be
// contiguous (unit stride).

#define RCH_PIECE SEISMIC_CONST_METAL_CHUNK_PIECE
#define RCH_ROWS SEISMIC_TUNE_ROWS
#define RCH_THREADS (2 * RCH_ROWS)
#define RCH_SIMDGROUPS (RCH_ROWS / 16)
#define RCH_WIDTH SEISMIC_DIM_W
#define RCH_TILES (RCH_WIDTH / 8)
#define RCH_LANE_QUADS ((RCH_WIDTH / 4 + 31) / 32)
#define RCH_KEY_HEADS SEISMIC_DIM_NK
#define RCH_VALUE_HEADS SEISMIC_DIM_NV
#define RCH_ROW_STRIDE (2 * RCH_WIDTH)
// K K^T and Q K^T are summed from 4 shares of the key columns, whatever the
// number of simdgroups, so ROWS never changes their bits.
#define RCH_SHARES 4
#define RCH_PADDED(n) ((n) % 32 == 8 || (n) % 32 == 24 ? (n) : (n) + 8)
#define RCH_INPUT_STRIDE RCH_PADDED(RCH_WIDTH)
#define RCH_VALUE_STRIDE RCH_PADDED(RCH_ROWS)

#include "convolution.h"

// The slot containing `row`, or false when the row is not chunked: outside
// every slot, in a sequential (short) slot, or after its slot's stop row.
inline bool rch_row_slot(device const int *segments, device const int *stop, device const int *previous_bank,
    device const int *previous_tape, device const int *following_bank, ulong row, thread recurrent::Slot &slot,
    constant ulong *seismic_words) {
    for (ulong candidate = 0; candidate < SEISMIC_DIM_B; ++candidate) {
        const long first = segments[candidate * SEISMIC_SEGMENTS_STRIDE_0];
        const long last = segments[candidate * SEISMIC_SEGMENTS_STRIDE_0
            + SEISMIC_SEGMENTS_STRIDE_1];
        if (long(row) >= first && long(row) < last) {
            slot = recurrent::slot_of(segments, stop, previous_bank, previous_tape, following_bank, candidate,
                seismic_words);
            return last - first > RECURRENT_SEQUENTIAL_ROWS && long(row) < first + slot.stop;
        }
    }
    return false;
}

// Rows after the last slot belong to no sequence: this simdgroup's 16 state
// rows of their raw output are zero.
inline void rch_zero_tail(device recurrent::Storage *mixed, ulong slot, long hi, ulong head, ulong row0,
    ushort lane, constant ulong *seismic_words) {
    if (slot + 1 != SEISMIC_DIM_B) {
        return;
    }
    const ulong rows = SEISMIC_DIM_M - ulong(hi);
    for (ulong index = lane; index < rows * 16; index += 32) {
        mixed[recurrent::raw_index(ulong(hi) + index / 16, head, row0 + index % 16, seismic_words)]
            = element::Act::store(0.0f);
    }
}

// The inputs launch's work for channel head `head` (the q heads, the k heads,
// then the v heads) of chunked row `row` of `slot`, by one simdgroup, lane l
// holding channel quads l, l + 32, ...: SiLU of the causal convolution of
// its channels, L2-normalized for q (scaled by W^-1/2) and k, stored to the
// `inputs` scratch's q | k rows [NK, M, 2W] (`qk_rows`) or v rows [NV, M, W]
// (`value_rows`) of `height` rows.
inline void rch_prepare_row(device const recurrent::Storage *projection, device const float *convolution,
    device const ulong *window, recurrent::Slot slot, ulong row, uint head, device float *qk_rows,
    device float *value_rows, ulong height, uint lane, constant ulong *seismic_words) {
    const uint width = RCH_WIDTH;
    const bool is_value = head >= 2 * RCH_KEY_HEADS;
    const bool is_key = head >= RCH_KEY_HEADS && !is_value;
    device float *destination = is_value
        ? value_rows + (ulong(head - 2 * RCH_KEY_HEADS) * height + row) * width
        : qk_rows + (ulong(head % RCH_KEY_HEADS) * height + row) * RCH_ROW_STRIDE + (is_key ? width : 0);

    // The tap rows: the window of the slot's source version before the slot,
    // the projection after.
    recurrent::Taps taps;
    recurrent::taps(projection, window, slot, long(row) - slot.lo, taps, seismic_words);
    // This lane's channel quads.
    float4 values[RCH_LANE_QUADS];
    float squares = 0.0f;
    RECURRENT_UNROLL for (uint j = 0; j < RCH_LANE_QUADS; ++j) {
        const uint channel = 4 * (lane + 32 * j);
        values[j] = channel < width
            ? recurrent::convolve4(convolution, taps, head * width + channel, seismic_words) : float4(0.0f);
        squares += metal::dot(values[j], values[j]);
    }
    if (!is_value) {
        const float epsilon = as_type<float>(uint(SEISMIC_PARAM_NORM_EPSILON));
        const float inverse = metal::rsqrt(simd_sum(squares) + epsilon)
            * (is_key ? 1.0f : metal::rsqrt(float(width)));
        RECURRENT_UNROLL for (uint j = 0; j < RCH_LANE_QUADS; ++j) {
            values[j] *= inverse;
        }
    }
    RECURRENT_UNROLL for (uint j = 0; j < RCH_LANE_QUADS; ++j) {
        const uint channel = 4 * (lane + 32 * j);
        if (channel < width) {
            *reinterpret_cast<device float4 *>(destination + channel) = values[j];
        }
    }
}

// The `inputs` scratch's regions.
template <uint DECAYS>
struct RchScratch {
    device float *rows;
    device float *values;
    device float *gates;
    ulong height;

    // Value head `head`'s log decays [M, DECAYS], then its betas [M].
    device float *head_gates(ulong head) const { return gates + head * (DECAYS + 1) * height; }
};

template <uint DECAYS>
inline RchScratch<DECAYS> rch_scratch(device float *inputs, constant ulong *seismic_words) {
    RchScratch<DECAYS> scratch;
    scratch.height = SEISMIC_DIM_M;
    scratch.rows = inputs;
    scratch.values = scratch.rows + RCH_KEY_HEADS * scratch.height * RCH_ROW_STRIDE;
    scratch.gates = scratch.values + RCH_VALUE_HEADS * scratch.height * RCH_WIDTH;
    return scratch;
}

// The inputs launch: the row's channel head (`rch_prepare_row`) and, for a
// value head, lane l's key channels l, l + 32, ... of its log decays and
// channel 0's beta.
template <typename Rule>
inline void rch_inputs(Rule rule, device const recurrent::Storage *projection, device const float *convolution,
    device const int *segments, device const int *stop, device const int *previous_bank,
    device const int *previous_tape, device const int *following_bank, device const ulong *window,
    device float *inputs, uint3 group, uint simdgroup, uint lane, constant ulong *seismic_words) {
    constexpr uint DECAYS = Rule::DECAYS;
    const uint head = group.y * 4 + simdgroup;
    const ulong row = group.x;
    recurrent::Slot slot;
    if (head >= 2 * RCH_KEY_HEADS + RCH_VALUE_HEADS
        || !rch_row_slot(segments, stop, previous_bank, previous_tape, following_bank, row, slot, seismic_words)) {
        return;
    }
    const RchScratch<DECAYS> scratch = rch_scratch<DECAYS>(inputs, seismic_words);
    rch_prepare_row(projection, convolution, window, slot, row, head, scratch.rows, scratch.values, scratch.height,
        lane, seismic_words);
    const bool is_value = head >= 2 * RCH_KEY_HEADS;
    const uint value_head = head - 2 * RCH_KEY_HEADS;
    RECURRENT_UNROLL for (uint k = 0; k < (DECAYS + 31) / 32; ++k) {
        const uint channel = lane + 32 * k;
        if (is_value && channel < DECAYS) {
            device float *gates = scratch.head_gates(value_head);
            float beta;
            gates[row * DECAYS + channel] = rule.gates(value_head, seismic_words).at(row, channel, beta,
                seismic_words);
            if (channel == 0) {
                gates[DECAYS * scratch.height + row] = beta;
            }
        }
    }
}

// X = K S^T and Y = Q S^T of a simdgroup's two state halves.
inline void rch_project(threadgroup const float *queries, threadgroup const float *keys,
    thread simdgroup_float8x8 (&state)[2][RCH_TILES], thread simdgroup_float8x8 (&removed)[2],
    thread simdgroup_float8x8 (&output)[2]) {
    RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
        removed[h] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        output[h] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
        simdgroup_float8x8 key;
        simdgroup_float8x8 query;
        simdgroup_load(key, keys + 8 * c, RCH_INPUT_STRIDE);
        simdgroup_load(query, queries + 8 * c, RCH_INPUT_STRIDE);
        RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
            simdgroup_multiply_accumulate(removed[h], key, state[h][c], removed[h]);
            simdgroup_multiply_accumulate(output[h], query, state[h][c], output[h]);
        }
    }
}

// The shares of A and D: RCH_SHARES shares of the key columns each.
typedef threadgroup float (*RchShares)[2][RCH_PIECE * RCH_PIECE];

// The rule's decayed products, the only arithmetic the rules' chunked forms
// do not share. Staged per piece: the log decays (row i's at gates[i *
// STRIDE], DECAYS of them), then the betas at gates[RCH_PIECE * STRIDE]. With
// G_i the cumulative log decays from the piece's start:
//
// DECAYS == W (Kimi): the decay enters inside the products, per key channel:
//   A_ij = beta_i sum_c k_ic k_jc exp(G_ic - G_jc), D_ij = sum_c q_ic k_jc
//   exp(G_ic - G_jc) as direct sums, each term's exponential formed in place,
//   X = (K o exp G) S^T, Y = (Q o exp G) S^T, so d = gamma = tail = 1;
//   f_c = exp G_last,c and K' = K o exp(G_last - G).
// Range: every exponent is a difference G_ic - G_jc (j <= i), G_ic or
// G_last,c - G_ic, a sum of log decays of rows of the piece, each <= 0. So no
// exponential exceeds 1 and none overflows, whatever the decays; the
// cumulative sums are of at most 8 terms, each finite. An exponential that
// underflows has a true value below 2^-126 and scales a product of unit-norm
// q and k channels (|q_c|, |k_c| <= 1) or a state entry, so it drops a term
// below 2^-126 |S| absolutely: below every output's rounding at any decay.
// FLA's factored form, (k_i o exp G_i) . (k_j o exp -G_j), would need |G| <=
// 87 for exp(-G) to stay finite, which holds only for a bounded decay (lower
// = -5: 8 * 5 = 40).
template <uint DECAYS>
struct RchDecays {
    enum : uint { STRIDE = RCH_INPUT_STRIDE };
    // K o exp G, then K o exp(G_last - G) (K').
    threadgroup float *decayed;
    threadgroup const float *log_decays;
    float row_beta;

    void stage(threadgroup float *gates, device const float *head_gates, ulong height, long first, long length,
        uint thread_index) {
        if (thread_index < RCH_PIECE) {
            gates[RCH_PIECE * STRIDE + thread_index] = long(thread_index) < length
                ? head_gates[DECAYS * height + ulong(first) + thread_index] : 0.0f;
        }
        for (uint item = thread_index; item < RCH_PIECE * DECAYS; item += RCH_THREADS) {
            const uint i = item / DECAYS;
            gates[i * STRIDE + item % DECAYS] = long(i) < length
                ? head_gates[(ulong(first) + i) * DECAYS + item % DECAYS] : 0.0f;
        }
    }

    void products(threadgroup float *queries, threadgroup const float *keys, threadgroup float *gates,
        RchShares shares, thread simdgroup_float8x8 (&state)[2][RCH_TILES], thread simdgroup_float8x8 (&removed)[2],
        thread simdgroup_float8x8 (&output)[2], uint thread_index, uint, ushort lane, ushort row) {
        log_decays = gates;
        // Each channel's cumulative log decays over the piece.
        for (uint channel = thread_index; channel < DECAYS; channel += RCH_THREADS) {
            float running = 0.0f;
            RECURRENT_UNROLL for (uint i = 0; i < RCH_PIECE; ++i) {
                running += gates[i * STRIDE + channel];
                gates[i * STRIDE + channel] = running;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        constexpr uint SHARE_COLUMNS = DECAYS / RCH_SHARES;
        for (uint task = thread_index; task < RCH_SHARES * 2 * RCH_PIECE * RCH_PIECE; task += RCH_THREADS) {
            const uint share = task % RCH_SHARES;
            const uint pair = (task / RCH_SHARES) % (RCH_PIECE * RCH_PIECE);
            const uint kind = task / (RCH_SHARES * RCH_PIECE * RCH_PIECE);
            const uint i = pair / RCH_PIECE;
            const uint j = pair % RCH_PIECE;
            threadgroup const float *left = (kind == 0 ? keys : queries) + i * RCH_INPUT_STRIDE;
            threadgroup const float *right = keys + j * RCH_INPUT_STRIDE;
            threadgroup const float *upper = gates + i * STRIDE;
            threadgroup const float *lower = gates + j * STRIDE;
            float sum = 0.0f;
            if (j <= i) {
                for (uint c = share * SHARE_COLUMNS; c < (share + 1) * SHARE_COLUMNS; ++c) {
                    sum = metal::fma(left[c] * right[c], metal::exp(upper[c] - lower[c]), sum);
                }
            }
            shares[share][kind][pair] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Q o exp G in place, and K o exp G.
        for (uint item = thread_index; item < RCH_PIECE * DECAYS; item += RCH_THREADS) {
            const uint at = (item / DECAYS) * RCH_INPUT_STRIDE + item % DECAYS;
            const float gamma = metal::exp(gates[(item / DECAYS) * STRIDE + item % DECAYS]);
            queries[at] *= gamma;
            decayed[at] = keys[at] * gamma;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        rch_project(queries, decayed, state, removed, output);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // K o exp(G_last - G) for the state update.
        for (uint item = thread_index; item < RCH_PIECE * DECAYS; item += RCH_THREADS) {
            const uint channel = item % DECAYS;
            const uint at = (item / DECAYS) * RCH_INPUT_STRIDE + channel;
            decayed[at] = keys[at]
                * metal::exp(gates[(RCH_PIECE - 1) * STRIDE + channel] - gates[(item / DECAYS) * STRIDE + channel]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Lane i < 8 holds row i's beta; padding rows have zero beta.
        row_beta = simd_shuffle(lane < RCH_PIECE ? gates[RCH_PIECE * STRIDE + lane] : 0.0f, row);
    }

    float pair(ushort, ushort) const { return 1.0f; }
    float gamma() const { return 1.0f; }
    float tail() const { return 1.0f; }
    threadgroup const float *update_keys(threadgroup const float *) const { return decayed; }
    void start_update() {}
    float factor(uint c, ushort row) const {
        return metal::exp(log_decays[(RCH_PIECE - 1) * STRIDE + 8 * c + row]);
    }
};

// DECAYS == 1 (Gated DeltaNet): one decay per value head, outside the
// products: K K^T and Q K^T from shares of 8x8 tiles, d_ij = exp(G_i - G_j),
// gamma_i = exp G_i, tail_i = exp(G_last - G_i), f = exp G_last and K' = K.
template <>
struct RchDecays<1> {
    enum : uint { STRIDE = 1, SHARE_TILES = RCH_TILES / RCH_SHARES };
    float row_beta;
    float cumulative;
    float row_cumulative;
    float row_gamma;
    float row_tail;
    float last;
    float last_factor;

    void stage(threadgroup float *gates, device const float *head_gates, ulong height, long first, long length,
        uint thread_index) {
        if (thread_index < 2 * RCH_PIECE) {
            const uint i = thread_index % RCH_PIECE;
            gates[thread_index] = long(i) < length
                ? head_gates[(thread_index / RCH_PIECE) * height + ulong(first) + i] : 0.0f;
        }
    }

    void products(threadgroup float *queries, threadgroup const float *keys, threadgroup float *gates,
        RchShares shares, thread simdgroup_float8x8 (&state)[2][RCH_TILES], thread simdgroup_float8x8 (&removed)[2],
        thread simdgroup_float8x8 (&output)[2], uint, uint simdgroup, ushort lane, ushort row) {
        for (uint share = simdgroup; share < RCH_SHARES; share += RCH_SIMDGROUPS) {
            simdgroup_float8x8 keys_share = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 queries_share = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            RECURRENT_UNROLL for (uint s = 0; s < SHARE_TILES; ++s) {
                const uint c = share * SHARE_TILES + s;
                simdgroup_float8x8 key;
                simdgroup_float8x8 query;
                simdgroup_float8x8 transposed;
                simdgroup_load(key, keys + 8 * c, RCH_INPUT_STRIDE);
                simdgroup_load(query, queries + 8 * c, RCH_INPUT_STRIDE);
                simdgroup_load(transposed, keys + 8 * c, RCH_INPUT_STRIDE, ulong2(0), true);
                simdgroup_multiply_accumulate(keys_share, key, transposed, keys_share);
                simdgroup_multiply_accumulate(queries_share, query, transposed, queries_share);
            }
            simdgroup_store(keys_share, shares[share][0], RCH_PIECE);
            simdgroup_store(queries_share, shares[share][1], RCH_PIECE);
        }
        rch_project(queries, keys, state, removed, output);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Lane i < 8 holds row i's cumulative log decay and beta; padding
        // rows have zero decay and beta.
        const float beta = lane < RCH_PIECE ? gates[RCH_PIECE + lane] : 0.0f;
        cumulative = simd_prefix_inclusive_sum(lane < RCH_PIECE ? gates[lane] : 0.0f);
        last = simd_shuffle(cumulative, ushort(RCH_PIECE - 1));
        row_cumulative = simd_shuffle(cumulative, row);
        row_beta = simd_shuffle(beta, row);
        row_gamma = metal::exp(row_cumulative);
        row_tail = metal::exp(last - row_cumulative);
    }

    float pair(ushort row, ushort j) const {
        return metal::exp(j <= row ? row_cumulative - simd_shuffle(cumulative, j) : 0.0f);
    }
    float gamma() const { return row_gamma; }
    float tail() const { return row_tail; }
    threadgroup const float *update_keys(threadgroup const float *keys) const { return keys; }
    void start_update() { last_factor = metal::exp(last); }
    float factor(uint, ushort) const { return last_factor; }
};

// The scan launch for `rule` and its decayed products `decays` (Kimi's with
// its `decayed` buffer [8, W]), over the entry's threadgroup buffers:
// Q and K [8, W], value channels [8, ROWS], the gates (`RchDecays::stage`;
// the row-sequential advance stages its betas at `gates` and its decay
// factors after RCH_PIECE of them) and the shares of A and D.
template <typename Rule>
inline void rch_scan(Rule rule, RchDecays<Rule::DECAYS> decays, device const recurrent::Storage *projection,
    device const float *convolution, device const int *segments, device const int *stop,
    device const int *previous_bank, device const int *previous_tape, device const int *following_bank,
    device const ulong *window, device const ulong *delta, device const ulong *tape, device float *inputs,
    device recurrent::Storage *mixed, threadgroup float *queries, threadgroup float *keys,
    threadgroup float *values, threadgroup float *gates, RchShares shares, uint3 group, uint thread_index,
    uint simdgroup, uint lane_index, constant ulong *seismic_words) {
    constexpr uint DECAYS = Rule::DECAYS;
    const ushort lane = ushort(lane_index);
    const ulong width = RCH_WIDTH;
    const ulong head = group.y;
    const ulong slot = group.z;
    const ulong block_row0 = ulong(group.x) * RCH_ROWS;
    const ulong row0 = block_row0 + ulong(simdgroup) * 16;
    const ulong key_head = recurrent::key_head(head, seismic_words);
    const RchScratch<DECAYS> scratch = rch_scratch<DECAYS>(inputs, seismic_words);
    const recurrent::Slot geometry = recurrent::slot_of(segments, stop, previous_bank, previous_tape,
        following_bank, slot, seismic_words);
    const ulong source = geometry.source;
    const ulong target = geometry.target;
    // The first row block publishes this head's window channels (for the
    // state after `stop` rows, and the tape rows).
    if (group.x == 0)
        recurrent::publish_window(projection, window, geometry, head, thread_index, RCH_THREADS, seismic_words);
    // Rows [begin, hi) advance row-sequentially (RCH_SEQUENTIAL): the step's
    // arithmetic, 16 state rows per simdgroup, spans of RCH_PIECE rows in the
    // piece buffers.
#define RCH_SEQUENTIAL(begin)                                                                               \
    recurrent::advance_rows<16, RCH_ROWS, RCH_PIECE, RCH_INPUT_STRIDE, RCH_VALUE_STRIDE>(                   \
        recurrent::Convolving{projection, window, convolution, geometry}, rule.gates(head, seismic_words),   \
        delta, tape, mixed, geometry, begin, head, block_row0, row0, queries, keys, values, gates,          \
        gates + RCH_PIECE, thread_index, RCH_THREADS, simdgroup, lane_index, seismic_words)
    if (geometry.hi - geometry.lo <= RECURRENT_SEQUENTIAL_ROWS) {
        // A short slot (an MTP verify) is not chunked.
        RCH_SEQUENTIAL(geometry.lo);
        rch_zero_tail(mixed, slot, geometry.hi, head, row0, lane, seismic_words);
        return;
    }

    // This lane's elements of an 8x8 simdgroup matrix: (row, column + e).
    const ushort quad = lane / 4;
    const ushort row = (quad & 4) + ((lane / 2) % 4);
    const ushort column = (quad & 2) * 2 + (lane % 2) * 2;

    // S^T tiles of the two 8-row halves: rows are key coordinates.
    const ulong state_stride = SEISMIC_DELTA_STRIDE_2;
    device float *published = recurrent::bank<float>(delta, target, SEISMIC_DELTA_STRIDE_0, seismic_words)
        + head * SEISMIC_DELTA_STRIDE_1 + row0 * state_stride;
    simdgroup_float8x8 state[2][RCH_TILES];
    {
        device const float *initial = recurrent::bank<float>(delta, source, SEISMIC_DELTA_STRIDE_0, seismic_words)
            + head * SEISMIC_DELTA_STRIDE_1 + row0 * state_stride;
        RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
            RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
                simdgroup_load(state[h][c], initial + 8 * h * state_stride + 8 * c, state_stride,
                    ulong2(0), true);
            }
        }
        // The source version's tape rows, with the step's update: element e of
        // tile (h, c) is S[row0 + 8h + column + e][8c + row], scaled by key
        // column 8c + row's factor (the head's one factor for one decay).
        constexpr uint FACTORS = DECAYS == 1 ? 1 : RCH_TILES;
        for (long entry = 0; entry < geometry.taped; ++entry) {
            device const float *tape_row = recurrent::tape_row(tape, source, entry, seismic_words);
            float factors[FACTORS];
            RECURRENT_UNROLL for (uint c = 0; c < FACTORS; ++c) {
                factors[c] = tape_row[recurrent::tape_decay<DECAYS>(head, 8 * c + row, seismic_words)];
            }
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
                    const float key_value = tape_row[RECURRENT_TAPE_K + key_head * width + 8 * c + row];
                    RECURRENT_UNROLL for (uint e = 0; e < 2; ++e) {
                        const float innovation = tape_row[RECURRENT_TAPE_U + head * width + row0 + 8 * h + column + e];
                        state[h][c].thread_elements()[e] *= factors[c % FACTORS];
                        state[h][c].thread_elements()[e] = metal::fma(innovation, key_value,
                            state[h][c].thread_elements()[e]);
                    }
                }
            }
        }
    }
    if (geometry.stop == 0) {
        RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
            RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
                simdgroup_store(state[h][c], published + 8 * h * state_stride + 8 * c,
                    state_stride, ulong2(0), true);
            }
        }
    }

    device const float *head_rows = scratch.rows + key_head * scratch.height * RCH_ROW_STRIDE;
    device const float *head_values = scratch.values + head * scratch.height * width + block_row0;
    device const float *head_gates = scratch.head_gates(head);
    const ulong pieces_before = recurrent::pieces_before<RCH_PIECE>(geometry);
    for (ulong piece = 0; piece < pieces_before; ++piece) {
        long first;
        long length;
        recurrent::piece_of<RCH_PIECE>(geometry, piece, first, length);

        // 1. Stage the piece's q/k rows, value channels and gates.
        for (uint item = thread_index; item < RCH_PIECE * RCH_ROW_STRIDE / 4; item += RCH_THREADS) {
            const uint i = item / (RCH_ROW_STRIDE / 4);
            const uint channel = 4 * (item % (RCH_ROW_STRIDE / 4));
            const float4 value = long(i) < length
                ? *reinterpret_cast<device const float4 *>(head_rows
                    + (ulong(first) + i) * RCH_ROW_STRIDE + channel) : 0.0f;
            *reinterpret_cast<threadgroup float4 *>((channel < width ? queries : keys)
                + i * RCH_INPUT_STRIDE + channel % width) = value;
        }
        for (uint item = thread_index; item < RCH_PIECE * RCH_ROWS / 4; item += RCH_THREADS) {
            const uint i = item / (RCH_ROWS / 4);
            const uint channel = 4 * (item % (RCH_ROWS / 4));
            *reinterpret_cast<threadgroup float4 *>(values + i * RCH_VALUE_STRIDE + channel)
                = long(i) < length ? *reinterpret_cast<device const float4 *>(head_values
                    + (ulong(first) + i) * width + channel) : 0.0f;
        }
        decays.stage(gates, head_gates, scratch.height, first, length, thread_index);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. The decayed products and X, Y.
        simdgroup_float8x8 removed[2];
        simdgroup_float8x8 output[2];
        decays.products(queries, keys, gates, shares, state, removed, output, thread_index, simdgroup, lane, row);

        // 3. A, D and I - A from the summed shares; T = (I - A)(I + A^2)(I + A^4).
        simdgroup_float8x8 system;
        simdgroup_float8x8 attention;
        simdgroup_float8x8 complement;
        RECURRENT_UNROLL for (ushort e = 0; e < 2; ++e) {
            const ushort j = ushort(column + e);
            float key_dot = 0.0f;
            float query_dot = 0.0f;
            RECURRENT_UNROLL for (uint s = 0; s < RCH_SHARES; ++s) {
                key_dot += shares[s][0][row * RCH_PIECE + j];
                query_dot += shares[s][1][row * RCH_PIECE + j];
            }
            const float decay = decays.pair(row, j);
            const float a = j < row ? decays.row_beta * decay * key_dot : 0.0f;
            system.thread_elements()[e] = a;
            complement.thread_elements()[e] = (j == row ? 1.0f : 0.0f) - a;
            attention.thread_elements()[e] = j <= row ? decay * query_dot : 0.0f;
        }
        simdgroup_float8x8 square;
        simdgroup_float8x8 fourth;
        simdgroup_multiply(square, system, system);
        simdgroup_multiply(fourth, square, square);
        RECURRENT_UNROLL for (ushort e = 0; e < 2; ++e) {
            const float identity = row == column + e ? 1.0f : 0.0f;
            square.thread_elements()[e] += identity;
            fourth.thread_elements()[e] += identity;
        }
        simdgroup_float8x8 partial;
        simdgroup_float8x8 inverse;
        simdgroup_multiply(partial, complement, square);
        simdgroup_multiply(inverse, partial, fourth);

        // Per half: Z = diag(beta) (V - diag(gamma) X), U = T Z,
        // O = diag(gamma) Y + D U, then U <- diag(tail) U.
        simdgroup_float8x8 update[2];
        RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
            simdgroup_float8x8 value;
            simdgroup_load(value, values + simdgroup * 16 + 8 * h, RCH_VALUE_STRIDE);
            RECURRENT_UNROLL for (ushort e = 0; e < 2; ++e) {
                removed[h].thread_elements()[e] = (value.thread_elements()[e]
                    - removed[h].thread_elements()[e] * decays.gamma()) * decays.row_beta;
                output[h].thread_elements()[e] *= decays.gamma();
            }
            simdgroup_multiply(update[h], inverse, removed[h]);
            simdgroup_multiply_accumulate(output[h], attention, update[h], output[h]);
            if (long(row) < length) {
                device recurrent::Storage *destination = mixed
                    + recurrent::raw_index(ulong(first + long(row)), head, row0 + 8 * h + column, seismic_words);
                RECURRENT_UNROLL for (ushort e = 0; e < 2; ++e) {
                    destination[e] = element::Act::store(output[h].thread_elements()[e]);
                }
            }
            update[h].thread_elements()[0] *= decays.tail();
            update[h].thread_elements()[1] *= decays.tail();
        }
        // S^T <- diag(f) S^T + K'^T U.
        decays.start_update();
        threadgroup const float *update_keys = decays.update_keys(keys);
        RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
            const float factor = decays.factor(c, row);
            simdgroup_float8x8 key;
            simdgroup_load(key, update_keys + 8 * c, RCH_INPUT_STRIDE, ulong2(0), true);
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                state[h][c].thread_elements()[0] *= factor;
                state[h][c].thread_elements()[1] *= factor;
                simdgroup_multiply_accumulate(state[h][c], key, update[h], state[h][c]);
            }
        }
        if (piece + 1 == pieces_before) {
            RECURRENT_UNROLL for (uint h = 0; h < 2; ++h) {
                RECURRENT_UNROLL for (uint c = 0; c < RCH_TILES; ++c) {
                    simdgroup_store(state[h][c], published + 8 * h * state_stride + 8 * c,
                        state_stride, ulong2(0), true);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The rows after the stop row advance row-sequentially from the published
    // state (their innovations feed the tape).
    if (geometry.stop < geometry.hi - geometry.lo) {
        threadgroup_barrier(mem_flags::mem_device);
        RCH_SEQUENTIAL(geometry.lo + geometry.stop);
    }
#undef RCH_SEQUENTIAL
    rch_zero_tail(mixed, slot, geometry.hi, head, row0, lane, seismic_words);
}
