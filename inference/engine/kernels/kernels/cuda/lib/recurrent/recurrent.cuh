// Shared device code of the CUDA delta-rule entries (`gated_delta_step`,
// `gated_delta_chunk`, and through `kimi.cuh` `kimi_delta_step` and
// `kimi_delta_chunk`; contracts in recurrent.seismic): slot and bank lookup,
// tensor addressing, Gated DeltaNet's per-head gates, piece splitting, the
// state-arena window publication, the row-sequential advance every entry
// runs for slots of at most SEQUENTIAL_ROWS rows, the gating of the raw
// outputs into the entries' result, and the step entries' body (`step`).
// They take the rule's gates (`HeadGates`, Kimi's `kimi::ChannelGates`) and
// the output gate's input (`GateInput`) as arguments.
// Activation tensors are canonical in their last axis.

#include "../core/activation.cuh"
#include "../core/arrive.cuh"
#include "../core/functions.cuh"
#include "versions.cuh"

namespace recurrent {

using element::u8;
using element::u16;
using element::u32;
using element::u64;
typedef activation::Act Act;
using Slot = versions::Slot;

constexpr int NK = static_cast<int>(SEISMIC_DIM_NK);
constexpr int NV = static_cast<int>(SEISMIC_DIM_NV);
constexpr int W = static_cast<int>(SEISMIC_DIM_W);
constexpr int C = static_cast<int>(SEISMIC_DIM_C);
constexpr int CH = (2 * NK + NV) * W;
// Columns of the projection's gate segments.
constexpr int ALPHA = CH + NV * W;
constexpr int BETA = ALPHA + NV;
// A tape row: the innovations u [NV, W], the normalized keys k [NK, W], the
// decay factors d [NV, DECAYS] (`HeadGates`).
constexpr int TAPE_U = 0;
constexpr int TAPE_K = NV * W;
constexpr int TAPE_D = NV * W + NK * W;

struct Inputs {
    // The launch's argument words, read by the ABI stride macros.
    const seismic_words_t *words;
    const u8 *projection;
    const float *convolution;
    const float *rate;
    const float *time_bias;
    versions::Slots slots;
    element::SlabOut<Act> window;
    element::SlabOut<element::F32> delta;
    element::SlabOut<element::F32> tape;
    float epsilon;
    bool grouped;
};

#define RECURRENT_INPUTS()                                                                    \
    recurrent::Inputs {                                                                       \
        &seismic_words_value, SEISMIC_PTR(SEISMIC_BUFFER_PROJECTION),                         \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_CONVOLUTION)),         \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RATE)),                \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_TIME_BIAS)),           \
            VERSIONS_SLOTS(),                                                                 \
            SEISMIC_OUT_WINDOW, SEISMIC_OUT_DELTA, SEISMIC_OUT_TAPE,                          \
            element::word_f32(SEISMIC_PARAM_NORM_EPSILON), SEISMIC_PARAM_GROUPED != 0         \
    }

__device__ __forceinline__ Slot slot_of(const Inputs &in, u64 slot) {
    return versions::slot_of(in.slots, slot, *in.words);
}

// Rows the slot records in its successor's tape: those after the stop row, at
// most T.
__device__ __forceinline__ int tape_rows(const Inputs &in, const Slot &slot) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return versions::tape_rows(slot, static_cast<int>(SEISMIC_DIM_T));
}

// Tape row `entry` of `bank`.
__device__ __forceinline__ element::Out<element::F32> tape_row(const Inputs &in, int bank, int entry) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return versions::bank(in.tape, bank, SEISMIC_TAPE_STRIDE_0, seismic_words_value)
        .offset(static_cast<u64>(entry) * SEISMIC_TAPE_STRIDE_1);
}

__device__ __forceinline__ element::Stored<element::F32> tape_row_view(const Inputs &in, int bank, int entry) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return versions::bank_view(in.tape, bank, SEISMIC_TAPE_STRIDE_0, seismic_words_value)
        .offset(static_cast<u64>(entry) * SEISMIC_TAPE_STRIDE_1);
}

// The first row no slot covers: slots partition a prefix of rows.
__device__ __forceinline__ int covered_end(const Inputs &in) {
    return versions::covered_end(in.slots, *in.words);
}

// Element (row, value head, state row) of the raw outputs, an [M, NV, W]
// scratch in A.
__device__ __forceinline__ u64 raw_index(int row, int head, int state_row) {
    return (static_cast<u64>(row) * NV + head) * W + state_row;
}

__device__ __forceinline__ float projection(const Inputs &in, u64 row, u64 column) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return element::at<Act>(in.projection, row * SEISMIC_PROJECTION_STRIDE_0 + column);
}

__device__ __forceinline__ element::Stored<Act> window_at(const Inputs &in, int bank, int tap, u64 channel) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return versions::bank_view(in.window, bank, SEISMIC_WINDOW_STRIDE_0, seismic_words_value)
        .offset(static_cast<u64>(tap) * SEISMIC_WINDOW_STRIDE_1 + channel);
}

// State row `state_row` of value head `head` in `bank`: W contiguous floats.
__device__ __forceinline__ element::Out<element::F32> state_row(const Inputs &in, int bank, int head, int state_row) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return versions::bank(in.delta, bank, SEISMIC_DELTA_STRIDE_0, seismic_words_value)
        .offset(static_cast<u64>(head) * SEISMIC_DELTA_STRIDE_1 + static_cast<u64>(state_row) * SEISMIC_DELTA_STRIDE_2);
}

__device__ __forceinline__ element::Stored<element::F32> state_row_view(const Inputs &in, int bank, int head,
                                                                        int state_row) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    return versions::bank_view(in.delta, bank, SEISMIC_DELTA_STRIDE_0, seismic_words_value)
        .offset(static_cast<u64>(head) * SEISMIC_DELTA_STRIDE_1 + static_cast<u64>(state_row) * SEISMIC_DELTA_STRIDE_2);
}

// The key head of value head `head`.
__device__ __forceinline__ int key_head(const Inputs &in, int head) {
    return in.grouped ? head * NK / NV : head % NK;
}

// beta = sigmoid(b) and the decay exp(log_decay), log_decay =
// rate * softplus(alpha + time_bias).
struct Gates {
    float beta;
    float log_decay;
    float decay;
};
__device__ __forceinline__ Gates gates(const Inputs &in, int row, int head) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const float alpha = projection(in, row, ALPHA + head);
    const float beta_input = projection(in, row, BETA + head);
    const float shifted = alpha + in.time_bias[head * SEISMIC_TIME_BIAS_STRIDE_0];
    const float softplus = fmaxf(shifted, 0.0f) + logf(1.0f + expf(-fabsf(shifted)));
    const float log_decay = in.rate[head * SEISMIC_RATE_STRIDE_0] * softplus;
    return Gates{1.0f / (1.0f + expf(-beta_input)), log_decay, expf(log_decay)};
}

// A lane's W / 32 state columns in the sequential layout.
constexpr int CPL = W / 32;

// The output gate's input z: column `column` of row `row` is element
// row * stride + first + column of `rows`, a value-head-major [NV, W] row.
struct GateInput {
    const u8 *rows;
    u64 stride;
    u64 first;
};

// A rule's gates, which the entry bodies (`step`, `chunk::prepare`,
// `chunk::scan`), `load_version` and `advance_rows` take: the rule's `Inputs`
// and their gated-delta operands (`base`); `at(row, head, lane, factor)`,
// value head `head`'s beta at `row`, setting `factor` to the decay factors of
// lane `lane`'s CPL key columns; and `gate_input()`, z. DECAYS is 1 (the
// head's one decay scales every key column) or W (one per key column); a tape
// row records DECAYS factors per value head. Gated DeltaNet's are columns of
// the projection row (`gates`), computed once per row, and z its columns from
// CH.
struct HeadGates {
    static constexpr int DECAYS = 1;
    using Inputs = recurrent::Inputs;
    const Inputs &in;
    __device__ static const Inputs &base(const Inputs &in) { return in; }
    // Row `row`'s gates of value head `head`, its one decay for every column.
    __device__ static Gates row_gates(const Inputs &in, int row, int head) { return gates(in, row, head); }
    __device__ __forceinline__ GateInput gate_input() const {
        [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
        return GateInput{in.projection, SEISMIC_PROJECTION_STRIDE_0, CH};
    }
    __device__ __forceinline__ float at(int row, int head, int, float (&factor)[CPL]) const {
        const Gates g = row_gates(in, row, head);
#pragma unroll
        for (int c = 0; c < CPL; ++c) factor[c] = g.decay;
        return g.beta;
    }
};

// The offset in a tape row of value head `head`'s decay factor of key
// column `column`, DECAYS factors per value head.
template <int DECAYS>
__device__ __forceinline__ int tape_decay(int head, int column) {
    static_assert(DECAYS == 1 || DECAYS == W, "one decay per value head or per key column");
    return TAPE_D + head * DECAYS + (DECAYS == 1 ? 0 : column);
}

// The chunked rows of a slot, the ones before its publication row lo + stop,
// split into pieces of at most `CHUNK` rows:
template <int CHUNK>
__device__ __forceinline__ int pieces_before(const Slot &slot) {
    return (slot.stop + CHUNK - 1) / CHUNK;
}
// Rows [first, first + count) of piece `index` (< pieces_before) of the slot.
struct Piece {
    int first;
    int count;
};
template <int CHUNK>
__device__ __forceinline__ Piece piece_of(const Slot &slot, int index) {
    const int first = index * CHUNK;
    return Piece{slot.lo + first, min(CHUNK, slot.stop - first)};
}

// Publish the slot's successor window: the C - 1 raw rows before the
// publication row, then the raw rows of its tape, moved bit-for-bit as stored.
// `part` of `parts` cooperating blocks writes an even share in 16-byte chunks.
__device__ __forceinline__ void publish_window(const Inputs &in, const Slot &slot, int part,
                                               int parts) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    constexpr int VEC = 16 / Act::bytes;
    static_assert(CH % VEC == 0, "window channels divide into 16-byte chunks");
    constexpr int CH_VECS = CH / VEC;
    const int rows = C - 1 + tape_rows(in, slot);
    const int total = rows * CH_VECS;
    const int per = (total + parts - 1) / parts;
    const int first = part * per;
    const int last = min(total, first + per);
    if (first >= last) return;
    const element::Stored<Act> source =
        versions::bank_view(in.window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words_value);
    const element::Out<Act> target =
        versions::bank(in.window, slot.target, SEISMIC_WINDOW_STRIDE_0, seismic_words_value);
    const int first_row = first / CH_VECS;
    const int last_row = (last - 1) / CH_VECS;
    for (int row = first_row; row <= last_row; ++row) {
        const int row_base = row * CH_VECS;
        const int col_lo = max(first, row_base) - row_base;
        const int col_hi = min(last, row_base + CH_VECS) - row_base;
        const int position = slot.stop + row - (C - 1);
        const element::Stored<Act> from_row =
            position < 0
                ? source.offset(static_cast<u64>(slot.taped + C - 1 + position) * SEISMIC_WINDOW_STRIDE_1)
                : element::stored<Act>(
                      in.projection, static_cast<u64>(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0);
        const element::Out<Act> to_row = target.offset(static_cast<u64>(row) * SEISMIC_WINDOW_STRIDE_1);
        for (int col = col_lo + static_cast<int>(threadIdx.x); col < col_hi; col += blockDim.x) {
            const u64 channel = static_cast<u64>(col) * VEC;
            to_row.move_stored(channel, from_row.offset(channel), VEC);
        }
    }
}

// The first member value head of `head`'s key head: the one head of each key
// head that records the key in a tape row.
__device__ __forceinline__ bool records_key(const Inputs &in, int head) {
    return in.grouped ? head % (NV / NK) == 0 : head < NK;
}

// Slots of at most this many rows advance row-sequentially in either entry,
// so a request's recurrent bits never depend on its row class or its peers
// (an MTP verify slot gets the bits of one-row decode).
constexpr int SEQUENTIAL_ROWS = SEISMIC_CONST_SEQUENTIAL_ROWS;

// Rows whose prologue the sequential advance stages at once.
constexpr int SPAN = 8;

// Shared memory of the sequential advance of a block owning BLOCK_ROWS state
// rows of one value head, for a span of rows: each row's L2-normalized q
// (scaled by W^-1/2) and k of the key head, the block's v channels, beta and
// the rule's DECAYS decay factors.
template <int BLOCK_ROWS, int DECAYS>
struct SequentialShared {
    float query[SPAN][W];
    float key[SPAN][W];
    float value[SPAN][BLOCK_ROWS];
    float beta[SPAN];
    float decay[SPAN][DECAYS];
};

// The C raw input row bases for the causal convolution at one slot-local row:
// the source version's window before the slot, the projection from row 0 on.
struct Taps {
    const u8 *row[C];
};

__device__ __forceinline__ Taps convolution_taps(const Inputs &in, const Slot &slot,
                                                 const element::Stored<Act> &window, int local) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    Taps taps;
#pragma unroll
    for (int tap = 0; tap < C - 1; ++tap) {
        const int position = local + tap - (C - 1);
        taps.row[tap] =
            position < 0
                ? window.bytes(static_cast<u64>(slot.taped + C - 1 + position) * SEISMIC_WINDOW_STRIDE_1)
                : in.projection + static_cast<u64>(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0 * Act::bytes;
    }
    taps.row[C - 1] = in.projection + static_cast<u64>(slot.lo + local) * SEISMIC_PROJECTION_STRIDE_0 * Act::bytes;
    return taps;
}

// SiLU of the causal depthwise convolution of `channel` from pre-resolved row
// bases `taps`.
__device__ __forceinline__ float convolved(const Inputs &in, const Taps &taps, int channel) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const float *weights = in.convolution + static_cast<u64>(channel) * SEISMIC_CONVOLUTION_STRIDE_0;
    float sum = weights[(C - 1) * SEISMIC_CONVOLUTION_STRIDE_1] *
                element::at<Act>(taps.row[C - 1], static_cast<u64>(channel));
#pragma unroll
    for (int tap = 0; tap < C - 1; ++tap)
        sum = __fmaf_rn(weights[tap * SEISMIC_CONVOLUTION_STRIDE_1],
                        element::at<Act>(taps.row[tap], static_cast<u64>(channel)), sum);
    return sum / (1.0f + expf(-sum));
}

// SiLU of the causal depthwise convolution of 4 contiguous channels starting at
// `channel`, loading each tap's 4 Act elements with a single vector load when
// aligned. Preserves `convolved`'s exact accumulation order.
__device__ __forceinline__ void convolved4(const Inputs &in, const Taps &taps, int channel, float (&out)[4]) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    constexpr u64 ALIGN = 4u * Act::bytes;
    float x[C][4];
#pragma unroll
    for (int tap = 0; tap < C; ++tap) {
        const u8 *base = taps.row[tap] + static_cast<u64>(channel) * Act::bytes;
        if ((reinterpret_cast<u64>(base) & (ALIGN - 1)) == 0) {
            element::span<Act, 4, false>(base, 0, x[tap]);
        } else {
#pragma unroll
            for (int c = 0; c < 4; ++c) x[tap][c] = element::at<Act>(base, static_cast<u64>(c));
        }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const float *weights = in.convolution + static_cast<u64>(channel + c) * SEISMIC_CONVOLUTION_STRIDE_0;
        float sum = weights[(C - 1) * SEISMIC_CONVOLUTION_STRIDE_1] * x[C - 1][c];
#pragma unroll
        for (int tap = 0; tap < C - 1; ++tap)
            sum = __fmaf_rn(weights[tap * SEISMIC_CONVOLUTION_STRIDE_1], x[tap][c], sum);
        out[c] = sum / (1.0f + expf(-sum));
    }
}

template <int N>
__device__ __forceinline__ void load_f32_span(const element::Stored<element::F32> &row, u64 offset,
                                              float (&values)[N]) {
    constexpr u64 ALIGN = N % 4 == 0 ? 16u : (N % 2 == 0 ? 8u : 4u);
    if ((reinterpret_cast<u64>(row.bytes(offset)) & (ALIGN - 1)) == 0) {
        row.span(offset, values);
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) values[i] = row.at(offset + i);
    }
}

template <int N>
__device__ __forceinline__ void put_f32_span(const element::Out<element::F32> &row, u64 offset,
                                             const float (&values)[N]) {
    constexpr u64 ALIGN = N % 4 == 0 ? 16u : (N % 2 == 0 ? 8u : 4u);
    if ((reinterpret_cast<u64>(row.stored(offset)) & (ALIGN - 1)) == 0) {
        row.put_span(offset, values);
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) row.put(offset + i, values[i]);
    }
}

// A warp's ROWS state rows in the sequential layout: lane l holds columns
// [l W / 32, (l + 1) W / 32) of each.
template <int ROWS>
using WarpRows = float[ROWS][CPL];

template <int ROWS>
__device__ __forceinline__ void load_rows(const Inputs &in, int bank, int head, int first_row,
                                          WarpRows<ROWS> &state) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int lane = threadIdx.x % 32;
    const element::Stored<element::F32> base = state_row_view(in, bank, head, first_row);
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
        base.offset(static_cast<u64>(r) * SEISMIC_DELTA_STRIDE_2).span(lane * CPL, state[r]);
}

template <int ROWS>
__device__ __forceinline__ void store_rows(const Inputs &in, int bank, int head, int first_row,
                                           const WarpRows<ROWS> &state) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int lane = threadIdx.x % 32;
    const element::Out<element::F32> base = state_row(in, bank, head, first_row);
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
        base.offset(static_cast<u64>(r) * SEISMIC_DELTA_STRIDE_2).put_span(lane * CPL, state[r]);
}

// A warp's ROWS rows of the slot's source version: the bank's state advanced
// by its first `taped` tape rows with the step's update (S <- S diag(d) +
// u k^T, the tape's DECAYS factors per value head of `gates`' rule), so the
// bits equal a run that published after those rows.
template <int ROWS, class RowGates>
__device__ __forceinline__ void load_version(const Inputs &in, const RowGates &, const Slot &slot, int head,
                                             int first_row, WarpRows<ROWS> &state) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    constexpr int DECAYS = RowGates::DECAYS;
    static_assert((NV * W + NK * W + NV * DECAYS) % 4 == 0, "16-byte tape rows");
    load_rows<ROWS>(in, slot.source, head, first_row, state);
    if (slot.taped <= 0) return;
    const int lane = threadIdx.x % 32;
    const int key = key_head(in, head);
    const element::Stored<element::F32> tape_bank = tape_row_view(in, slot.source, 0);
    for (int entry = 0; entry < slot.taped; ++entry) {
        const element::Stored<element::F32> row =
            tape_bank.offset(static_cast<u64>(entry) * SEISMIC_TAPE_STRIDE_1);
        float k[CPL], factor[CPL];
        load_f32_span(row, static_cast<u64>(TAPE_K + key * W + lane * CPL), k);
        if constexpr (DECAYS == 1) {
            const float d = row.at(tape_decay<1>(head, 0));
#pragma unroll
            for (int c = 0; c < CPL; ++c) factor[c] = d;
        } else {
            load_f32_span(row, static_cast<u64>(tape_decay<DECAYS>(head, lane * CPL)), factor);
        }
#pragma unroll
        for (int s = 0; s < ROWS; ++s) {
            const float u = row.at(TAPE_U + head * W + first_row + s);
#pragma unroll
            for (int c = 0; c < CPL; ++c) state[s][c] = __fmaf_rn(u, k[c], state[s][c] * factor[c]);
        }
    }
}

// The row-sequential delta rule over the slot's rows [begin, hi) with the
// rule's gates `gates` (`HeadGates`), the arithmetic of the step entries (a
// block's shape never changes bits).
// Every thread of the block calls it. The block owns state rows [block_row,
// block_row + BLOCK_ROWS) of value head `head`; each warp holds its ROWS rows
// from `first_row` in `state` (the state before row `begin`), advances them,
// writes their raw outputs to `mixed` (`raw_index`), publishes them after the
// slot's first `stop` rows (before any row when `begin` = lo and stop = 0)
// and records the rows after the stop row in the successor's tape. Per span
// of up to SPAN rows the block stages every row's prologue in one pass, all
// its loads in flight at once: a warp per (row, q or k) convolves (causal
// depthwise conv + SiLU) the key head's channels, W / 32 per lane, and
// L2-normalizes them (q scaled by W^-1/2), a warp per row forms the rule's
// gates, and the remaining threads convolve the block's v channels. After
// one barrier each warp advances its rows through the span: S <- S diag(d) +
// beta (v - S diag(d) k) k^T, output S q, with d the row's decay factors of
// each lane's columns.
template <int ROWS, int BLOCK_ROWS, class RowGates>
__device__ __forceinline__ void advance_rows(const Inputs &in, const RowGates &gates, const Slot &slot, int begin,
                                             int head, int block_row, int first_row, WarpRows<ROWS> &state, u8 *mixed,
                                             SequentialShared<BLOCK_ROWS, RowGates::DECAYS> &shared) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    constexpr int DECAYS = RowGates::DECAYS;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int warps = blockDim.x / 32;
    const int key_row = key_head(in, head);
    const int publish = slot.lo + slot.stop;
    const int taped = tape_rows(in, slot);
    const element::Stored<Act> window_source =
        versions::bank_view(in.window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words_value);
    const element::Out<element::F32> tape_target =
        taped > 0 ? tape_row(in, slot.target, 0) : element::Out<element::F32>(nullptr);
    // The warp that owns state row 0 records the head's decay factors, and
    // the key when the head records its key head's key.
    const bool records = first_row == 0;
    const bool records_keys = records && records_key(in, head);
    if (begin == slot.lo && slot.stop == 0) store_rows<ROWS>(in, slot.target, head, first_row, state);

    const float root = rsqrtf(static_cast<float>(W));
    for (int first = begin; first < slot.hi; first += SPAN) {
        const int count = min(SPAN, slot.hi - first);
        // Warp tasks: (row, q | k) normalizations, then each row's gates.
        for (int task = warp; task < 3 * count; task += warps) {
            const int r = task < 2 * count ? task / 2 : task - 2 * count;
            const int row = first + r;
            if (task < 2 * count) {
                const bool is_key = task % 2 != 0;
                const int channel = (is_key ? NK + key_row : key_row) * W + lane * CPL;
                const Taps taps = convolution_taps(in, slot, window_source, row - slot.lo);
                float values[CPL];
                if constexpr (CPL % 4 == 0) {
#pragma unroll
                    for (int c = 0; c < CPL; c += 4) {
                        float quad[4];
                        convolved4(in, taps, channel + c, quad);
#pragma unroll
                        for (int i = 0; i < 4; ++i) values[c + i] = quad[i];
                    }
                } else {
#pragma unroll
                    for (int c = 0; c < CPL; ++c) values[c] = convolved(in, taps, channel + c);
                }
                float squares = 0.0f;
#pragma unroll
                for (int c = 0; c < CPL; ++c) squares = __fmaf_rn(values[c], values[c], squares);
                const float inverse = rsqrtf(seismic_warp_sum_f32(squares) + in.epsilon) * (is_key ? 1.0f : root);
#pragma unroll
                for (int c = 0; c < CPL; ++c) values[c] *= inverse;
                element::f32_span_store<CPL>((is_key ? shared.key[r] : shared.query[r]) + lane * CPL, values);
            } else {
                float factor[CPL];
                const float beta = gates.at(row, head, lane, factor);
                if (lane == 0) shared.beta[r] = beta;
                if constexpr (DECAYS == 1) {
                    if (lane == 0) shared.decay[r][0] = factor[0];
                } else {
                    element::f32_span_store<CPL>(shared.decay[r] + lane * CPL, factor);
                }
            }
        }
        // The v channels, from the last thread down, off the warp tasks.
        for (int item = blockDim.x - 1 - threadIdx.x; item < count * BLOCK_ROWS; item += blockDim.x) {
            const int r = item / BLOCK_ROWS;
            const int col = item % BLOCK_ROWS;
            const Taps taps = convolution_taps(in, slot, window_source, first + r - slot.lo);
            shared.value[r][col] = convolved(in, taps, (2 * NK + head) * W + block_row + col);
        }
        __syncthreads();

        for (int r = 0; r < count; ++r) {
            const int row = first + r;
            float q[CPL], k[CPL], factor[CPL];
            element::f32_span(shared.query[r] + lane * CPL, q);
            element::f32_span(shared.key[r] + lane * CPL, k);
            if constexpr (DECAYS == 1) {
                const float d = shared.decay[r][0];
#pragma unroll
                for (int c = 0; c < CPL; ++c) factor[c] = d;
            } else {
                element::f32_span(shared.decay[r] + lane * CPL, factor);
            }
            const float beta = shared.beta[r];

            float remembered[ROWS];
#pragma unroll
            for (int s = 0; s < ROWS; ++s) {
                float sum = 0.0f;
#pragma unroll
                for (int c = 0; c < CPL; ++c) {
                    state[s][c] *= factor[c];
                    sum = __fmaf_rn(state[s][c], k[c], sum);
                }
                remembered[s] = sum;
            }
#pragma unroll
            for (int s = 0; s < ROWS; ++s) remembered[s] = seismic_warp_sum_f32(remembered[s]);
            // Rows after the stop row are recorded in the successor's tape.
            const bool recording = row >= publish && row - publish < taped;
            const element::Out<element::F32> entry =
                recording ? tape_target.offset(static_cast<u64>(row - publish) * SEISMIC_TAPE_STRIDE_1)
                          : element::Out<element::F32>(nullptr);
            if (recording && records_keys) {
                put_f32_span(entry, static_cast<u64>(TAPE_K + key_row * W + lane * CPL), k);
            }
            if (recording && records) {
                if constexpr (DECAYS == 1) {
                    if (lane == 0) entry.put(tape_decay<1>(head, 0), factor[0]);
                } else {
                    put_f32_span(entry, static_cast<u64>(tape_decay<DECAYS>(head, lane * CPL)), factor);
                }
            }
            float output[ROWS];
#pragma unroll
            for (int s = 0; s < ROWS; ++s) {
                const float v = shared.value[r][first_row - block_row + s];
                const float residual = (v - remembered[s]) * beta;
                if (recording && lane == s) entry.put(TAPE_U + head * W + first_row + s, residual);
                float sum = 0.0f;
#pragma unroll
                for (int c = 0; c < CPL; ++c) {
                    state[s][c] = __fmaf_rn(residual, k[c], state[s][c]);
                    sum = __fmaf_rn(state[s][c], q[c], sum);
                }
                output[s] = sum;
            }
#pragma unroll
            for (int s = 0; s < ROWS; ++s) output[s] = seismic_warp_sum_f32(output[s]);
            if (lane < ROWS) {
                float value = output[0];
#pragma unroll
                for (int s = 1; s < ROWS; ++s)
                    if (lane == s) value = output[s];
                element::put<Act>(mixed, raw_index(row, head, first_row + lane), value);
            }
            if (row + 1 == publish) store_rows<ROWS>(in, slot.target, head, first_row, state);
        }
        // The next span restages the shared rows.
        if (first + SPAN < slot.hi) __syncthreads();
    }
}

// Unpacks 32 bits of E (one F32 or two 16-bit elements) into F32 registers.
template <class E>
__device__ __forceinline__ void unpack_word(u32 bits, float *out) {
    if constexpr (E::bytes == 4) {
        out[0] = __uint_as_float(bits);
    } else {
        const float2 pair = E::unpack2(bits);
        out[0] = pair.x;
        out[1] = pair.y;
    }
}

// Loads `N` contiguous elements of E from `base` at `index` into F32 registers
// using the widest aligned vector load (`uint4`, `uint2`, `u32`), falling back
// to scalar loads when unaligned. `COHERENT` reads through L2 (`__ldcg`).
template <class E, int N, bool COHERENT>
__device__ __forceinline__ void load_span(const u8 *base, u64 index, float (&out)[N]) {
    constexpr int bytes = N * E::bytes;
    constexpr int per_word = 4 / E::bytes;
    const u8 *address = base + index * E::bytes;
    if constexpr (bytes % 16 == 0) {
        if ((reinterpret_cast<u64>(address) & 15u) == 0) {
            const uint4 *src = reinterpret_cast<const uint4 *>(address);
#pragma unroll
            for (int chunk = 0; chunk < bytes / 16; ++chunk) {
                uint4 word;
                if constexpr (COHERENT) {
                    word = __ldcg(src + chunk);
                } else {
                    word = src[chunk];
                }
                unpack_word<E>(word.x, out + chunk * 4 * per_word + 0 * per_word);
                unpack_word<E>(word.y, out + chunk * 4 * per_word + 1 * per_word);
                unpack_word<E>(word.z, out + chunk * 4 * per_word + 2 * per_word);
                unpack_word<E>(word.w, out + chunk * 4 * per_word + 3 * per_word);
            }
            return;
        }
    }
    if constexpr (bytes % 8 == 0) {
        if ((reinterpret_cast<u64>(address) & 7u) == 0) {
            const uint2 *src = reinterpret_cast<const uint2 *>(address);
#pragma unroll
            for (int chunk = 0; chunk < bytes / 8; ++chunk) {
                uint2 word;
                if constexpr (COHERENT) {
                    word = __ldcg(src + chunk);
                } else {
                    word = src[chunk];
                }
                unpack_word<E>(word.x, out + chunk * 2 * per_word + 0 * per_word);
                unpack_word<E>(word.y, out + chunk * 2 * per_word + 1 * per_word);
            }
            return;
        }
    }
    if constexpr (bytes % 4 == 0) {
        if ((reinterpret_cast<u64>(address) & 3u) == 0) {
            const u32 *src = reinterpret_cast<const u32 *>(address);
#pragma unroll
            for (int chunk = 0; chunk < bytes / 4; ++chunk) {
                u32 word;
                if constexpr (COHERENT) {
                    word = __ldcg(src + chunk);
                } else {
                    word = src[chunk];
                }
                unpack_word<E>(word, out + chunk * per_word);
            }
            return;
        }
    }
    const typename E::storage *src = reinterpret_cast<const typename E::storage *>(base) + index;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        if constexpr (COHERENT) {
            out[i] = E::load(__ldcg(src + i));
        } else {
            out[i] = E::load(src[i]);
        }
    }
}

// Gates (row, head) with one warp: each lane keeps its CPL = W / 32 contiguous
// columns in registers across the warp sum so `mixed` is read once with vector
// loads/stores. `COHERENT` reads through L2 (`__ldcg`) for values other blocks
// published in the same launch (`gate_head`).
template <class NORM, bool COHERENT>
__device__ __forceinline__ void gate_warp(const Inputs &in, const GateInput &input, const u8 *mixed, const u8 *norm,
                                          float eps, const element::Out<Act> &gated, int row, int head, bool covered) {
    static_assert(W % 32 == 0, "head width divides into warp lanes");
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int lane = threadIdx.x % 32;
    const int col0 = lane * CPL;
    float values[CPL];
    if (covered) {
        load_span<Act, CPL, COHERENT>(mixed, raw_index(row, head, col0), values);
    } else {
#pragma unroll
        for (int c = 0; c < CPL; ++c) values[c] = 0.0f;
    }
    float norm_values[CPL], z_values[CPL];
    load_span<NORM, CPL, false>(norm, static_cast<u64>(col0), norm_values);
    load_span<Act, CPL, false>(input.rows,
                               static_cast<u64>(row) * input.stride + input.first + static_cast<u64>(head) * W + col0,
                               z_values);
    float total = 0.0f;
#pragma unroll
    for (int c = 0; c < CPL; ++c) total = __fmaf_rn(values[c], values[c], total);
    // The butterfly leaves every lane the same sum.
    total = seismic_warp_sum_f32(total);
    const float inverse = rsqrtf(total / static_cast<float>(W) + eps);
    float out_values[CPL];
#pragma unroll
    for (int c = 0; c < CPL; ++c) {
        const float normalized = Act::round(values[c] * inverse * norm_values[c]);
        const float activated = Act::round(functions::output_gate((int)SEISMIC_PARAM_GATE_FUNCTION, z_values[c]));
        out_values[c] = normalized * activated;
    }
    const u64 out_base = static_cast<u64>(row) * SEISMIC_RESULT_0_STRIDE_0 +
                         static_cast<u64>(head) * SEISMIC_RESULT_0_STRIDE_1;
    constexpr u64 OUT_ALIGN = Act::bytes == 4 ? (CPL % 4 == 0 ? 16u : (CPL % 2 == 0 ? 8u : 4u)) : 2u;
    if (SEISMIC_RESULT_0_STRIDE_2 == 1 &&
        (reinterpret_cast<u64>(gated.stored(out_base + col0)) & (OUT_ALIGN - 1)) == 0) {
        gated.put_span(out_base + col0, out_values);
    } else {
#pragma unroll
        for (int c = 0; c < CPL; ++c) {
            gated.put(out_base + static_cast<u64>(col0 + c) * SEISMIC_RESULT_0_STRIDE_2, out_values[c]);
        }
    }
}

// The `gate` launch of every entry: block `row` forms the row's gated result
// from its raw outputs `mixed`. Per value head (one warp each, each lane
// owning CPL = W / 32 contiguous columns, then a warp sum),
//   gated = round_A(round_A(raw * rsqrt(sum_head raw^2 / W + eps) * norm[i])
//                   * round_A(gate(z)))
// with gate the output gate of `gate_function` (`functions::output_gate`)
// with z column head * W + i of the row's `input`.
template <class NORM>
__device__ __forceinline__ void gate_row(const Inputs &in, const GateInput &input, const u8 *mixed, const u8 *norm,
                                         float eps, const element::Out<Act> &gated, int row) {
    // A row no slot covers has zero raw outputs, which no block stores.
    const bool covered = row < covered_end(in);
    const int warps = blockDim.x / 32;
    for (int head = threadIdx.x / 32; head < NV; head += warps)
        gate_warp<NORM, false>(in, input, mixed, norm, eps, gated, row, head, covered);
}

// `gate_row`'s arithmetic for one value head over every row, one warp per
// row, by the last of the head's step blocks to arrive: the raw outputs the
// other blocks published are read from L2.
template <class NORM>
__device__ __forceinline__ void gate_head(const Inputs &in, const GateInput &input, const u8 *mixed, const u8 *norm,
                                          float eps, const element::Out<Act> &gated, int head) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int covered = covered_end(in);
    for (int row = threadIdx.x / 32; row < static_cast<int>(SEISMIC_DIM_M); row += blockDim.x / 32)
        gate_warp<NORM, true>(in, input, mixed, norm, eps, gated, row, head, row < covered);
}

// The row-sequential step entries' body (`gated_delta_step`,
// `kimi_delta_step`) over the rule's gates: one block per (value head,
// state-row block, slot); slots are a grid axis. Each warp owns ROWS state
// rows (value coordinates), a lane W / 32 columns (key coordinates) of each,
// held in registers for the whole slot. The state is read from version
// (previous_bank, previous_tape)[slot] and published to following_bank[slot]
// after the slot's first stop[slot] rows, with the window and the tape of the
// rows after it. The raw outputs of the slot's rows go to the `mixed`
// scratch. The last of a value head's blocks to arrive (`arrivals`, one
// counter per head) gates the head's rows (`gate_head`: `gate_row`'s
// arithmetic), every row of the result. ROWS and WARPS never change result
// bits, and the chunked entries give their short slots the same bits.
template <unsigned ROWS, unsigned WARPS, class NORM, class RowGates>
__device__ __forceinline__ void step(const typename RowGates::Inputs &rule, u8 *mixed, unsigned *arrivals,
                                     const u8 *norm, float eps, const element::Out<Act> &gated) {
    constexpr int BLOCK_ROWS = ROWS * WARPS;
    static_assert(W % 32 == 0 && W % BLOCK_ROWS == 0, "state rows split evenly");
    const Inputs &in = RowGates::base(rule);
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int head = blockIdx.x;
    const int block_row = blockIdx.y * BLOCK_ROWS;
    const int first_row = block_row + threadIdx.x / 32 * ROWS;  // this warp's state rows
    // Let a programmatic dependent start (its first weight loads overlap this
    // launch; it still waits for this launch's completion).
    seismic_dependents_launch();

    const Slot slot = slot_of(in, blockIdx.z);
    if (slot.lo < slot.hi) {
        WarpRows<ROWS> state;
        load_version<ROWS>(in, RowGates{rule}, slot, head, first_row, state);
        publish_window(in, slot, head * gridDim.y + blockIdx.y, NV * gridDim.y);
        __shared__ __align__(16) SequentialShared<BLOCK_ROWS, RowGates::DECAYS> shared;
        advance_rows<ROWS, BLOCK_ROWS>(in, RowGates{rule}, slot, slot.lo, head, block_row, first_row, state, mixed,
                                       shared);
    }

    __shared__ unsigned last;
    if (!arrive::last(arrivals + head, gridDim.y * gridDim.z, &last)) return;
    gate_head<NORM>(in, RowGates{rule}.gate_input(), mixed, norm, eps, gated, head);
}


}  // namespace recurrent
