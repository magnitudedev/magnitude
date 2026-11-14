// Device code of the CUDA chunked delta-rule entries (`gated_delta_chunk`,
// `kimi_delta_chunk`; contracts in recurrent.seismic), templated over the
// rule's gates (`recurrent::HeadGates`, `kimi::ChannelGates`): the chunked WY
// form over pieces of 16 rows.
//
// The slot's rows before its publication (stop) row split into pieces of at
// most 16 rows. Within a piece (rows t, G_t the cumulative log decays from the
// piece's start, one per head or one per key channel) with initial state S:
//   A[t][s] = beta_t sum_c k_tc k_sc exp(G_tc - G_sc) (s < t),
//   Tb = (I + A)^-1 diag(beta),
//   D[t][s] = sum_c q_tc k_sc exp(G_tc - G_sc) (s <= t),
//   U = Tb (V - (K o exp G) S^T),  O = (Q o exp G) S^T + D U,
//   S <- S diag(exp G_last) + U^T (K o exp(G_last - G)).
// Padding rows past a piece carry zero q, k, v, decay and beta. How the decay
// enters is the rule's `Decay` form: one decay per head factors out of every
// product (`Decay<.., 1>`: K K^T and Q K^T once per key head, the decays as
// F32 scalars on the products), one per key channel enters inside them
// (`Decay<.., W>`: A and D as direct sums, decayed f16 operand rows per value
// head).
//
// `prepare`, one block per (piece, key head): convolution, SiLU and L2 norms
// of the piece's q/k rows and of the v rows of the key head's value heads,
// the form's products, A and D, and per value head Tb (forward substitution,
// one lane per column). It writes per value head the f16 v rows, Tb and D,
// then the form's operands and gates.
// `scan`, one block per (ROWS state rows, value head, slot): each warp keeps
// 16 state rows as m16n8k16 accumulators (S rows by key columns, F32) and
// applies the slot's pieces in order on tensor cores (f16 operands, F32
// accumulation), the next piece's operands staged into shared memory with
// cp.async while the current one computes. The scan starts from the slot's
// source version (the bank's state advanced by its tape rows with the step's
// update) and publishes the state after the last piece, with the window; the
// rows after the stop row then advance row-sequentially from that state with
// the step's arithmetic (`recurrent::advance_rows`), recording the tape.
//
// A slot of at most recurrent::SEQUENTIAL_ROWS rows (an MTP verify) has no
// pieces: its scan blocks advance all its rows row-sequentially with the
// step's bits. The scan stores the raw outputs to the `mixed` scratch, and
// grid z = B zeroes the raw rows no slot covers; the gate launch then gates
// each row (`recurrent::gate_row`, one block per row). ROWS never changes
// result bits.

#include "recurrent.cuh"

namespace chunk {

using recurrent::Act;
using recurrent::C;
using recurrent::CPL;
using recurrent::NK;
using recurrent::NV;
using recurrent::u16;
using recurrent::u32;
using recurrent::u64;
using recurrent::u8;
using recurrent::W;

constexpr int PIECE = SEISMIC_CONST_CUDA_CHUNK_PIECE;
constexpr int GROUP = NV / NK;
constexpr int PREPARE_THREADS = 256;
constexpr int PREPARE_WARPS = PREPARE_THREADS / 32;
// Prepare vectors per row (q, k, and each value head's v). The convolution
// work is PARTS x RUNS runs of ROW_RUN consecutive rows of one vector, dealt
// round-robin to the warps. ROW_RUN (a divisor of PIECE) minimizes the rows a
// warp convolves, ceil(runs / warps) x ROW_RUN, preferring longer runs (each
// run reloads the C - 1 window rows before it). The run split never changes
// result bits: every row's convolution is computed the same way.
constexpr int PARTS = 2 + GROUP;
__host__ __device__ constexpr int prepare_row_run() {
    int best = PIECE;
    for (int run = PIECE; run >= 1; run /= 2) {
        const int runs = PARTS * (PIECE / run);
        const int rows = (runs + PREPARE_WARPS - 1) / PREPARE_WARPS * run;
        const int best_runs = PARTS * (PIECE / best);
        if (rows < (best_runs + PREPARE_WARPS - 1) / PREPARE_WARPS * best) best = run;
    }
    return best;
}
constexpr int ROW_RUN = prepare_row_run();
constexpr int RUNS = PIECE / ROW_RUN;
static_assert(NV % NK == 0 && W % 32 == 0, "chunk geometry");
static_assert(PIECE * PIECE == PREPARE_THREADS, "one prepare thread per (t, s) pair");
static_assert(GROUP <= 8, "the prepare block's static shared memory holds up to eight value heads per key head");
static_assert(SEISMIC_CONVOLUTION_STRIDE_1 == 1 && SEISMIC_CONVOLUTION_STRIDE_0 == C,
              "convolution taps are contiguous");

// A value head's piece record starts with its v rows, Tb and D; the form
// appends its own operands and gates.
constexpr u64 RECORD_V = 0;                                       // f16 [16][W]
constexpr u64 RECORD_TB = static_cast<u64>(PIECE) * W * 2;        // f16 [16][16]
constexpr u64 RECORD_D = RECORD_TB + PIECE * PIECE * 2;           // f16 [16][16]

// The slot's chunked pieces (the rows before its stop row): none for a
// sequential (short) slot.
__device__ __forceinline__ int scratch_pieces(const recurrent::Slot &slot) {
    return slot.hi - slot.lo <= recurrent::SEQUENTIAL_ROWS ? 0 : recurrent::pieces_before<PIECE>(slot);
}

// The global index of the first piece of slot `slot_index`.
__device__ __forceinline__ int first_piece(const recurrent::Inputs &in, u64 slot_index) {
    int total = 0;
    for (u64 slot = 0; slot < slot_index; ++slot) total += scratch_pieces(recurrent::slot_of(in, slot));
    return total;
}

// Value head `member` of key head `key_head` (inverse of recurrent::key_head).
__device__ __forceinline__ int member_head(const recurrent::Inputs &in, int key_head, int member) {
    return in.grouped ? key_head * GROUP + member : key_head + member * NK;
}

// Input x_j of the causal convolution for slot-local position j (the window of
// the slot's source version before the slot, the projection after), CPL
// channels from `channel`, in F32.
__device__ __forceinline__ void convolution_input(const recurrent::Inputs &in, const recurrent::Slot &slot,
                                                  int position, int channel, float (&out)[CPL]) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    if (position < 0)
        element::span<Act, CPL, true>(
            recurrent::window_at(in, slot.source, slot.taped + C - 1 + position, channel), 0, out);
    else
        element::span<Act, CPL, true>(in.projection,
                                static_cast<u64>(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0 +
                                    channel,
                                out);
}

__device__ __forceinline__ u32 pack(float lo, float hi) { return seismic_pack_f16x2(lo, hi); }

// An F32 operand as the sum of two f16 operands: `high` = f16(x) and `low` =
// f16(x - high), so a pair of MMAs carries about 22 bits of the operand.
struct Split {
    u32 high;
    u32 low;
};
__device__ __forceinline__ Split split(float first, float second) {
    const u32 high = pack(first, second);
    const float2 rounded = seismic_unpack_f16x2(high);
    return Split{high, pack(first - rounded.x, second - rounded.y)};
}
// A-operand fragments of a 16x16 tile from two 16x8 accumulator tiles
// (columns 0-7 and 8-15), split into high and low parts.
__device__ __forceinline__ void split_tiles(const float (&left)[4], const float (&right)[4],
                                            u32 (&high)[4], u32 (&low)[4]) {
    const Split parts[4] = {split(left[0], left[1]), split(left[2], left[3]),
                            split(right[0], right[1]), split(right[2], right[3])};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        high[i] = parts[i].high;
        low[i] = parts[i].low;
    }
}

// Pieces in flight: the scan computes one while the next STAGES - 1 load.
constexpr int STAGES = 3;

// Eight consecutive elements of E from F32, with 16-byte stores.
template <class E>
__device__ __forceinline__ void store_row8(u8 *base, u64 index, const float (&values)[8]) {
    if constexpr (E::bytes == 4) {
        float4 *out = reinterpret_cast<float4 *>(base + index * 4);
        out[0] = make_float4(values[0], values[1], values[2], values[3]);
        out[1] = make_float4(values[4], values[5], values[6], values[7]);
    } else {
        u32 words[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) words[k] = E::pack2(values[2 * k], values[2 * k + 1]);
        *reinterpret_cast<uint4 *>(base + index * 2) = make_uint4(words[0], words[1], words[2], words[3]);
    }
}

// How a rule's decay enters the chunked form. One decay per value head
// (DECAYS == 1) factors out of the products: the key head's f16 q | k | k_low
// rows lead each piece, K K^T and Q K^T are formed once per key head, and a
// value head's record carries gamma = exp(G), tail = exp(G_last - G) and
// exp(G_last) as F32 scalars, which the scan applies to the products. One
// decay per key channel (DECAYS == W) enters inside them: a value head's
// record carries its decayed operand rows Q o exp G | K o exp G | low |
// K o exp(G_last - G) | low and the channels' exp(G_last). A key's low part
// carries its products to about 22 bits.
template <class Rule, int DECAYS = Rule::DECAYS>
struct Decay;

template <class Rule>
struct Decay<Rule, 1> {
    static constexpr int OPERAND_WIDTH = 3 * W;
    // Operand offsets: queries, keys and their low parts for X and Y, and the
    // keys of the state update.
    static constexpr int QUERIES = 0, KEYS = W, KEYS_LOW = 2 * W, UPDATE_KEYS = W, UPDATE_KEYS_LOW = 2 * W;
    static constexpr u64 OPERAND_BYTES = static_cast<u64>(NK) * PIECE * OPERAND_WIDTH * 2;
    static constexpr u64 RECORD_GATES = RECORD_D + PIECE * PIECE * 2;  // f32 gamma [16], tail [16], last, 3 pad
    static constexpr int GATES = 2 * PIECE + 4;
    static constexpr u64 RECORD_BYTES = RECORD_GATES + GATES * 4;
    static constexpr u64 PIECE_BYTES = OPERAND_BYTES + NV * RECORD_BYTES;
    static_assert(RECORD_BYTES % 16 == 0 && OPERAND_BYTES % 16 == 0, "16-byte scratch records");
    // A scan stage's shared bytes beyond its 32 per state row.
    static constexpr unsigned STAGE_BYTES = 96 * W + 1680;

    __device__ static u8 *record(u8 *pieces, int piece, int head) {
        return pieces + static_cast<u64>(piece) * PIECE_BYTES + OPERAND_BYTES +
               static_cast<u64>(head) * RECORD_BYTES;
    }
    __device__ static u8 *operands(u8 *pieces, int piece, int key_head, int) {
        return pieces + static_cast<u64>(piece) * PIECE_BYTES +
               static_cast<u64>(key_head) * PIECE * OPERAND_WIDTH * 2;
    }
    // A normalized q (part 0) or k (part 1) element of row t, as the key
    // head's f16 operand (keys as high + low pairs).
    __device__ static void store_operand(u16 *rows, int t, int part, int column, float value) {
        const u16 high = seismic_f32_to_f16(value);
        rows[t * OPERAND_WIDTH + part * W + column] = high;
        if (part == 1)
            rows[t * OPERAND_WIDTH + 2 * W + column] = seismic_f32_to_f16(value - seismic_f16_to_f32(high));
    }

    // From the piece's normalized q and k rows: beta, A (in `system`), D and
    // the record's gates of the key head's value heads. Every thread calls it.
    __device__ static void products(const typename Rule::Inputs &in, u8 *pieces, const recurrent::Piece &piece,
                                    int key_head, const float (&queries)[PIECE][W + 1],
                                    const float (&keys)[PIECE][W + 1], float (&beta)[GROUP][PIECE],
                                    float (&system)[GROUP][PIECE][PIECE + 1]) {
        const recurrent::Inputs &base = Rule::base(in);
        const int n = piece.count;
        __shared__ float key_gram[PIECE][PIECE + 1];
        __shared__ float query_gram[PIECE][PIECE + 1];
        __shared__ float cumulative[GROUP][PIECE];
        // Gates of the value heads; padding rows have zero decay and beta.
        if (threadIdx.x < GROUP * PIECE) {
            const int member = threadIdx.x / PIECE;
            const int t = threadIdx.x % PIECE;
            float log_decay = 0.0f, b = 0.0f;
            if (t < n) {
                const recurrent::Gates gates = Rule::row_gates(in, piece.first + t, member_head(base, key_head, member));
                log_decay = gates.log_decay;
                b = gates.beta;
            }
            cumulative[member][t] = log_decay;
            beta[member][t] = b;
        }
        __syncthreads();

        // K K^T and Q K^T in F32, one (t, s) pair per thread (two partial sums
        // each); the inclusive cumulative log decays.
        {
            const int t = threadIdx.x / PIECE;
            const int s = threadIdx.x % PIECE;
            float kk[2] = {0.0f, 0.0f}, qk_dot[2] = {0.0f, 0.0f};
#pragma unroll 8
            for (int c = 0; c < W; c += 2) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    kk[h] = __fmaf_rn(keys[t][c + h], keys[s][c + h], kk[h]);
                    qk_dot[h] = __fmaf_rn(queries[t][c + h], keys[s][c + h], qk_dot[h]);
                }
            }
            key_gram[t][s] = kk[0] + kk[1];
            query_gram[t][s] = qk_dot[0] + qk_dot[1];
        }
        if (threadIdx.x < GROUP) {
            float sum = 0.0f;
            for (int t = 0; t < PIECE; ++t) {
                sum += cumulative[threadIdx.x][t];
                cumulative[threadIdx.x][t] = sum;
            }
        }
        __syncthreads();

        // Per value head: A in shared memory, D and the decays to the record.
        {
            const int t = threadIdx.x / PIECE;
            const int s = threadIdx.x % PIECE;
#pragma unroll
            for (int member = 0; member < GROUP; ++member) {
                const float *G = cumulative[member];
                const float decay = s <= t ? expf(G[t] - G[s]) : 0.0f;
                system[member][t][s] = s < t ? beta[member][t] * decay * key_gram[t][s] : 0.0f;
                u8 *out = record(pieces, blockIdx.x, member_head(base, key_head, member));
                reinterpret_cast<u16 *>(out + RECORD_D)[t * PIECE + s] =
                    seismic_f32_to_f16(t < n ? decay * query_gram[t][s] : 0.0f);
                if (threadIdx.x < PIECE) {
                    const float last = G[PIECE - 1];
                    float *gates = reinterpret_cast<float *>(out + RECORD_GATES);
                    gates[s] = expf(G[s]);
                    gates[PIECE + s] = expf(last - G[s]);
                    if (s == 0) gates[2 * PIECE] = expf(last);
                }
            }
        }
        __syncthreads();
    }

    // The staged gates' scalars: gamma_t on X and Y, tail_t on U, and the
    // state's factor of key column `column`.
    __device__ static float gamma(const float *gates, int t) { return gates[t]; }
    __device__ static float tail(const float *gates, int t) { return gates[PIECE + t]; }
    __device__ static float factor(const float *gates, int) { return gates[2 * PIECE]; }

    // A tape row's decay factors of value head `head`: its one decay, loaded
    // once.
    struct TapeFactors {
        float decay;
        __device__ TapeFactors(const float *row, int head) : decay(row[recurrent::tape_decay<1>(head, 0)]) {}
        __device__ float operator[](int) const { return decay; }
    };
};

template <class Rule>
struct Decay<Rule, W> {
    static_assert(W <= 128, "the prepare block's static shared memory holds per-channel decays to W 128");
    static constexpr int OPERAND_WIDTH = 5 * W;
    static constexpr int QUERIES = 0, KEYS = W, KEYS_LOW = 2 * W, UPDATE_KEYS = 3 * W, UPDATE_KEYS_LOW = 4 * W;
    static constexpr u64 RECORD_ROWS = RECORD_D + PIECE * PIECE * 2;                               // f16 [16][5W]
    static constexpr u64 RECORD_GATES = RECORD_ROWS + static_cast<u64>(PIECE) * OPERAND_WIDTH * 2;  // f32 [W]
    static constexpr int GATES = W;
    static constexpr u64 RECORD_BYTES = RECORD_GATES + GATES * 4;
    static constexpr u64 PIECE_BYTES = NV * RECORD_BYTES;
    static_assert(RECORD_BYTES == 196 * W + 1024, "the declaration's pieces bytes");
    static_assert(RECORD_BYTES % 16 == 0 && RECORD_GATES % 16 == 0, "16-byte scratch records");
    static constexpr unsigned STAGE_BYTES = 164 * W + 1536;

    __device__ static u8 *record(u8 *pieces, int piece, int head) {
        return pieces + static_cast<u64>(piece) * PIECE_BYTES + static_cast<u64>(head) * RECORD_BYTES;
    }
    __device__ static u8 *operands(u8 *pieces, int piece, int, int head) {
        return record(pieces, piece, head) + RECORD_ROWS;
    }
    // The decayed operands are formed per value head in `products`.
    __device__ static void store_operand(u16 *, int, int, int, float) {}

    // From the piece's normalized q and k rows: beta, then one member value
    // head at a time its per-channel cumulative log decays G [16][W] (zero on
    // padding rows), A (in `system`) and D as direct sums with each term's
    // exp(G_tc - G_sc) formed in place, the decayed operand rows and
    // exp(G_last). Every exponent is a sum of log decays of the piece's rows,
    // each <= 0, so no factor exceeds 1 and none overflows; a factor that
    // underflows F32 has a true value below 2^-126. An element of a decayed
    // f16 operand in f16's subnormal range is held to within 2^-25 absolutely
    // (high + low as well as high alone), against 2^-12 and 2^-23 relative in
    // the normal range. Every thread calls it.
    __device__ static void products(const typename Rule::Inputs &in, u8 *pieces, const recurrent::Piece &piece,
                                    int key_head, const float (&queries)[PIECE][W + 1],
                                    const float (&keys)[PIECE][W + 1], float (&beta)[GROUP][PIECE],
                                    float (&system)[GROUP][PIECE][PIECE + 1]) {
        const recurrent::Inputs &base = Rule::base(in);
        const int n = piece.count;
        __shared__ float decays[PIECE][W + 1];
        // Beta of the value heads; padding rows have zero beta.
        if (threadIdx.x < GROUP * PIECE) {
            const int member = threadIdx.x / PIECE;
            const int t = threadIdx.x % PIECE;
            beta[member][t] = t < n ? Rule::beta(in, piece.first + t, member_head(base, key_head, member)) : 0.0f;
        }
        __syncthreads();

        const int t = threadIdx.x / PIECE;
        const int s = threadIdx.x % PIECE;
        for (int member = 0; member < GROUP; ++member) {
            const int head = member_head(base, key_head, member);
            for (int item = threadIdx.x; item < PIECE * W; item += PREPARE_THREADS) {
                const int row = item / W;
                decays[row][item % W] = row < n ? Rule::log_decay(in, piece.first + row, head, item % W) : 0.0f;
            }
            __syncthreads();
            for (int c = threadIdx.x; c < W; c += PREPARE_THREADS) {
                float sum = 0.0f;
                for (int row = 0; row < PIECE; ++row) {
                    sum += decays[row][c];
                    decays[row][c] = sum;
                }
            }
            __syncthreads();
            float kk[2] = {0.0f, 0.0f}, qk_dot[2] = {0.0f, 0.0f};
            if (s <= t) {
#pragma unroll 8
                for (int c = 0; c < W; c += 2) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float decay = expf(decays[t][c + h] - decays[s][c + h]);
                        kk[h] = __fmaf_rn(keys[t][c + h] * keys[s][c + h], decay, kk[h]);
                        qk_dot[h] = __fmaf_rn(queries[t][c + h] * keys[s][c + h], decay, qk_dot[h]);
                    }
                }
            }
            system[member][t][s] = s < t ? beta[member][t] * (kk[0] + kk[1]) : 0.0f;
            u8 *out = record(pieces, blockIdx.x, head);
            reinterpret_cast<u16 *>(out + RECORD_D)[t * PIECE + s] =
                seismic_f32_to_f16(t < n ? qk_dot[0] + qk_dot[1] : 0.0f);
            u16 *rows = reinterpret_cast<u16 *>(out + RECORD_ROWS);
            for (int item = threadIdx.x; item < PIECE * W; item += PREPARE_THREADS) {
                const int row = item / W, c = item % W;
                const float last = decays[PIECE - 1][c];
                const float gamma = expf(decays[row][c]);
                const float key = keys[row][c];
                const float decayed[3] = {queries[row][c] * gamma, key * gamma, key * expf(last - decays[row][c])};
                u16 *operand = rows + row * OPERAND_WIDTH + c;
                operand[0] = seismic_f32_to_f16(decayed[0]);
#pragma unroll
                for (int k = 1; k < 3; ++k) {
                    const u16 high = seismic_f32_to_f16(decayed[k]);
                    operand[(2 * k - 1) * W] = high;
                    operand[2 * k * W] = seismic_f32_to_f16(decayed[k] - seismic_f16_to_f32(high));
                }
                if (row == 0) reinterpret_cast<float *>(out + RECORD_GATES)[c] = expf(last);
            }
            // The next member restages the decays.
            __syncthreads();
        }
    }

    // The decay is inside the operands: X, Y and U take no factor, and the
    // state's key column `column` decays by its exp(G_last).
    __device__ static float gamma(const float *, int) { return 1.0f; }
    __device__ static float tail(const float *, int) { return 1.0f; }
    __device__ static float factor(const float *gates, int column) { return gates[column]; }

    // A tape row's decay factors of value head `head`, one per key column.
    struct TapeFactors {
        const float *factors;
        __device__ TapeFactors(const float *row, int head) : factors(row + recurrent::tape_decay<W>(head, 0)) {}
        __device__ float operator[](int column) const { return factors[column]; }
    };
};

// L1: one block per (piece, key head).
template <class Rule>
__device__ __forceinline__ void prepare(const typename Rule::Inputs &in, u8 *pieces) {
    using Form = Decay<Rule>;
    const recurrent::Inputs &base = Rule::base(in);
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *base.words;
    const int key_head = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    // Locate this block's piece.
    int remaining = blockIdx.x;
    recurrent::Slot slot{};
    recurrent::Piece piece{0, 0};
    bool owned = false;
    for (u64 index = 0; index < SEISMIC_DIM_B && !owned; ++index) {
        const recurrent::Slot candidate = recurrent::slot_of(base, index);
        const int count = scratch_pieces(candidate);
        if (remaining < count) {
            slot = candidate;
            piece = recurrent::piece_of<PIECE>(candidate, remaining);
            owned = true;
        } else {
            remaining -= count;
        }
    }
    if (!owned) return;
    const int n = piece.count;

    __shared__ float queries[PIECE][W + 1];
    __shared__ float keys[PIECE][W + 1];
    __shared__ float beta[GROUP][PIECE];
    __shared__ float system[GROUP][PIECE][PIECE + 1];

    // Rows: the key head's q and k (L2-normalized, q scaled by W^-1/2) and
    // the v rows of its value heads. A run is one vector (part) of ROW_RUN
    // consecutive rows; its warp holds CPL channels per lane and slides the
    // convolution window along the run so each input row is loaded once.
    u16 *operands = reinterpret_cast<u16 *>(Form::operands(pieces, blockIdx.x, key_head, 0));
    for (int item = warp; item < PARTS * RUNS; item += PREPARE_WARPS) {
        const int part = item % PARTS;
        const int first = ROW_RUN * (item / PARTS);
        const int column = lane * CPL;
        const int channel = (part == 0   ? key_head * W
                             : part == 1 ? (NK + key_head) * W
                                         : (2 * NK + member_head(base, key_head, part - 2)) * W) +
                            column;
        float weights[C][CPL];
#pragma unroll
        for (int c = 0; c < CPL; ++c) {
            float taps[C];
            element::f32_span<C>(base.convolution + static_cast<u64>(channel + c) * C, taps);
#pragma unroll
            for (int tap = 0; tap < C; ++tap) weights[tap][c] = taps[tap];
        }
        const int local = piece.first - slot.lo + first;  // slot-local position of the run
        float inputs[C][CPL];
#pragma unroll
        for (int tap = 0; tap < C - 1; ++tap) {
            if (first < n) {
                convolution_input(base, slot, local + tap - (C - 1), channel, inputs[tap]);
            } else {
#pragma unroll
                for (int c = 0; c < CPL; ++c) inputs[tap][c] = 0.0f;
            }
        }
        u16 *v = part < 2 ? nullptr
                          : reinterpret_cast<u16 *>(
                                Form::record(pieces, blockIdx.x, member_head(base, key_head, part - 2)) +
                                RECORD_V);
        const float scale = part == 0 ? rsqrtf(static_cast<float>(W)) : 1.0f;
#pragma unroll
        for (int i = 0; i < ROW_RUN; ++i) {
            const int t = first + i;
            if (t < n) {
                convolution_input(base, slot, local + i, channel, inputs[C - 1]);
            } else {
#pragma unroll
                for (int c = 0; c < CPL; ++c) inputs[C - 1][c] = 0.0f;
            }
            float values[CPL];
            float squares = 0.0f;
#pragma unroll
            for (int c = 0; c < CPL; ++c) {
                float sum = weights[C - 1][c] * inputs[C - 1][c];
#pragma unroll
                for (int tap = 0; tap < C - 1; ++tap) sum = __fmaf_rn(weights[tap][c], inputs[tap][c], sum);
                values[c] = t < n ? sum / (1.0f + expf(-sum)) : 0.0f;
                squares = __fmaf_rn(values[c], values[c], squares);
            }
#pragma unroll
            for (int tap = 0; tap < C - 1; ++tap)
#pragma unroll
                for (int c = 0; c < CPL; ++c) inputs[tap][c] = inputs[tap + 1][c];
            if (part < 2) {
                const float inverse = t < n ? rsqrtf(seismic_warp_sum_f32(squares) + base.epsilon) * scale
                                            : 0.0f;
                float (*rows)[W + 1] = part == 0 ? queries : keys;
#pragma unroll
                for (int c = 0; c < CPL; ++c) {
                    values[c] *= inverse;
                    rows[t][column + c] = values[c];
                    Form::store_operand(operands, t, part, column + c, values[c]);
                }
            } else {
#pragma unroll
                for (int c = 0; c < CPL; ++c) v[t * W + column + c] = seismic_f32_to_f16(values[c]);
            }
        }
    }
    Form::products(in, pieces, piece, key_head, queries, keys, beta, system);

    // Tb = (I + A)^-1 diag(beta) by forward substitution: warp `member`, lane
    // j < 16 owning column j.
    for (int member = warp; member < GROUP; member += PREPARE_WARPS) {
        if (lane >= PIECE) continue;
        u16 *tb = reinterpret_cast<u16 *>(
            Form::record(pieces, blockIdx.x, member_head(base, key_head, member)) + RECORD_TB);
        float x[PIECE];
#pragma unroll
        for (int i = 0; i < PIECE; ++i) {
            float value = i == lane ? 1.0f : 0.0f;
#pragma unroll
            for (int k = 0; k < i; ++k) value = __fmaf_rn(-system[member][i][k], x[k], value);
            x[i] = value;
        }
        const float b = beta[member][lane];
#pragma unroll
        for (int i = 0; i < PIECE; ++i) tb[i * PIECE + lane] = seismic_f32_to_f16(x[i] * b);
    }
}

// One piece's operands in shared memory: the form's operand rows (row pitch
// padded by 8 halfs so ldmatrix rows fall in distinct banks), the block's v
// columns [16][ROWS], Tb and D [16][16] f16, and the form's gates.
template <class Form, unsigned ROWS>
struct Stage {
    static constexpr int QK_PITCH = Form::OPERAND_WIDTH + 8;
    static constexpr int V_PITCH = ROWS + 8;
    u16 qk[PIECE * QK_PITCH];
    u16 v[PIECE * V_PITCH];
    u16 td[2 * PIECE * PIECE];
    float gates[Form::GATES];
};

template <class Form, unsigned ROWS>
__device__ __forceinline__ void stage_piece(Stage<Form, ROWS> &stage, u8 *pieces, int piece, int key_head,
                                            int head, int row0) {
    constexpr int OPERAND_WIDTH = Form::OPERAND_WIDTH;
    constexpr int QK_PITCH = Stage<Form, ROWS>::QK_PITCH;
    constexpr int V_PITCH = Stage<Form, ROWS>::V_PITCH;
    constexpr int SCAN_THREADS = ROWS * 2;
    const u8 *qk = Form::operands(pieces, piece, key_head, head);
    const u8 *rec_bytes = Form::record(pieces, piece, head);
    constexpr int QK_CHUNKS = PIECE * OPERAND_WIDTH * 2 / 16;
    constexpr int V_CHUNKS = PIECE * ROWS * 2 / 16;
    constexpr int TD_CHUNKS = 2 * PIECE * PIECE * 2 / 16;
    constexpr int GATE_CHUNKS = Form::GATES * 4 / 16;
    for (int index = threadIdx.x; index < QK_CHUNKS + V_CHUNKS + TD_CHUNKS + GATE_CHUNKS;
         index += SCAN_THREADS) {
        if (index < QK_CHUNKS) {
            const int t = index / (OPERAND_WIDTH / 8);
            const int c = 8 * (index % (OPERAND_WIDTH / 8));
            seismic_cp_async_16(stage.qk + t * QK_PITCH + c, qk + (t * OPERAND_WIDTH + c) * 2);
        } else if (index < QK_CHUNKS + V_CHUNKS) {
            const int local = index - QK_CHUNKS;
            const int t = local / (ROWS / 8);
            const int c = 8 * (local % (ROWS / 8));
            seismic_cp_async_16(stage.v + t * V_PITCH + c,
                                rec_bytes + RECORD_V + (static_cast<u64>(t) * W + row0 + c) * 2);
        } else if (index < QK_CHUNKS + V_CHUNKS + TD_CHUNKS) {
            const int local = index - QK_CHUNKS - V_CHUNKS;
            seismic_cp_async_16(stage.td + 8 * local, rec_bytes + RECORD_TB + 16 * local);
        } else {
            const int local = index - QK_CHUNKS - V_CHUNKS - TD_CHUNKS;
            seismic_cp_async_16(stage.gates + 4 * local, rec_bytes + Form::RECORD_GATES + 16 * local);
        }
    }
}

// L2: one block per (ROWS state rows, value head, slot).
template <class Rule, unsigned ROWS>
__device__ __forceinline__ void scan(const typename Rule::Inputs &in, u8 *pieces, u8 *mixed) {
    using Form = Decay<Rule>;
    using Staged = Stage<Form, ROWS>;
    constexpr int SCAN_THREADS = ROWS * 2;
    constexpr int QK_PITCH = Staged::QK_PITCH;
    constexpr int V_PITCH = Staged::V_PITCH;
    static_assert(ROWS % 16 == 0 && W % ROWS == 0, "chunk scan geometry");
    static_assert(sizeof(Staged) == Form::STAGE_BYTES + 32 * ROWS, "the declaration's shared_bytes");
    static_assert(sizeof(recurrent::SequentialShared<ROWS>) <= STAGES * sizeof(Staged),
                  "the sequential advance fits the stage ring");
    const recurrent::Inputs &base = Rule::base(in);
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *base.words;
    const int head = blockIdx.y;
    const int row0 = blockIdx.x * ROWS;  // the block's first state row
    const u64 slot_index = blockIdx.z;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int g = lane / 4;
    const int t4 = lane % 4;
    const int warp_row = row0 + 16 * warp;  // this warp's first state row
    auto mixed_at = [&](int row, int state_row) { return recurrent::raw_index(row, head, state_row); };
    if (slot_index == SEISMIC_DIM_B) {
        const u64 covered = recurrent::covered_end(base);
        for (u64 index = covered * ROWS + threadIdx.x; index < SEISMIC_DIM_M * ROWS;
             index += SCAN_THREADS)
            element::put<Act>(mixed, mixed_at(static_cast<int>(index / ROWS), row0 + index % ROWS),
                          0.0f);
        return;
    }
    const recurrent::Slot slot = recurrent::slot_of(base, slot_index);
    recurrent::publish_window(base, slot, head * gridDim.x + blockIdx.x, NV * gridDim.x);
    extern __shared__ __align__(16) unsigned char scan_shared[];
    auto &sequential = *reinterpret_cast<recurrent::SequentialShared<ROWS> *>(scan_shared);
    if (slot.hi - slot.lo <= recurrent::SEQUENTIAL_ROWS) {
        recurrent::WarpRows<16> rows;
        recurrent::load_version<16>(base, Rule{in}, slot, head, warp_row, rows);
        recurrent::advance_rows<16, ROWS>(base, Rule{in}, slot, slot.lo, head, row0, true, warp_row, rows, mixed,
                                          sequential);
        return;
    }
    const int key_head = recurrent::key_head(base, head);

    // The warp's 16 state rows as accumulators: tile j holds S[row g, g + 8]
    // [key columns 8j + 2t4, + 1]; the source version's tape rows are applied
    // with the step's update.
    float state[W / 8][4];
    auto state_at = [&](int bank, int e) {
        return recurrent::state_row(base, bank, head, warp_row + g + 8 * (e / 2));
    };
    {
        const float *upper = state_at(slot.source, 0);
        const float *lower = state_at(slot.source, 2);
#pragma unroll
        for (int j = 0; j < W / 8; ++j) {
            const float2 top = *reinterpret_cast<const float2 *>(upper + 8 * j + 2 * t4);
            const float2 bottom = *reinterpret_cast<const float2 *>(lower + 8 * j + 2 * t4);
            state[j][0] = top.x;
            state[j][1] = top.y;
            state[j][2] = bottom.x;
            state[j][3] = bottom.y;
        }
        for (int entry = 0; entry < slot.taped; ++entry) {
            const float *row = recurrent::tape_row(base, slot.source, entry);
            const typename Form::TapeFactors factors(row, head);
            const float u[2] = {row[recurrent::TAPE_U + head * W + warp_row + g],
                                row[recurrent::TAPE_U + head * W + warp_row + g + 8]};
            const float *k = row + recurrent::TAPE_K + key_head * W;
#pragma unroll
            for (int j = 0; j < W / 8; ++j) {
                const float2 pair = *reinterpret_cast<const float2 *>(k + 8 * j + 2 * t4);
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    state[j][e] = __fmaf_rn(u[e / 2], e & 1 ? pair.y : pair.x,
                                            state[j][e] * factors[8 * j + 2 * t4 + (e & 1)]);
            }
        }
    }
    auto publish = [&]() {
        float *upper = state_at(slot.target, 0);
        float *lower = state_at(slot.target, 2);
#pragma unroll
        for (int j = 0; j < W / 8; ++j) {
            *reinterpret_cast<float2 *>(upper + 8 * j + 2 * t4) = make_float2(state[j][0], state[j][1]);
            *reinterpret_cast<float2 *>(lower + 8 * j + 2 * t4) = make_float2(state[j][2], state[j][3]);
        }
    };
    if (slot.stop == 0) publish();

    Staged *stages = reinterpret_cast<Staged *>(scan_shared);
    float (*transpose)[17] =
        reinterpret_cast<float (*)[17]>(scan_shared + STAGES * sizeof(Staged)) + 16 * warp;
    const int count = recurrent::pieces_before<PIECE>(slot);
    const int first = first_piece(base, slot_index);
    // One commit group per piece slot, empty past the last piece.
#pragma unroll
    for (int ahead = 0; ahead < STAGES - 1; ++ahead) {
        if (ahead < count) stage_piece<Form, ROWS>(stages[ahead], pieces, first + ahead, key_head, head, row0);
        seismic_cp_async_commit();
    }
    for (int local = 0; local < count; ++local) {
        const recurrent::Piece piece = recurrent::piece_of<PIECE>(slot, local);
        const Staged &stage = stages[local % STAGES];
        const int ahead = local + STAGES - 1;
        if (ahead < count)
            stage_piece<Form, ROWS>(stages[ahead % STAGES], pieces, first + ahead, key_head, head, row0);
        seismic_cp_async_commit();
        seismic_cp_async_wait<STAGES - 1>();
        __syncthreads();
        const u16 *queries = stage.qk + Form::QUERIES;
        const u16 *keys = stage.qk + Form::KEYS;
        const u16 *keys_low = stage.qk + Form::KEYS_LOW;
        const u16 *update_keys = stage.qk + Form::UPDATE_KEYS;
        const u16 *update_keys_low = stage.qk + Form::UPDATE_KEYS_LOW;

        // X^T = S K^T and Y^T = S Q^T: rows are state rows, columns piece
        // rows (two n-tiles).
        // Separate accumulators for S_high K_high, S_low K_high and S_high
        // K_low keep the MMA dependency chains short. The outputs, rounded to
        // the activation element, need only S_high Q.
        float removed[2][4] = {}, removed_low[2][4] = {}, removed_key_low[2][4] = {};
        float output[2][4] = {};
#pragma unroll
        for (int step = 0; step < W / 16; ++step) {
            u32 a[4], a_low[4];
            split_tiles(state[2 * step], state[2 * step + 1], a, a_low);
            const int t = (lane % 8) + 8 * (lane / 16);
            const int c = 16 * step + 8 * ((lane / 8) % 2);
            u32 kb[4], qb[4], lb[4];
            seismic_ldmatrix_x4(kb, keys + t * QK_PITCH + c);
            seismic_ldmatrix_x4(qb, queries + t * QK_PITCH + c);
            seismic_ldmatrix_x4(lb, keys_low + t * QK_PITCH + c);
            const u32 k0[2] = {kb[0], kb[1]}, k1[2] = {kb[2], kb[3]};
            const u32 l0[2] = {lb[0], lb[1]}, l1[2] = {lb[2], lb[3]};
            const u32 q0[2] = {qb[0], qb[1]}, q1[2] = {qb[2], qb[3]};
            seismic_mma_m16n8k16_f16(removed[0], a, k0);
            seismic_mma_m16n8k16_f16(removed[1], a, k1);
            seismic_mma_m16n8k16_f16(output[0], a, q0);
            seismic_mma_m16n8k16_f16(output[1], a, q1);
            seismic_mma_m16n8k16_f16(removed_low[0], a_low, k0);
            seismic_mma_m16n8k16_f16(removed_low[1], a_low, k1);
            seismic_mma_m16n8k16_f16(removed_key_low[0], a, l0);
            seismic_mma_m16n8k16_f16(removed_key_low[1], a, l1);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                removed[j][e] += removed_low[j][e] + removed_key_low[j][e];
            }

        // Z^T = V^T - X^T diag(gamma); U^T = Z^T Tb^T; O^T = Y^T diag(gamma) + U^T D^T
        // (gamma 1 when the decay is inside the operands).
        const int local_row = 16 * warp + g;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t = 8 * j + 2 * t4 + (e & 1);
                const int r = local_row + 8 * (e / 2);
                const float v = seismic_f16_to_f32(stage.v[t * V_PITCH + r]);
                removed[j][e] = v - removed[j][e] * Form::gamma(stage.gates, t);
                output[j][e] *= Form::gamma(stage.gates, t);
            }
        }
        u32 za[4], za_low[4];
        split_tiles(removed[0], removed[1], za, za_low);
        float update[2][4] = {};
        const u32 *td = reinterpret_cast<const u32 *>(stage.td);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const u32 tb[2] = {td[((8 * j + g) * PIECE + 2 * t4) / 2],
                               td[((8 * j + g) * PIECE + 2 * t4 + 8) / 2]};
            seismic_mma_m16n8k16_f16(update[j], za, tb);
            seismic_mma_m16n8k16_f16(update[j], za_low, tb);
        }
        const u32 ua[4] = {pack(update[0][0], update[0][1]), pack(update[0][2], update[0][3]),
                           pack(update[1][0], update[1][1]), pack(update[1][2], update[1][3])};
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const u32 d[2] = {td[(PIECE * PIECE + (8 * j + g) * PIECE + 2 * t4) / 2],
                              td[(PIECE * PIECE + (8 * j + g) * PIECE + 2 * t4 + 8) / 2]};
            seismic_mma_m16n8k16_f16(output[j], ua, d);
#pragma unroll
            for (int e = 0; e < 4; ++e)
                transpose[8 * j + 2 * t4 + (e & 1)][g + 8 * (e / 2)] = output[j][e];
        }
        // The outputs leave through the warp's transpose tile as rows of 8
        // state rows: lane l stores piece row l / 2, state rows 8 (l % 2) + 0..7.
        __syncwarp();
        {
            const int t = lane / 2;
            if (t < piece.count) {
                float values[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) values[k] = transpose[t][8 * (lane % 2) + k];
                store_row8<Act>(mixed, mixed_at(piece.first + t, warp_row + 8 * (lane % 2)), values);
            }
        }

        // S <- S diag(exp G_last) + (U diag(tail))^T K' (tail 1 and K' the
        // decayed update keys when the decay is inside the operands).
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) update[j][e] *= Form::tail(stage.gates, 8 * j + 2 * t4 + (e & 1));
        u32 wa[4], wa_low[4];
        split_tiles(update[0], update[1], wa, wa_low);
#pragma unroll
        for (int pair = 0; pair < W / 16; ++pair) {
            const int t = (lane % 8) + 8 * ((lane / 8) % 2);
            const int c = 16 * pair + 8 * (lane / 16);
            u32 kb[4], lb[4];
            seismic_ldmatrix_x4_trans(kb, update_keys + t * QK_PITCH + c);
            seismic_ldmatrix_x4_trans(lb, update_keys_low + t * QK_PITCH + c);
            const u32 k0[2] = {kb[0], kb[1]}, k1[2] = {kb[2], kb[3]};
            const u32 l0[2] = {lb[0], lb[1]}, l1[2] = {lb[2], lb[3]};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                // Tile j holds key columns 8j + 2 t4 + (e & 1).
                const int column = 16 * pair + 2 * t4 + (e & 1);
                state[2 * pair][e] *= Form::factor(stage.gates, column);
                state[2 * pair + 1][e] *= Form::factor(stage.gates, column + 8);
            }
            seismic_mma_m16n8k16_f16(state[2 * pair], wa, k0);
            seismic_mma_m16n8k16_f16(state[2 * pair + 1], wa, k1);
            seismic_mma_m16n8k16_f16(state[2 * pair], wa_low, k0);
            seismic_mma_m16n8k16_f16(state[2 * pair + 1], wa_low, k1);
            seismic_mma_m16n8k16_f16(state[2 * pair], wa, l0);
            seismic_mma_m16n8k16_f16(state[2 * pair + 1], wa, l1);
        }
        __syncthreads();
    }
    if (count > 0) publish();
    // The rows after the stop row advance row-sequentially from the published
    // state (their innovations feed the tape).
    if (slot.stop < slot.hi - slot.lo) {
        seismic_cp_async_wait<0>();
        __syncthreads();
        recurrent::WarpRows<16> rows;
        recurrent::load_rows<16>(base, slot.target, head, warp_row, rows);
        recurrent::advance_rows<16, ROWS>(base, Rule{in}, slot, slot.lo + slot.stop, head, row0, true, warp_row,
                                          rows, mixed, sequential);
    }
}
}  // namespace chunk
