// gated_delta_chunk (prefill row classes): the gated delta rule in the
// chunked WY form over pieces of 16 rows.
//
// The slot's rows before its publication (stop) row split into pieces of at
// most 16 rows. Within a piece (rows t, cumulative log decay G_t,
// gamma_t = exp(G_t)) with initial state S:
//   A[t][s] = beta_t exp(G_t - G_s) k_t.k_s (s < t),  Tb = (I + A)^-1 diag(beta),
//   D[t][s] = exp(G_t - G_s) q_t.k_s (s <= t),
//   U = Tb (V - diag(gamma) K S^T),  O = diag(gamma) Q S^T + D U,
//   S <- gamma_last S + U^T diag(gamma_last / gamma) K.
// Padding rows past a piece carry zero q, k, v, decay and beta.
//
// L1 `gated_delta_chunk_prepare`, one block per (piece, key head):
// convolution, SiLU and L2 norms of the piece's q/k rows and of the v rows of
// the key head's value heads, K K^T and Q K^T in F32, and per value head
// A, Tb (forward substitution, one lane per column) and D. It writes, per
// piece: the f16 q|k rows of each key head, and per value head the f16 v
// rows, Tb and D and the F32 gamma, gamma_last / gamma and gamma_last.
// L2 `gated_delta_chunk_scan`, one block per (ROWS state rows, value head,
// slot): each warp keeps 16 state rows as m16n8k16 accumulators (S rows by key
// columns, F32) and applies the slot's pieces in order on tensor cores
// (f16 operands, F32 accumulation), the next piece's operands staged into
// shared memory with cp.async while the current one computes. The scan starts
// from the slot's source version (the bank's state advanced by its tape rows
// with the step's update) and publishes the state after the last piece, with
// the window; the rows after the stop row then advance row-sequentially from
// that state with the step's arithmetic (`recurrent::advance_rows`), recording
// the tape.
//
// A slot of at most recurrent::SEQUENTIAL_ROWS rows (an MTP verify) has no
// pieces: its scan blocks advance all its rows row-sequentially with the
// step's bits. Grid z = B zeroes the mixed rows no slot covers. ROWS never
// changes result bits.

#include "lib/recurrent/recurrent.cuh"

namespace {

using recurrent::Act;
using recurrent::u16;
using recurrent::u32;
using recurrent::u64;
using recurrent::u8;
using recurrent::C;
using recurrent::NK;
using recurrent::NV;
using recurrent::W;

constexpr int PIECE = 16;
constexpr int GROUP = NV / NK;
constexpr int PREPARE_THREADS = 256;
constexpr int PREPARE_WARPS = PREPARE_THREADS / 32;
constexpr int CPL = W / 32;  // channels per lane
// Prepare vectors per row (q, k, and each value head's v) and the rows each
// warp convolves.
constexpr int PARTS = 2 + GROUP;
constexpr int ROW_RUN = PIECE * PARTS / PREPARE_WARPS;
static_assert(NV % NK == 0 && W % 32 == 0, "chunk geometry");
static_assert(PIECE * PIECE == PREPARE_THREADS, "one prepare thread per (t, s) pair");
static_assert(PREPARE_WARPS % PARTS == 0 && PIECE % (PREPARE_WARPS / PARTS) == 0,
              "prepare warps split evenly over vectors and rows");
static_assert(SEISMIC_CONVOLUTION_STRIDE_1 == 1 && SEISMIC_CONVOLUTION_STRIDE_0 == C,
              "convolution taps are contiguous");

// Scratch bytes per piece: the f16 q | k | k - f16(k) rows of every key head,
// then one record per value head. The key's low part carries the key's
// products to about 22 bits.
constexpr int QK_WIDTH = 3 * W;
constexpr u64 QK_BYTES = static_cast<u64>(NK) * PIECE * QK_WIDTH * 2;
constexpr u64 RECORD_V = 0;                                       // f16 [16][W]
constexpr u64 RECORD_TB = static_cast<u64>(PIECE) * W * 2;        // f16 [16][16]
constexpr u64 RECORD_D = RECORD_TB + PIECE * PIECE * 2;           // f16 [16][16]
constexpr u64 RECORD_GAMMA = RECORD_D + PIECE * PIECE * 2;        // f32 [16]
constexpr u64 RECORD_TAIL = RECORD_GAMMA + PIECE * 4;             // f32 [16]
constexpr u64 RECORD_LAST = RECORD_TAIL + PIECE * 4;              // f32, 3 pad
constexpr u64 RECORD_BYTES = RECORD_LAST + 16;
constexpr u64 PIECE_BYTES = QK_BYTES + NV * RECORD_BYTES;
static_assert(RECORD_BYTES % 16 == 0 && QK_BYTES % 16 == 0, "16-byte scratch records");

__device__ __forceinline__ u8 *qk_rows(u8 *pieces, int piece, int key_head) {
    return pieces + static_cast<u64>(piece) * PIECE_BYTES +
           static_cast<u64>(key_head) * PIECE * QK_WIDTH * 2;
}
__device__ __forceinline__ u8 *record(u8 *pieces, int piece, int head) {
    return pieces + static_cast<u64>(piece) * PIECE_BYTES + QK_BYTES +
           static_cast<u64>(head) * RECORD_BYTES;
}

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
            in.window, recurrent::window_at(in, slot.source, slot.taped + C - 1 + position, channel), out);
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

}  // namespace

#ifdef SEISMIC_FORMING_GATED_DELTA_CHUNK_PREPARE
__global__ void gated_delta_chunk_prepare(SEISMIC_KERNEL_PARAMS) {
    const recurrent::Inputs in = RECURRENT_INPUTS();
    u8 *pieces = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PIECES);
    const int key_head = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    // Locate this block's piece.
    int remaining = blockIdx.x;
    recurrent::Slot slot{};
    recurrent::Piece piece{0, 0};
    bool owned = false;
    for (u64 index = 0; index < SEISMIC_DIM_B && !owned; ++index) {
        const recurrent::Slot candidate = recurrent::slot_of(in, index);
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
    __shared__ float key_gram[PIECE][PIECE + 1];
    __shared__ float query_gram[PIECE][PIECE + 1];
    __shared__ float cumulative[GROUP][PIECE];
    __shared__ float beta[GROUP][PIECE];
    __shared__ float system[GROUP][PIECE][PIECE + 1];

    // Rows: the key head's q and k (L2-normalized, q scaled by W^-1/2) and
    // the v rows of its value heads. A warp owns one vector (part) of
    // ROW_RUN consecutive rows, CPL channels per lane, and slides the
    // convolution window along them so each input row is loaded once.
    u16 *qk = reinterpret_cast<u16 *>(qk_rows(pieces, blockIdx.x, key_head));
    {
        const int part = warp % PARTS;
        const int first = ROW_RUN * (warp / PARTS);
        const int column = lane * CPL;
        const int channel = (part == 0   ? key_head * W
                             : part == 1 ? (NK + key_head) * W
                                         : (2 * NK + member_head(in, key_head, part - 2)) * W) +
                            column;
        float weights[C][CPL];
#pragma unroll
        for (int c = 0; c < CPL; ++c) {
            float taps[C];
            element::f32_span<C>(in.convolution + static_cast<u64>(channel + c) * C, taps);
#pragma unroll
            for (int tap = 0; tap < C; ++tap) weights[tap][c] = taps[tap];
        }
        const int local = piece.first - slot.lo + first;  // slot-local position of the run
        float inputs[C][CPL];
#pragma unroll
        for (int tap = 0; tap < C - 1; ++tap) {
            if (first < n) {
                convolution_input(in, slot, local + tap - (C - 1), channel, inputs[tap]);
            } else {
#pragma unroll
                for (int c = 0; c < CPL; ++c) inputs[tap][c] = 0.0f;
            }
        }
        u16 *v = part < 2 ? nullptr
                          : reinterpret_cast<u16 *>(
                                record(pieces, blockIdx.x, member_head(in, key_head, part - 2)) +
                                RECORD_V);
        const float scale = part == 0 ? rsqrtf(static_cast<float>(W)) : 1.0f;
#pragma unroll
        for (int i = 0; i < ROW_RUN; ++i) {
            const int t = first + i;
            if (t < n) {
                convolution_input(in, slot, local + i, channel, inputs[C - 1]);
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
                const float inverse = t < n ? rsqrtf(seismic_warp_sum_f32(squares) + in.epsilon) * scale
                                            : 0.0f;
                float (*rows)[W + 1] = part == 0 ? queries : keys;
#pragma unroll
                for (int c = 0; c < CPL; ++c) {
                    values[c] *= inverse;
                    rows[t][column + c] = values[c];
                    const u16 high = seismic_f32_to_f16(values[c]);
                    qk[t * QK_WIDTH + part * W + column + c] = high;
                    if (part == 1)
                        qk[t * QK_WIDTH + 2 * W + column + c] =
                            seismic_f32_to_f16(values[c] - seismic_f16_to_f32(high));
                }
            } else {
#pragma unroll
                for (int c = 0; c < CPL; ++c) v[t * W + column + c] = seismic_f32_to_f16(values[c]);
            }
        }
    }
    // Gates of the value heads; padding rows have zero decay and beta.
    if (threadIdx.x < GROUP * PIECE) {
        const int member = threadIdx.x / PIECE;
        const int t = threadIdx.x % PIECE;
        float log_decay = 0.0f, b = 0.0f;
        if (t < n) {
            const recurrent::Gates gates = recurrent::gates(in, piece.first + t, member_head(in, key_head, member));
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
            u8 *out = record(pieces, blockIdx.x, member_head(in, key_head, member));
            reinterpret_cast<u16 *>(out + RECORD_D)[t * PIECE + s] =
                seismic_f32_to_f16(t < n ? decay * query_gram[t][s] : 0.0f);
            if (threadIdx.x < PIECE) {
                const float last = G[PIECE - 1];
                reinterpret_cast<float *>(out + RECORD_GAMMA)[s] = expf(G[s]);
                reinterpret_cast<float *>(out + RECORD_TAIL)[s] = expf(last - G[s]);
                if (s == 0) reinterpret_cast<float *>(out + RECORD_LAST)[0] = expf(last);
            }
        }
    }
    __syncthreads();

    // Tb = (I + A)^-1 diag(beta) by forward substitution: warp `member`, lane
    // j < 16 owning column j.
    for (int member = warp; member < GROUP; member += PREPARE_WARPS) {
        if (lane >= PIECE) continue;
        u16 *tb = reinterpret_cast<u16 *>(
            record(pieces, blockIdx.x, member_head(in, key_head, member)) + RECORD_TB);
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

#endif

namespace {

// One piece's operands in shared memory: q|k|k_low rows [16][3W] (row pitch
// padded by 8 halfs so ldmatrix rows fall in distinct banks), the block's v
// columns [16][ROWS], Tb and D [16][16] f16, gamma, tail, last.
constexpr int QK_PITCH = QK_WIDTH + 8;
template <unsigned ROWS>
struct Stage {
    static constexpr int V_PITCH = ROWS + 8;
    u16 qk[PIECE * QK_PITCH];
    u16 v[PIECE * V_PITCH];
    u16 td[2 * PIECE * PIECE];
    float gates[2 * PIECE + 4];
};
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

template <unsigned ROWS>
__device__ __forceinline__ void stage_piece(Stage<ROWS> &stage, u8 *pieces, int piece, int key_head,
                                            int head, int row0) {
    constexpr int V_PITCH = Stage<ROWS>::V_PITCH;
    constexpr int SCAN_THREADS = ROWS * 2;
    const u8 *qk = qk_rows(pieces, piece, key_head);
    const u8 *rec_bytes = record(pieces, piece, head);
    constexpr int QK_CHUNKS = PIECE * QK_WIDTH * 2 / 16;
    constexpr int V_CHUNKS = PIECE * ROWS * 2 / 16;
    constexpr int TD_CHUNKS = 2 * PIECE * PIECE * 2 / 16;
    constexpr int GATE_CHUNKS = (2 * PIECE + 4) * 4 / 16;
    for (int index = threadIdx.x; index < QK_CHUNKS + V_CHUNKS + TD_CHUNKS + GATE_CHUNKS;
         index += SCAN_THREADS) {
        if (index < QK_CHUNKS) {
            const int t = index / (QK_WIDTH / 8);
            const int c = 8 * (index % (QK_WIDTH / 8));
            seismic_cp_async_16(stage.qk + t * QK_PITCH + c, qk + (t * QK_WIDTH + c) * 2);
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
            seismic_cp_async_16(stage.gates + 4 * local, rec_bytes + RECORD_GAMMA + 16 * local);
        }
    }
}

}  // namespace

#ifdef SEISMIC_FORMING_GATED_DELTA_CHUNK_SCAN
template <unsigned ROWS>
__global__ void gated_delta_chunk_scan(SEISMIC_KERNEL_PARAMS) {
    constexpr int SCAN_THREADS = ROWS * 2;
    constexpr int V_PITCH = Stage<ROWS>::V_PITCH;
    static_assert(ROWS % 16 == 0 && W % ROWS == 0, "chunk scan geometry");
    static_assert(sizeof(Stage<ROWS>) == 96 * W + 32 * ROWS + 1680,
                  "the declaration's shared_bytes");
    static_assert(sizeof(recurrent::SequentialShared<ROWS>) <= STAGES * sizeof(Stage<ROWS>),
                  "the sequential advance fits the stage ring");
    const recurrent::Inputs in = RECURRENT_INPUTS();
    u8 *pieces = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PIECES);
    u8 *mixed = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    const int head = blockIdx.y;
    const int row0 = blockIdx.x * ROWS;  // the block's first state row
    const u64 slot_index = blockIdx.z;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int g = lane / 4;
    const int t4 = lane % 4;
    const int warp_row = row0 + 16 * warp;  // this warp's first state row
    auto mixed_at = [&](int row, int state_row) {
        return static_cast<u64>(row) * SEISMIC_RESULT_0_STRIDE_0 +
               static_cast<u64>(head) * SEISMIC_RESULT_0_STRIDE_1 +
               static_cast<u64>(state_row) * SEISMIC_RESULT_0_STRIDE_2;
    };
    if (slot_index == SEISMIC_DIM_B) {
        const u64 covered = recurrent::covered_end(in);
        for (u64 index = covered * ROWS + threadIdx.x; index < SEISMIC_DIM_M * ROWS;
             index += SCAN_THREADS)
            element::put<Act>(mixed, mixed_at(static_cast<int>(index / ROWS), row0 + index % ROWS),
                          0.0f);
        return;
    }
    const recurrent::Slot slot = recurrent::slot_of(in, slot_index);
    recurrent::publish_window(in, slot, head * gridDim.x + blockIdx.x, NV * gridDim.x);
    extern __shared__ __align__(16) unsigned char scan_shared[];
    auto &sequential = *reinterpret_cast<recurrent::SequentialShared<ROWS> *>(scan_shared);
    if (slot.hi - slot.lo <= recurrent::SEQUENTIAL_ROWS) {
        recurrent::WarpRows<16> rows;
        recurrent::load_version<16>(in, slot, head, warp_row, rows);
        recurrent::advance_rows<16, ROWS>(in, slot, slot.lo, head, row0, true, warp_row, rows, mixed, sequential);
        return;
    }
    const int key_head = recurrent::key_head(in, head);

    // The warp's 16 state rows as accumulators: tile j holds S[row g, g + 8]
    // [key columns 8j + 2t4, + 1]; the source version's tape rows are applied
    // with the step's update.
    float state[W / 8][4];
    auto state_at = [&](int bank, int e) {
        return recurrent::state_row(in, bank, head, warp_row + g + 8 * (e / 2));
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
            const recurrent::TapeEntry tape = recurrent::tape_entry(in, slot, head, entry);
            const float u[2] = {tape.row[recurrent::TAPE_U + head * W + warp_row + g],
                                tape.row[recurrent::TAPE_U + head * W + warp_row + g + 8]};
            const float *k = tape.row + recurrent::TAPE_K + key_head * W;
#pragma unroll
            for (int j = 0; j < W / 8; ++j) {
                const float2 pair = *reinterpret_cast<const float2 *>(k + 8 * j + 2 * t4);
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    state[j][e] = __fmaf_rn(u[e / 2], e & 1 ? pair.y : pair.x, state[j][e] * tape.decay);
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

    Stage<ROWS> *stages = reinterpret_cast<Stage<ROWS> *>(scan_shared);
    float (*transpose)[17] =
        reinterpret_cast<float (*)[17]>(scan_shared + STAGES * sizeof(Stage<ROWS>)) + 16 * warp;
    const int count = recurrent::pieces_before<PIECE>(slot);
    const int base = first_piece(in, slot_index);
    // One commit group per piece slot, empty past the last piece.
#pragma unroll
    for (int ahead = 0; ahead < STAGES - 1; ++ahead) {
        if (ahead < count) stage_piece<ROWS>(stages[ahead], pieces, base + ahead, key_head, head, row0);
        seismic_cp_async_commit();
    }
    for (int local = 0; local < count; ++local) {
        const recurrent::Piece piece = recurrent::piece_of<PIECE>(slot, local);
        const Stage<ROWS> &stage = stages[local % STAGES];
        const int ahead = local + STAGES - 1;
        if (ahead < count)
            stage_piece<ROWS>(stages[ahead % STAGES], pieces, base + ahead, key_head, head, row0);
        seismic_cp_async_commit();
        seismic_cp_async_wait<STAGES - 1>();
        __syncthreads();
        const u16 *queries = stage.qk;
        const u16 *keys = stage.qk + W;
        const u16 *keys_low = stage.qk + 2 * W;

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

        // Z^T = V^T - X^T diag(gamma); U^T = Z^T Tb^T; O^T = Y^T diag(gamma) + U^T D^T.
        const float *gamma = stage.gates;
        const float *tail = stage.gates + PIECE;
        const float last = stage.gates[2 * PIECE];
        const int local_row = 16 * warp + g;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t = 8 * j + 2 * t4 + (e & 1);
                const int r = local_row + 8 * (e / 2);
                const float v = seismic_f16_to_f32(stage.v[t * V_PITCH + r]);
                removed[j][e] = v - removed[j][e] * gamma[t];
                output[j][e] *= gamma[t];
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

        // S <- gamma_last S + (U diag(tail))^T K.
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) update[j][e] *= tail[8 * j + 2 * t4 + (e & 1)];
        u32 wa[4], wa_low[4];
        split_tiles(update[0], update[1], wa, wa_low);
#pragma unroll
        for (int pair = 0; pair < W / 16; ++pair) {
            const int t = (lane % 8) + 8 * ((lane / 8) % 2);
            const int c = 16 * pair + 8 * (lane / 16);
            u32 kb[4], lb[4];
            seismic_ldmatrix_x4_trans(kb, keys + t * QK_PITCH + c);
            seismic_ldmatrix_x4_trans(lb, keys_low + t * QK_PITCH + c);
            const u32 k0[2] = {kb[0], kb[1]}, k1[2] = {kb[2], kb[3]};
            const u32 l0[2] = {lb[0], lb[1]}, l1[2] = {lb[2], lb[3]};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                state[2 * pair][e] *= last;
                state[2 * pair + 1][e] *= last;
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
        recurrent::load_rows<16>(in, slot.target, head, warp_row, rows);
        recurrent::advance_rows<16, ROWS>(in, slot, slot.lo + slot.stop, head, row0, true, warp_row, rows, mixed,
                                          sequential);
    }
}
#endif
