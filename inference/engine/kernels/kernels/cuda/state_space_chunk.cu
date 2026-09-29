// state_space_chunk (prefill): the chunked state-space advance in the
// state-space dual form, the CUDA form of `metal/state_space_chunk.metal`.
// `inputs` convolves the B and C channels of every row once into scratch with
// each head's step size and log decay. `products` forms, once per piece and
// group, C_i . B_j (j <= i) of the piece's rows from B and C staged in shared
// memory. In `scan` a block of four warps owns (head, block of ROWS state
// rows, slot) with the step's geometry. Per piece of at most PIECE rows
// before the stop row, with gamma_i = exp(sum of the piece's log decays to
// row i) and u_j = delta_j x_j:
//   y_i = gamma_i (S C_i) + sum_{j <= i} (gamma_i / gamma_j) (C_i . B_j) u_j + D x_i
//   S  <- gamma_last S + sum_j (gamma_last / gamma_j) u_j B_j^T
// The S C_i sums of 32 / LANE_ROWS rows at a time are reduce-scattered over
// the warp, so each lane finishes one (row, state row) output with the
// butterfly's bits. Slots of at most SEQUENTIAL_ROWS rows are not chunked, and
// the rows after a stop row advance row-sequentially from the published
// state, both with `state_space_step`'s bits. ROWS never changes result bits.

#include "lib/recurrent/state_space.cuh"

namespace state_space {

// The staged inputs of row `row`.
__device__ __forceinline__ const float *staged(const float *inputs, int row) {
    return inputs + static_cast<u64>(row) * INPUT_WIDTH;
}

} // namespace state_space

#ifdef SEISMIC_FORMING_STATE_SPACE_CHUNK_INPUTS
// One row per block x, 256 of its B and C channels and step sizes per block y.
// Rows no slot covers are no slot's input.
__global__ void state_space_chunk_inputs(SEISMIC_KERNEL_PARAMS) {
    using namespace state_space;
    const Operands in = STATE_SPACE_OPERANDS();
    float *inputs = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_INPUTS));
    const int row = blockIdx.x;
    const u64 slot_index = versions::slot_index_of_row(in.slots, row, seismic_words_value);
    if (slot_index == SEISMIC_DIM_B)
        return;
    const versions::Slot slot = versions::slot_of(in.slots, slot_index, seismic_words_value);
    const int item = blockIdx.y * 256 + threadIdx.x;
    float *out = inputs + static_cast<u64>(row) * INPUT_WIDTH;
    if (item < 2 * G * N) {
        out[INPUT_B + item] = convolve(in, slot, row - slot.lo, B_CHANNEL + item);
    } else if (item < 2 * G * N + NH) {
        const int head = item - 2 * G * N;
        const Step step = step_of(in, row, head);
        out[INPUT_DELTA + head] = step.delta;
        out[INPUT_LOG_DECAY + head] = step.log_decay;
    }
}
#endif

#ifdef SEISMIC_FORMING_STATE_SPACE_CHUNK_PRODUCTS
// Piece x (numbered over the slots), group y: C_i . B_j for j <= i, an F32 FMA
// chain over the state coordinate, B and C staged in shared memory 32 state
// columns at a time. A thread owns pairs t, t + 256, t + 512 of the piece's
// PIECE * (PIECE + 1) / 2 lower-triangle pairs.
__global__ void state_space_chunk_products(SEISMIC_KERNEL_PARAMS) {
    using namespace state_space;
    constexpr int SPAN = 32;
    constexpr int PAIRS = PIECE * (PIECE + 1) / 2;
    constexpr int OWNED = (PAIRS + 255) / 256;
    const Operands in = STATE_SPACE_OPERANDS();
    const float *inputs = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_INPUTS));
    float *products = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCTS));
    __shared__ float c_rows[PIECE][SPAN + 1];
    __shared__ float b_rows[PIECE][SPAN + 1];
    versions::Slot slot;
    int local;
    if (!piece_at(in, blockIdx.x, slot, local))
        return;
    const int first = slot.lo + local * PIECE;
    const int count = min(PIECE, slot.stop - local * PIECE);
    const int g = blockIdx.y;
    int rows[OWNED];
    int columns[OWNED];
    float sums[OWNED];
#pragma unroll
    for (int k = 0; k < OWNED; ++k) {
        // Pair p = i (i + 1) / 2 + j of the lower triangle.
        const int pair = threadIdx.x + 256 * k;
        int i = 0;
        while ((i + 1) * (i + 2) / 2 <= pair) ++i;
        rows[k] = pair < PAIRS ? i : PIECE;
        columns[k] = pair - i * (i + 1) / 2;
        sums[k] = 0.0f;
    }
    for (int base = 0; base < N; base += SPAN) {
        for (int item = threadIdx.x; item < count * SPAN; item += blockDim.x) {
            const int i = item / SPAN;
            const int n = item % SPAN;
            c_rows[i][n] = staged(inputs, first + i)[INPUT_C + g * N + base + n];
            b_rows[i][n] = staged(inputs, first + i)[INPUT_B + g * N + base + n];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < OWNED; ++k) {
            if (rows[k] < count)
                for (int n = 0; n < SPAN; ++n)
                    sums[k] = seismic_fma_rn(c_rows[rows[k]][n], b_rows[columns[k]][n], sums[k]);
        }
        __syncthreads();
    }
    float *out = products + (static_cast<u64>(blockIdx.x) * G + g) * PIECE * PIECE;
#pragma unroll
    for (int k = 0; k < OWNED; ++k)
        if (rows[k] < count)
            out[rows[k] * PIECE + columns[k]] = sums[k];
}
#endif

#ifdef SEISMIC_FORMING_STATE_SPACE_CHUNK_SCAN
namespace state_space {

// Shared memory of a piece: x rows of the block's state rows, the weights
// (gamma_i / gamma_j) (C_i . B_j), the cumulative log decays, gamma, the tail
// factors gamma_last / gamma_j, the step sizes. Weight rows are padded so the
// lanes reading different rows at one column hit different banks.
template <int BLOCK_ROWS> struct PieceShared {
    float x[PIECE][BLOCK_ROWS];
    float weights[PIECE][PIECE + 1];
    float log_gamma[PIECE];
    float gamma[PIECE];
    float tail[PIECE];
    float delta[PIECE];
};

// Sums each of `values`' 32 elements over the warp and leaves lane l with
// element l's total (recursive halving: at mask m a lane keeps the half its
// bit m selects and adds its partner's copy). Every total combines the lanes
// in the order of `seismic_warp_sum_f32`, so the bits are the butterfly's.
__device__ __forceinline__ float reduce_scatter(float (&values)[32], int lane) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        const bool upper = (lane & mask) != 0;
#pragma unroll
        for (int m = 0; m < mask; ++m) {
            const float send = upper ? values[m] : values[m + mask];
            const float keep = upper ? values[m + mask] : values[m];
            values[m] = seismic_add_rn(keep, seismic_shfl_xor_f32(send, mask));
        }
    }
    return values[0];
}

// Advances this warp's `rows` over piece `piece` (the slot's `local`-th) in the
// state-space dual form and writes the piece's mixed outputs.
template <int LANE_ROWS, int BLOCK_ROWS>
__device__ __forceinline__ void advance_piece(const Operands &in, const float *inputs, const float *products,
                                              const versions::Slot &slot, int piece, int local, int head,
                                              int block_row0, int row0, Rows<LANE_ROWS> &rows,
                                              PieceShared<BLOCK_ROWS> &shared) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    // Rows of the piece whose S C sums one reduce-scatter finishes.
    constexpr int GROUP = 32 / LANE_ROWS;
    const int lane = threadIdx.x % 32;
    const int g = group_of(head);
    const int first = slot.lo + local * PIECE;
    const int count = min(PIECE, slot.stop - local * PIECE);
    for (int item = threadIdx.x; item < count * BLOCK_ROWS; item += blockDim.x) {
        const int i = item / BLOCK_ROWS;
        shared.x[i][item % BLOCK_ROWS] =
            convolve(in, slot, first + i - slot.lo, head * P + block_row0 + item % BLOCK_ROWS);
    }
    for (int i = threadIdx.x; i < count; i += blockDim.x) shared.delta[i] = staged(inputs, first + i)[INPUT_DELTA + head];
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < count; ++i) {
            sum = seismic_add_rn(sum, staged(inputs, first + i)[INPUT_LOG_DECAY + head]);
            shared.log_gamma[i] = sum;
        }
    }
    __syncthreads();
    const float *piece_products = products + (static_cast<u64>(piece) * G + g) * PIECE * PIECE;
    for (int item = threadIdx.x; item < count * count; item += blockDim.x) {
        const int i = item / count;
        const int j = item % count;
        shared.weights[i][j] = j <= i ? seismic_mul_rn(expf(shared.log_gamma[i] - shared.log_gamma[j]),
                                                       piece_products[i * PIECE + j])
                                      : 0.0f;
    }
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        shared.gamma[i] = expf(shared.log_gamma[i]);
        shared.tail[i] = expf(shared.log_gamma[count - 1] - shared.log_gamma[i]);
    }
    __syncthreads();

    const int local_row = row0 - block_row0;
    const float skip = in.skip[head * SEISMIC_SKIP_STRIDE_0];
    // Lane l finishes row first + i0 + l / LANE_ROWS, state row l % LANE_ROWS.
    const int r = lane % LANE_ROWS;
    for (int i0 = 0; i0 < count; i0 += GROUP) {
        float sums[32];
#pragma unroll
        for (int k = 0; k < GROUP; ++k) {
            const int i = min(i0 + k, count - 1);
            const float *c = staged(inputs, first + i) + INPUT_C + g * N + lane * COLUMNS;
            float cs[COLUMNS];
#pragma unroll
            for (int j = 0; j < COLUMNS; ++j) cs[j] = c[j];
#pragma unroll
            for (int s = 0; s < LANE_ROWS; ++s) {
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < COLUMNS; ++j) sum = seismic_fma_rn(rows.values[s][j], cs[j], sum);
                sums[k * LANE_ROWS + s] = sum;
            }
        }
        const float inter = reduce_scatter(sums, lane);
        const int i = i0 + lane / LANE_ROWS;
        if (i < count) {
            float sum = 0.0f;
            for (int j = 0; j <= i; ++j)
                sum = seismic_fma_rn(shared.weights[i][j], seismic_mul_rn(shared.delta[j], shared.x[j][local_row + r]),
                                     sum);
            const float value = shared.x[i][local_row + r];
            const float output = seismic_fma_rn(skip, value, seismic_fma_rn(shared.gamma[i], inter, sum));
            element::put<Act>(in.mixed,
                              static_cast<u64>(first + i) * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1 +
                                  (row0 + r) * SEISMIC_RESULT_0_STRIDE_2,
                              output);
        }
    }
    // S <- gamma_last S + sum_j (gamma_last / gamma_j) u_j B_j^T.
    const float last = shared.gamma[count - 1];
#pragma unroll
    for (int s = 0; s < LANE_ROWS; ++s)
#pragma unroll
        for (int j = 0; j < COLUMNS; ++j) rows.values[s][j] = seismic_mul_rn(rows.values[s][j], last);
    for (int j = 0; j < count; ++j) {
        const float *b = staged(inputs, first + j) + INPUT_B + g * N + lane * COLUMNS;
        float bs[COLUMNS];
#pragma unroll
        for (int c = 0; c < COLUMNS; ++c) bs[c] = b[c];
#pragma unroll
        for (int s = 0; s < LANE_ROWS; ++s) {
            const float coefficient =
                seismic_mul_rn(shared.tail[j], seismic_mul_rn(shared.delta[j], shared.x[j][local_row + s]));
#pragma unroll
            for (int c = 0; c < COLUMNS; ++c) rows.values[s][c] = seismic_fma_rn(coefficient, bs[c], rows.values[s][c]);
        }
    }
    __syncthreads();
}

} // namespace state_space

template <unsigned ROWS>
__global__ void state_space_chunk_scan(SEISMIC_KERNEL_PARAMS) {
    using namespace state_space;
    constexpr int LANE_ROWS = ROWS / 4;
    // The sequential advance's span (as the step's).
    constexpr int SPAN = 2048 / N;
    static_assert(P % ROWS == 0, "state rows split evenly");
    const Operands in = STATE_SPACE_OPERANDS();
    const float *inputs = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_INPUTS));
    const float *products = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PRODUCTS));
    const int head = blockIdx.x;
    const int block_row0 = blockIdx.y * ROWS;
    const int row0 = block_row0 + (threadIdx.x / 32) * LANE_ROWS;
    if (blockIdx.z == SEISMIC_DIM_B) {
        zero_uncovered<LANE_ROWS>(in, head, row0);
        return;
    }
    const versions::Slot slot = versions::slot_of(in.slots, blockIdx.z, seismic_words_value);
    Rows<LANE_ROWS> rows;
    load_version<LANE_ROWS>(in, slot, head, row0, threadIdx.x % 32, rows);
    publish_window(in, slot, head, blockIdx.y, gridDim.y);
    // The piece and sequential stagings are used in turn.
    __shared__ __align__(16) union {
        PieceShared<ROWS> piece;
        SequentialShared<ROWS, SPAN> sequential;
    } shared;
    const int pieces = pieces_of(slot);
    const int base = first_piece(in, blockIdx.z);
    for (int piece = 0; piece < pieces; ++piece)
        advance_piece<LANE_ROWS, ROWS>(in, inputs, products, slot, base + piece, piece, head, block_row0, row0, rows,
                                       shared.piece);
    advance_rows<LANE_ROWS, ROWS, SPAN>(in, slot, pieces == 0 ? slot.lo : slot.lo + slot.stop, head, block_row0, row0,
                                        rows, shared.sequential);
}
#endif
