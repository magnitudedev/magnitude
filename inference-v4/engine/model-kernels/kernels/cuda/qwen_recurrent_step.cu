// qwen_recurrent_step (decode and small row classes): the row-sequential
// gated delta rule with its preparation fused.
//
// One block per (value head, state-row block, slot); slots are a grid axis.
// Each warp owns ROWS state rows (value coordinates), a lane W / 32 columns
// (key coordinates) of each, held in registers for the whole slot. Per row
// the block convolves (causal depthwise conv + SiLU) the q and k channels of
// the head's key head and the v channels of its state rows; each warp then
// L2-normalizes q and k, derives beta and the decay, and advances its rows:
// S <- decay S + beta (v - decay S k) k^T, output S q. The state is read from
// bank previous_bank[slot] and published to following_bank[slot] after the
// slot's first stop[slot] rows; the window likewise. Grid z = B zeroes the
// mixed rows no slot covers. ROWS and WARPS never change result bits.

#include "common/recurrent.cuh"

namespace {

using rec::C;
using rec::NK;
using rec::NV;
using rec::W;
using mx::u64;

constexpr int ROWS = SEISMIC_TUNE_ROWS;
constexpr int WARPS = SEISMIC_TUNE_WARPS;
constexpr int BLOCK_ROWS = ROWS * WARPS;
constexpr int CPL = W / 32;  // state columns per lane
static_assert(W % 32 == 0 && W % BLOCK_ROWS == 0, "state rows split evenly");

}  // namespace

extern "C" __global__ void __launch_bounds__(WARPS * 32) qwen_recurrent_step(SEISMIC_KERNEL_PARAMS) {
    const rec::Inputs in = REC_INPUTS();
    mx::u8 *mixed = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    const int head = blockIdx.x;
    const int block = blockIdx.y;
    const u64 slot_index = blockIdx.z;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int first_row = block * BLOCK_ROWS + warp * ROWS;  // this warp's state rows
    auto mixed_at = [&](int row, int state_row) {
        return static_cast<u64>(row) * SEISMIC_RESULT_0_STRIDE_0 +
               static_cast<u64>(head) * SEISMIC_RESULT_0_STRIDE_1 +
               static_cast<u64>(state_row) * SEISMIC_RESULT_0_STRIDE_2;
    };

    if (slot_index == SEISMIC_DIM_B) {
        const int covered = rec::covered_end(in);
        for (u64 row = covered; row < SEISMIC_DIM_M; ++row)
            if (lane < ROWS) mx::act_store(mixed, mixed_at(row, first_row + lane), 0.0f);
        return;
    }

    const rec::Slot slot = rec::slot_of(in, slot_index);
    rec::publish_window(in, slot, head * gridDim.y + block, NV * gridDim.y);

    const int key_head = rec::key_head(in, head);
    float state[ROWS][CPL];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
        mx::f32_span(rec::state_row(in, slot.source, head, first_row + r) + lane * CPL, state[r]);
    auto publish = [&]() {
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
            mx::f32_span_store(rec::state_row(in, slot.target, head, first_row + r) + lane * CPL,
                               state[r]);
    };
    if (slot.stop == 0) publish();

    // Convolved rows, double-buffered by row parity: q | k of the key head,
    // then the block's v channels.
    __shared__ __align__(16) float prepared[2][2 * W + BLOCK_ROWS];
    const float root = rsqrtf(static_cast<float>(W));
    for (int row = slot.lo; row < slot.hi; ++row) {
        float *buffer = prepared[row & 1];
        for (int index = threadIdx.x; index < 2 * W + BLOCK_ROWS; index += WARPS * 32) {
            const int channel = index < W       ? key_head * W + index
                                : index < 2 * W ? (NK + key_head) * W + index - W
                                                : (2 * NK + head) * W + block * BLOCK_ROWS +
                                                      index - 2 * W;
            buffer[index] = rec::convolved(in, slot, row, channel);
        }
        __syncthreads();

        float q[CPL], k[CPL];
        mx::f32_span(buffer + lane * CPL, q);
        mx::f32_span(buffer + W + lane * CPL, k);
        float q_squares = 0.0f, k_squares = 0.0f;
#pragma unroll
        for (int c = 0; c < CPL; ++c) {
            q_squares = __fmaf_rn(q[c], q[c], q_squares);
            k_squares = __fmaf_rn(k[c], k[c], k_squares);
        }
        const float q_inverse = rsqrtf(seismic_warp_sum_f32(q_squares) + in.epsilon) * root;
        const float k_inverse = rsqrtf(seismic_warp_sum_f32(k_squares) + in.epsilon);
#pragma unroll
        for (int c = 0; c < CPL; ++c) {
            q[c] *= q_inverse;
            k[c] *= k_inverse;
        }
        const rec::Gates gates = rec::gates(in, row, head);

        float remembered[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < CPL; ++c) sum = __fmaf_rn(state[r][c] * gates.decay, k[c], sum);
            remembered[r] = sum;
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) remembered[r] = seismic_warp_sum_f32(remembered[r]);
        float output[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const float v = buffer[2 * W + warp * ROWS + r];
            const float residual = (v - remembered[r]) * gates.beta;
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < CPL; ++c) {
                state[r][c] = __fmaf_rn(residual, k[c], state[r][c] * gates.decay);
                sum = __fmaf_rn(state[r][c], q[c], sum);
            }
            output[r] = sum;
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) output[r] = seismic_warp_sum_f32(output[r]);
        if (lane < ROWS) {
            float value = output[0];
#pragma unroll
            for (int r = 1; r < ROWS; ++r)
                if (lane == r) value = output[r];
            mx::act_store(mixed, mixed_at(row, first_row + lane), value);
        }
        if (row + 1 == slot.lo + slot.stop) publish();
    }
}
