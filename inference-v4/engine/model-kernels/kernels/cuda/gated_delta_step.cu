// gated_delta_step (decode, MTP verify and small row classes): the
// row-sequential gated delta rule with its preparation fused
// (`recurrent::advance_rows`).
//
// One block per (value head, state-row block, slot); slots are a grid axis.
// Each warp owns ROWS state rows (value coordinates), a lane W / 32 columns
// (key coordinates) of each, held in registers for the whole slot. The state
// is read from version (previous_bank, previous_tape)[slot] and published to
// following_bank[slot] after the slot's first stop[slot] rows, with the window
// and the tape of the rows after it. Grid z = B
// zeroes the mixed rows no slot covers. ROWS and WARPS never change result
// bits, and `gated_delta_chunk` gives its short slots the same bits.

#include "lib/recurrent/recurrent.cuh"

#ifdef SEISMIC_FORMING_GATED_DELTA_STEP
template <unsigned ROWS, unsigned WARPS>
__global__ void gated_delta_step(SEISMIC_KERNEL_PARAMS) {
    using recurrent::Act;
    using recurrent::u64;
    constexpr int BLOCK_ROWS = ROWS * WARPS;
    static_assert(recurrent::W % 32 == 0 && recurrent::W % BLOCK_ROWS == 0,
                  "state rows split evenly");
    const recurrent::Inputs in = RECURRENT_INPUTS();
    recurrent::u8 *mixed = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    const int head = blockIdx.x;
    const int block_row = blockIdx.y * BLOCK_ROWS;
    const u64 slot_index = blockIdx.z;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int first_row = block_row + warp * ROWS;  // this warp's state rows

    if (slot_index == SEISMIC_DIM_B) {
        const int covered = recurrent::covered_end(in);
        for (u64 row = covered; row < SEISMIC_DIM_M; ++row)
            if (lane < ROWS)
                element::put<Act>(mixed,
                                  row * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1 +
                                      (first_row + lane) * SEISMIC_RESULT_0_STRIDE_2,
                                  0.0f);
        return;
    }

    const recurrent::Slot slot = recurrent::slot_of(in, slot_index);
    recurrent::WarpRows<ROWS> state;
    recurrent::load_version<ROWS>(in, slot, head, first_row, state);
    recurrent::publish_window(in, slot, head * gridDim.y + blockIdx.y, recurrent::NV * gridDim.y);
    __shared__ __align__(16) recurrent::SequentialShared<BLOCK_ROWS> shared;
    recurrent::advance_rows<ROWS, BLOCK_ROWS>(in, slot, slot.lo, head, block_row, true, first_row, state, mixed,
                                              shared);
}
#endif
