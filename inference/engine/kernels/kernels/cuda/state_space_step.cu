// state_space_step (decode, speculative verify and small row classes): the
// row-sequential state-space rule with its preparation fused
// (`state_space::advance_rows`).
//
// One block of four warps per (head, block of ROWS state rows, slot); slots
// are a grid axis. Each warp owns ROWS / 4 state rows, a lane N / 32 columns
// of each, held in registers for the whole slot. The state is read from
// version (previous_bank, previous_tape)[slot] and published to
// following_bank[slot] after the slot's first stop[slot] rows, with the window
// and the tape of the rows after it. Grid z = B zeroes the mixed rows no slot
// covers. ROWS never changes result bits, and `state_space_chunk` gives its
// short slots the same bits.

#include "lib/recurrent/state_space.cuh"

#ifdef SEISMIC_FORMING_STATE_SPACE_STEP
template <unsigned ROWS>
__global__ void state_space_step(SEISMIC_KERNEL_PARAMS) {
    constexpr int LANE_ROWS = ROWS / 4;
    // Rows whose inputs are staged at once: 8 KiB each of B and C.
    constexpr int SPAN = 2048 / state_space::N;
    static_assert(state_space::P % ROWS == 0, "state rows split evenly");
    const state_space::Operands in = STATE_SPACE_OPERANDS();
    const int head = blockIdx.x;
    const int block_row0 = blockIdx.y * ROWS;
    const int row0 = block_row0 + (threadIdx.x / 32) * LANE_ROWS;
    if (blockIdx.z == SEISMIC_DIM_B) {
        state_space::zero_uncovered<LANE_ROWS>(in, head, row0);
        return;
    }
    const versions::Slot slot = versions::slot_of(in.slots, blockIdx.z, seismic_words_value);
    state_space::Rows<LANE_ROWS> rows;
    state_space::load_version<LANE_ROWS>(in, slot, head, row0, threadIdx.x % 32, rows);
    state_space::publish_window(in, slot, head, blockIdx.y, gridDim.y);
    __shared__ __align__(16) state_space::SequentialShared<ROWS, SPAN> shared;
    state_space::advance_rows<LANE_ROWS, ROWS, SPAN>(in, slot, slot.lo, head, block_row0, row0, rows, shared);
}
#endif
