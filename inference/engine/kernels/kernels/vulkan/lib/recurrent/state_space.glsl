// Shared device code of the Vulkan state-space (Mamba-2) entries
// (`state_space_step`, `state_space_chunk`; contracts and the portable
// `state_space_rows` in state_space.seismic): operand addressing, the causal
// convolution with its bias and SiLU, the step size and decay, the successor
// window publication, the version load (tape replay), the chunk's pieces, and
// the row-sequential advance both entries run: all of the step, and in the
// chunk the slots of at most STATE_SPACE_SEQUENTIAL_ROWS rows and the rows
// after a stop row. The counterpart of `cuda/lib/recurrent/state_space.cuh`,
// whose arithmetic it follows. A workgroup is four logical subgroups; each
// holds SS_LANE_ROWS = ROWS / 4 state rows, SS_COLUMNS columns per lane.
#include "../core/activation.glsl"
#include "versions.glsl"

#define SS_NH uint(SEISMIC_DIM_NH)
#define SS_P uint(SEISMIC_DIM_P)
#define SS_G uint(SEISMIC_DIM_G)
#define SS_N uint(SEISMIC_DIM_N)
#define SS_TAPS int(SEISMIC_DIM_C)
// Convolved channels x | B | C, and their first projection column.
#define SS_CH (SS_NH * SS_P + 2u * SS_G * SS_N)
#define SS_X_COLUMN (SS_NH * SS_P)
#define SS_DT_COLUMN (2u * SS_NH * SS_P + 2u * SS_G * SS_N)
// Offsets of the group's B and C among the convolved channels.
#define SS_B_CHANNEL (SS_NH * SS_P)
#define SS_C_CHANNEL (SS_NH * SS_P + SS_G * SS_N)
// A tape row: the inputs u [NH, P], the convolved B [G, N], the decays d [NH].
#define SS_TAPE_U 0u
#define SS_TAPE_B (SS_NH * SS_P)
#define SS_TAPE_D (SS_NH * SS_P + SS_G * SS_N)
// State columns per lane, state rows per workgroup and per subgroup.
#define SS_COLUMNS (uint(SEISMIC_DIM_N) / 32u)
#define SS_BLOCK_ROWS uint(SEISMIC_TUNE_ROWS)
#define SS_LANE_ROWS (uint(SEISMIC_TUNE_ROWS) / 4u)
// Floats of one row of the chunk's `inputs` scratch: the convolved B and C
// channels, then each head's step size, then its log decay.
#define SS_INPUT_B 0u
#define SS_INPUT_C (SS_G * SS_N)
#define SS_INPUT_DELTA (2u * SS_G * SS_N)
#define SS_INPUT_LOG_DECAY (2u * SS_G * SS_N + SS_NH)
#define SS_INPUT_WIDTH (2u * SS_G * SS_N + 2u * SS_NH)
// Rows whose inputs the sequential advance stages at once: 8 KiB each of B
// and C.
#define SS_SPAN (2048u / SS_N)
// Its shared floats: each row's B then C, x rows, step sizes, decays.
#define SS_SEQUENTIAL_BC 0u
#define SS_SEQUENTIAL_X (2u * SS_SPAN * SS_N)
#define SS_SEQUENTIAL_DELTA (SS_SEQUENTIAL_X + SS_SPAN * SS_BLOCK_ROWS)
#define SS_SEQUENTIAL_DECAY (SS_SEQUENTIAL_DELTA + SS_SPAN)

// Slots of at most this many rows advance row-sequentially in either entry,
// so a request's state bits never depend on its row class or its peers.
#define STATE_SPACE_SEQUENTIAL_ROWS 16
// The chunk's pieces: a slot of more than STATE_SPACE_SEQUENTIAL_ROWS rows
// splits its rows before the stop row into pieces of at most
// STATE_SPACE_PIECE rows from its first row; other slots have none. Pieces
// are numbered over the slots in order.
#define STATE_SPACE_PIECE 32

// The group whose B and C head `head` reads.
uint ss_group_of(uint head) {
    return head * SS_G / SS_NH;
}

// Whether `head` is its group's first head: the one that publishes and
// records the group's B and C.
bool ss_leads_group(uint head) {
    return head == 0u || ss_group_of(head - 1u) != ss_group_of(head);
}

// Byte address of tape row `entry` of `bank`.
uint64_t ss_tape_row(int bank, int entry) {
    return versions_bank(SEISMIC_PTR(SEISMIC_BUFFER_TAPE), bank, SEISMIC_TAPE_STRIDE_0 * 4ul)
        + uint64_t(entry) * SEISMIC_TAPE_STRIDE_1 * 4ul;
}

// Byte address of row `row` of head `head`'s state in `bank`: N floats.
uint64_t ss_state_row(int bank, uint head, uint row) {
    return versions_bank(SEISMIC_PTR(SEISMIC_BUFFER_STATE), bank, SEISMIC_STATE_STRIDE_0 * 4ul)
        + (uint64_t(head) * SEISMIC_STATE_STRIDE_1 + uint64_t(row) * SEISMIC_STATE_STRIDE_2) * 4ul;
}

uint64_t ss_window_row(int bank, int tap) {
    const uint64_t element_bytes = ELEMENT_BYTES(ELEMENT_ACT);
    return versions_bank(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW), bank, SEISMIC_WINDOW_STRIDE_0 * element_bytes)
        + uint64_t(tap) * SEISMIC_WINDOW_STRIDE_1 * element_bytes;
}

// The raw convolution input of `channel` at slot-local `position`: the
// source version's window rows before the slot, the projection after.
float ss_raw(versions_slot slot, int position, uint channel) {
    if (position < 0)
        return element_at(ELEMENT_ACT, ss_window_row(slot.source, slot.taped + SS_TAPS - 1 + position), uint64_t(channel));
    return element_at(ELEMENT_ACT, SEISMIC_PTR(SEISMIC_BUFFER_PROJECTION),
        uint64_t(slot.lo + position) * SEISMIC_PROJECTION_STRIDE_0 + SS_X_COLUMN + channel);
}

// SiLU of the causal depthwise convolution of `channel` at slot-local row
// `local` plus the channel's bias, in the body's order: the current row's
// tap, then taps 0..C - 1 fused in turn, then the bias.
float ss_convolve(versions_slot slot, int local, uint channel) {
    const uint64_t weights = SEISMIC_PTR(SEISMIC_BUFFER_CONVOLUTION) + uint64_t(channel) * SEISMIC_CONVOLUTION_STRIDE_0 * 4ul;
    float sum = seismic_mul_rn(element_f32_at(weights + uint64_t(SS_TAPS - 1) * SEISMIC_CONVOLUTION_STRIDE_1 * 4ul),
        ss_raw(slot, local, channel));
    [[unroll]] for (int tap = 0; tap + 1 < SS_TAPS; ++tap)
        sum = seismic_fma_rn(element_f32_at(weights + uint64_t(tap) * SEISMIC_CONVOLUTION_STRIDE_1 * 4ul),
            ss_raw(slot, local + tap - (SS_TAPS - 1), channel), sum);
    sum = seismic_add_rn(sum, element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_CONVOLUTION_BIAS)
        + uint64_t(channel) * SEISMIC_CONVOLUTION_BIAS_STRIDE_0 * 4ul));
    return seismic_div_rn(sum, 1.0 + exp(-sum));
}

// The step size delta = softplus(dt + time_bias) and the log decay
// rate * delta of head `head` at `row`.
void ss_step_of(int row, uint head, out float delta, out float log_decay) {
    const float shifted = element_at(ELEMENT_ACT, SEISMIC_PTR(SEISMIC_BUFFER_PROJECTION),
            uint64_t(row) * SEISMIC_PROJECTION_STRIDE_0 + SS_DT_COLUMN + head)
        + element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_TIME_BIAS) + uint64_t(head) * SEISMIC_TIME_BIAS_STRIDE_0 * 4ul);
    delta = max(shifted, 0.0) + log(1.0 + exp(-abs(shifted)));
    log_decay = seismic_mul_rn(element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_RATE)
        + uint64_t(head) * SEISMIC_RATE_STRIDE_0 * 4ul), delta);
}

// Publishes head `head`'s share of the slot's successor window (the C - 1 raw
// rows before the publication row, then the raw rows of its tape): its x
// channels, and its group's B and C channels when it leads the group. `part`
// of `parts` cooperating workgroups copies an even share.
void ss_publish_window(versions_slot slot, uint head, uint part, uint parts) {
    const int taps = SS_TAPS - 1;
    const uint rows = uint(taps + versions_tape_rows(slot));
    const uint group = ss_group_of(head);
    const uint per_tap = SS_P + (ss_leads_group(head) ? 2u * SS_N : 0u);
    const uint total = rows * per_tap;
    const uint per = (total + parts - 1u) / parts;
    const uint last = min(total, (part + 1u) * per);
    for (uint item = part * per + gl_LocalInvocationIndex; item < last; item += gl_WorkGroupSize.x) {
        const int tap = int(item / per_tap);
        const uint offset = item % per_tap;
        const uint channel = offset < SS_P ? head * SS_P + offset
            : offset < SS_P + SS_N         ? SS_B_CHANNEL + group * SS_N + offset - SS_P
                                           : SS_C_CHANNEL + group * SS_N + offset - SS_P - SS_N;
        element_put(ELEMENT_ACT, ss_window_row(slot.target, tap), uint64_t(channel),
            ss_raw(slot, slot.stop + tap - taps, channel));
    }
}

void ss_store_rows(int bank, uint head, uint row0, float rows[SS_LANE_ROWS][SS_COLUMNS]) {
    const uint lane = SEISMIC_LANE;
    [[unroll]] for (uint r = 0u; r < SS_LANE_ROWS; ++r)
        [[unroll]] for (uint c = 0u; c < SS_COLUMNS; ++c)
            element_f32_put(ss_state_row(bank, head, row0 + r) + uint64_t(lane * SS_COLUMNS + c) * 4ul, rows[r][c]);
}

// The rows of the slot's source version: the bank's state advanced by its
// first `taped` tape rows with the step's update, so the bits equal a run
// that published after those rows.
void ss_load_version(versions_slot slot, uint head, uint row0, out float rows[SS_LANE_ROWS][SS_COLUMNS]) {
    const uint lane = SEISMIC_LANE;
    [[unroll]] for (uint r = 0u; r < SS_LANE_ROWS; ++r)
        [[unroll]] for (uint c = 0u; c < SS_COLUMNS; ++c)
            rows[r][c] = element_f32_at(ss_state_row(slot.source, head, row0 + r) + uint64_t(lane * SS_COLUMNS + c) * 4ul);
    const uint group = ss_group_of(head);
    for (int entry = 0; entry < slot.taped; ++entry) {
        const uint64_t tape = ss_tape_row(slot.source, entry);
        const float decay = element_f32_at(tape + uint64_t(SS_TAPE_D + head) * 4ul);
        float b[SS_COLUMNS];
        [[unroll]] for (uint c = 0u; c < SS_COLUMNS; ++c)
            b[c] = element_f32_at(tape + uint64_t(SS_TAPE_B + group * SS_N + lane * SS_COLUMNS + c) * 4ul);
        [[unroll]] for (uint r = 0u; r < SS_LANE_ROWS; ++r) {
            const float drive = element_f32_at(tape + uint64_t(SS_TAPE_U + head * SS_P + row0 + r) * 4ul);
            [[unroll]] for (uint c = 0u; c < SS_COLUMNS; ++c)
                rows[r][c] = seismic_fma_rn(drive, b[c], seismic_mul_rn(rows[r][c], decay));
        }
    }
}

// The row-sequential state-space rule over the slot's rows [begin, hi), the
// arithmetic of `state_space_step` (a workgroup's shape never changes bits).
// Every invocation of the workgroup calls it. The workgroup owns state rows
// [block_row0, block_row0 + SS_BLOCK_ROWS) of head `head`; each subgroup
// holds SS_LANE_ROWS of them from `row0` in `rows` (the state before row
// `begin`), advances them, writes their mixed outputs and publishes them
// after the slot's first `stop` rows (at the start when `begin` is the
// publication row). Rows after the stop row are recorded in the successor's
// tape. Per span of up to SS_SPAN rows the workgroup convolves the group's B
// and C and its x channels into shared memory with each row's step size and
// decay; after one barrier the rows advance in order: S <- decay S +
// (delta x) B^T, output S C + D x.
void ss_advance_rows(versions_slot slot, int begin, uint head, uint block_row0, uint row0,
    inout float rows[SS_LANE_ROWS][SS_COLUMNS]) {
    const uint lane = SEISMIC_LANE, thread = gl_LocalInvocationIndex, threads = gl_WorkGroupSize.x;
    const uint group = ss_group_of(head);
    const int publish = slot.lo + slot.stop;
    const int recorded = versions_tape_rows(slot);
    const float skip = element_f32_at(SEISMIC_PTR(SEISMIC_BUFFER_SKIP) + uint64_t(head) * SEISMIC_SKIP_STRIDE_0 * 4ul);
    const uint64_t mixed = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    // The subgroup that owns state row 0 records the decay, and B when the
    // head leads its group.
    const bool records = row0 == 0u;
    const bool records_b = records && ss_leads_group(head);
    const uint local_row = row0 - block_row0;
    if (begin == publish)
        ss_store_rows(slot.target, head, row0, rows);
    for (int first = begin; first < slot.hi; first += int(SS_SPAN)) {
        const uint count = uint(min(int(SS_SPAN), slot.hi - first));
        for (uint item = thread; item < count * 2u * SS_N; item += threads) {
            const uint i = item / (2u * SS_N);
            const uint column = item % (2u * SS_N);
            const uint channel = column < SS_N ? SS_B_CHANNEL + group * SS_N + column
                                               : SS_C_CHANNEL + group * SS_N + column - SS_N;
            seismic_shared_f32[SS_SEQUENTIAL_BC + i * 2u * SS_N + column] =
                ss_convolve(slot, first + int(i) - slot.lo, channel);
        }
        // The x channels go to the last invocations, off the B/C work.
        for (uint item = threads - 1u - thread; item < count * SS_BLOCK_ROWS; item += threads) {
            const uint i = item / SS_BLOCK_ROWS;
            seismic_shared_f32[SS_SEQUENTIAL_X + i * SS_BLOCK_ROWS + item % SS_BLOCK_ROWS] =
                ss_convolve(slot, first + int(i) - slot.lo, head * SS_P + block_row0 + item % SS_BLOCK_ROWS);
        }
        for (uint i = thread; i < count; i += threads) {
            float delta, log_decay;
            ss_step_of(first + int(i), head, delta, log_decay);
            seismic_shared_f32[SS_SEQUENTIAL_DELTA + i] = delta;
            seismic_shared_f32[SS_SEQUENTIAL_DECAY + i] = exp(log_decay);
        }
        barrier();
        for (uint i = 0u; i < count; ++i) {
            const int row = first + int(i);
            const float decay = seismic_shared_f32[SS_SEQUENTIAL_DECAY + i];
            const float delta = seismic_shared_f32[SS_SEQUENTIAL_DELTA + i];
            float b[SS_COLUMNS], c[SS_COLUMNS];
            [[unroll]] for (uint j = 0u; j < SS_COLUMNS; ++j) {
                b[j] = seismic_shared_f32[SS_SEQUENTIAL_BC + i * 2u * SS_N + lane * SS_COLUMNS + j];
                c[j] = seismic_shared_f32[SS_SEQUENTIAL_BC + i * 2u * SS_N + SS_N + lane * SS_COLUMNS + j];
            }
            const bool taping = row >= publish && row - publish < recorded;
            const uint64_t entry = taping ? ss_tape_row(slot.target, row - publish) : 0ul;
            if (taping && records_b)
                [[unroll]] for (uint j = 0u; j < SS_COLUMNS; ++j)
                    element_f32_put(entry + uint64_t(SS_TAPE_B + group * SS_N + lane * SS_COLUMNS + j) * 4ul, b[j]);
            if (taping && records && lane == 0u)
                element_f32_put(entry + uint64_t(SS_TAPE_D + head) * 4ul, decay);
            float mine = 0.0;
            [[unroll]] for (uint r = 0u; r < SS_LANE_ROWS; ++r) {
                const float value = seismic_shared_f32[SS_SEQUENTIAL_X + i * SS_BLOCK_ROWS + local_row + r];
                const float drive = seismic_mul_rn(delta, value);
                if (taping && lane == r)
                    element_f32_put(entry + uint64_t(SS_TAPE_U + head * SS_P + row0 + r) * 4ul, drive);
                float sum = 0.0;
                [[unroll]] for (uint j = 0u; j < SS_COLUMNS; ++j) {
                    rows[r][j] = seismic_fma_rn(drive, b[j], seismic_mul_rn(rows[r][j], decay));
                    sum = seismic_fma_rn(rows[r][j], c[j], sum);
                }
                const float result = seismic_fma_rn(skip, value, seismic_subgroup_sum_f32(sum));
                mine = lane == r ? result : mine;
            }
            if (lane < SS_LANE_ROWS)
                element_put(ELEMENT_ACT, mixed,
                    uint64_t(row) * SEISMIC_RESULT_0_STRIDE_0 + uint64_t(head) * SEISMIC_RESULT_0_STRIDE_1
                        + uint64_t(row0 + lane) * SEISMIC_RESULT_0_STRIDE_2,
                    mine);
            if (row + 1 == publish)
                ss_store_rows(slot.target, head, row0, rows);
        }
        barrier();
    }
}

// Zeroes this subgroup's state rows of head `head` in the mixed rows no slot
// covers.
void ss_zero_uncovered(uint head, uint row0) {
    const uint lane = SEISMIC_LANE;
    const uint64_t mixed = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    for (uint64_t row = uint64_t(versions_covered_end()); row < SEISMIC_DIM_M; ++row)
        if (lane < SS_LANE_ROWS)
            element_put(ELEMENT_ACT, mixed,
                row * SEISMIC_RESULT_0_STRIDE_0 + uint64_t(head) * SEISMIC_RESULT_0_STRIDE_1
                    + uint64_t(row0 + lane) * SEISMIC_RESULT_0_STRIDE_2,
                0.0);
}

// The pieces of a slot (see STATE_SPACE_PIECE).
int ss_pieces_of(versions_slot slot) {
    return slot.hi - slot.lo > STATE_SPACE_SEQUENTIAL_ROWS ? (slot.stop + STATE_SPACE_PIECE - 1) / STATE_SPACE_PIECE : 0;
}

// The number of the first piece of slot `index`.
int ss_first_piece(uint64_t index) {
    int total = 0;
    for (uint64_t slot = 0ul; slot < index; ++slot)
        total += ss_pieces_of(versions_slot_of(slot));
    return total;
}

// Piece `piece`'s slot and its piece within the slot; false past the last
// piece.
bool ss_piece_at(int piece, out versions_slot slot, out int local) {
    local = 0;
    for (uint64_t index = 0ul; index < SEISMIC_DIM_B; ++index) {
        slot = versions_slot_of(index);
        const int count = ss_pieces_of(slot);
        if (piece < count) {
            local = piece;
            return true;
        }
        piece -= count;
    }
    return false;
}
