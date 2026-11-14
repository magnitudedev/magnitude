// Device code of the Vulkan chunked delta-rule entries (`gated_delta_chunk`,
// `kimi_delta_chunk`; contracts in recurrent.seismic), over the rule's
// library (`recurrent.glsl`, or `kimi.glsl` for Kimi's gates), which the
// entry includes first. Two launches share the `inputs` scratch (f32, the
// chunked rows only), then the gate. The chunked rows of a slot are those
// before its stop row in a slot of more than CHUNK_SEQUENTIAL_ROWS rows; a
// shorter slot (an MTP verify) and a slot with a zero stop advance
// row-sequentially with the step's bits (`recurrent_sequential`), and the
// rows after a stop row advance row-sequentially from the published state,
// recording the tape. The scratch:
//   rows    [NK, M, 2W]   per key head: L2-normalized q (scaled by W^-1/2), k
//   values  [NV, M, W]    convolved v heads
//   gates   [NV, M * D + M]  per value head: decay factors [M, D], then beta
//                            [M] (D = RECURRENT_DECAYS)
//
// `chunk_inputs`: one subgroup per (row, channel head) convolves the row once
// with the step's arithmetic (so q, k, v, beta and the decay factors carry
// the step's bits), W / 32 channels per lane.
//
// `chunk_scan`: the workgroup geometry of the step, (value head, block of
// ROWS * WARPS state rows, slot), with the state transposed onto the lanes:
// each subgroup's ROWS state rows sit on groups of CHUNK_LANES = 8 adjacent
// lanes (groups past ROWS idle), lane j of a group holding columns j,
// j + CHUNK_LANES, ... of its row. A row's dot products are then a lane's
// short FMA chain and a butterfly over its group, and the rows' chains run in
// parallel instead of one reduction per state row. Per piece of
// RECURRENT_SPAN rows the workgroup stages the piece's q, k, v, decay factors
// and betas in the shared region, the next piece's loads in flight while the
// piece advances; a lane scales each of its columns by that column's factor.
// The raw outputs go to the `mixed` scratch. ROWS and WARPS never change
// result bits.
//
// The gate launch is `recurrent_gate(false)` (the staging launch's order).
#include "../core/reduce.glsl"

#if SEISMIC_TUNE_ROWS > RECURRENT_MAX_ROWS
#error "the chunked entries hold at most RECURRENT_MAX_ROWS state rows per subgroup"
#endif

#define CHUNK_SEQUENTIAL_ROWS int(SEISMIC_CONST_SEQUENTIAL_ROWS)
#define CHUNK_ROWS uint(SEISMIC_TUNE_ROWS)
#define CHUNK_WARPS uint(SEISMIC_TUNE_WARPS)
#define CHUNK_BLOCK_ROWS (CHUNK_ROWS * CHUNK_WARPS)
#define CHUNK_THREADS (32u * CHUNK_WARPS)
// Lanes per state row, fixed so that ROWS never changes a row's summation
// order: groups of lanes past ROWS idle.
#define CHUNK_LANES 8u
#define CHUNK_COLUMNS (RECURRENT_W / CHUNK_LANES)
#define CHUNK_MAX_COLUMNS CHUNK_COLUMNS
// A piece's staged q, k and v rows, in float4 items, and the items each
// invocation carries between pieces.
#define CHUNK_PREPARED (2u * RECURRENT_W + CHUNK_BLOCK_ROWS)
#define CHUNK_ITEMS (RECURRENT_SPAN * CHUNK_PREPARED / 4u)
#define CHUNK_ITEMS_PER_THREAD ((CHUNK_ITEMS + CHUNK_THREADS - 1u) / CHUNK_THREADS)
// A piece's staged gates, decay factors [SPAN][DECAYS] then betas [SPAN],
// and the floats each invocation carries between pieces.
#define CHUNK_GATES (RECURRENT_SPAN * (RECURRENT_DECAYS + 1u))
#define CHUNK_GATES_PER_THREAD ((CHUNK_GATES + CHUNK_THREADS - 1u) / CHUNK_THREADS)

// Byte addresses of the scratch regions.
uint64_t chunk_rows_at(uint64_t inputs, uint key_head, uint64_t row) {
    return inputs + (uint64_t(key_head) * SEISMIC_DIM_M + row) * 2ul * RECURRENT_W * 4ul;
}

uint64_t chunk_values_at(uint64_t inputs, uint value_head, uint64_t row) {
    return inputs
        + (uint64_t(2u * RECURRENT_NK) * SEISMIC_DIM_M * RECURRENT_W
              + (uint64_t(value_head) * SEISMIC_DIM_M + row) * RECURRENT_W)
            * 4ul;
}

// A value head's gates: its rows' decay factors, then their betas.
uint64_t chunk_gates_at(uint64_t inputs, uint value_head) {
    return inputs
        + (uint64_t(2u * RECURRENT_NK + RECURRENT_NV) * SEISMIC_DIM_M * RECURRENT_W
              + uint64_t(value_head) * SEISMIC_DIM_M * (RECURRENT_DECAYS + 1u))
            * 4ul;
}

uint64_t chunk_betas_at(uint64_t inputs, uint value_head, uint64_t row) {
    return chunk_gates_at(inputs, value_head) + (SEISMIC_DIM_M * RECURRENT_DECAYS + row) * 4ul;
}

// The row's RECURRENT_DECAYS decay factors.
uint64_t chunk_decays_at(uint64_t inputs, uint value_head, uint64_t row) {
    return chunk_gates_at(inputs, value_head) + row * RECURRENT_DECAYS * 4ul;
}

bool chunk_long(recurrent_slot slot) { return slot.hi - slot.lo > CHUNK_SEQUENTIAL_ROWS && slot.stop > 0; }

// The slot containing `row`, or false when the row is not chunked: outside
// every slot, in a slot the scan advances row-sequentially, or after its
// slot's stop row.
bool chunk_row_slot(uint64_t row, out recurrent_slot slot) {
    for (uint64_t candidate = 0ul; candidate < SEISMIC_DIM_B; ++candidate) {
        const recurrent_slot s = recurrent_slot_of(candidate);
        if (int(row) >= s.lo && int(row) < s.hi) {
            slot = s;
            return chunk_long(s) && int(row) < s.lo + s.stop;
        }
    }
    return false;
}

// The inputs launch: one subgroup per (row, channel head) convolves the row
// once with the step's arithmetic, W / 32 channels per lane.
void chunk_inputs() {
    const recurrent_inputs in_ = recurrent_inputs_of();
    const uint64_t inputs = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_INPUTS);
    const uint W = RECURRENT_W, C = RECURRENT_C, CPL = RECURRENT_CPL;
    const uint channel_head = gl_WorkGroupID.y * SEISMIC_SUBGROUPS + SEISMIC_SUBGROUP;
    const uint64_t row = gl_WorkGroupID.x;
    recurrent_slot slot;
    if (channel_head >= 2u * RECURRENT_NK + RECURRENT_NV || !chunk_row_slot(row, slot))
        return;
    const uint lane = SEISMIC_LANE;
    const int position = int(row) - slot.lo;
    float values[RECURRENT_CPL];
    float squares = 0.0;
    [[unroll]] for (uint c = 0u; c < CPL; ++c) {
        const uint channel = channel_head * W + lane * CPL + c;
        float weights[RECURRENT_MAX_C];
        [[unroll]] for (uint tap = 0u; tap < RECURRENT_MAX_C; ++tap)
            if (tap < C)
                weights[tap] = element_f32_at(in_.convolution
                    + (uint64_t(channel) * SEISMIC_CONVOLUTION_STRIDE_0 + uint64_t(tap) * SEISMIC_CONVOLUTION_STRIDE_1) * 4ul);
        float sum = weights[C - 1u] * recurrent_raw_input(in_, slot, position, channel);
        [[unroll]] for (uint tap = 0u; tap + 1u < RECURRENT_MAX_C; ++tap)
            if (tap + 1u < C)
                sum = seismic_fma_rn(weights[tap], recurrent_raw_input(in_, slot, position + int(tap) - (int(C) - 1), channel), sum);
        values[c] = seismic_div_rn(sum, 1.0 + exp(-sum));
        squares = seismic_fma_rn(values[c], values[c], squares);
    }
    if (channel_head < 2u * RECURRENT_NK) {
        const bool key = channel_head >= RECURRENT_NK;
        const float inverse = key ? inversesqrt(seismic_subgroup_sum_f32(squares) + in_.epsilon)
                                  : inversesqrt(seismic_subgroup_sum_f32(squares) + in_.epsilon) * inversesqrt(float(W));
        const uint64_t destination = chunk_rows_at(inputs, channel_head % RECURRENT_NK, row) + (key ? W : 0u) * 4ul;
        [[unroll]] for (uint c = 0u; c < CPL; ++c)
            element_f32_put(destination + uint64_t(lane * CPL + c) * 4ul, values[c] * inverse);
    } else {
        const uint head = channel_head - 2u * RECURRENT_NK;
        const uint64_t destination = chunk_values_at(inputs, head, row);
        [[unroll]] for (uint c = 0u; c < CPL; ++c)
            element_f32_put(destination + uint64_t(lane * CPL + c) * 4ul, values[c]);
        for (uint channel = lane; channel < RECURRENT_DECAYS; channel += 32u) {
            float beta, decay;
            recurrent_row_gates(in_, row, head, channel, beta, decay);
            element_f32_put(chunk_decays_at(inputs, head, row) + uint64_t(channel) * 4ul, decay);
            if (channel == 0u)
                element_f32_put(chunk_betas_at(inputs, head, row), beta);
        }
    }
}

// This lane's state row (block-local) and column part.
uint chunk_row() { return SEISMIC_SUBGROUP * CHUNK_ROWS + SEISMIC_LANE / CHUNK_LANES; }
uint chunk_part() { return SEISMIC_LANE % CHUNK_LANES; }
// The lane's group holds one of the subgroup's ROWS state rows.
bool chunk_active() { return SEISMIC_LANE / CHUNK_LANES < CHUNK_ROWS; }

// The lane's columns of its state row in the slot's source version (the
// bank's state advanced by its first `taped` tape rows with the step's
// update).
void chunk_load_version(recurrent_inputs in_, recurrent_slot slot, uint head, uint state_row,
    out float state[CHUNK_MAX_COLUMNS]) {
    [[unroll]] for (uint c = 0u; c < CHUNK_COLUMNS; ++c)
        state[c] = 0.0;
    if (!chunk_active())
        return;
    const uint part = chunk_part();
    const uint64_t source = recurrent_state_row(in_, slot.source, head, state_row);
    [[unroll]] for (uint c = 0u; c < CHUNK_COLUMNS; ++c)
        state[c] = element_f32_at(source + uint64_t(part + CHUNK_LANES * c) * 4ul);
    const uint key = recurrent_key_head(in_, head);
    for (int entry = 0; entry < slot.taped; ++entry) {
        const uint64_t tape = recurrent_tape_row(in_, slot.source, entry);
        const float u = element_f32_at(tape + uint64_t(RECURRENT_TAPE_U + head * RECURRENT_W + state_row) * 4ul);
        // One decay per value head is loaded once.
        float factor[CHUNK_MAX_COLUMNS];
        [[unroll]] for (uint c = 0u; c < CHUNK_COLUMNS; ++c)
            factor[c] = RECURRENT_DECAYS == 1u && c > 0u ? factor[0]
                : element_f32_at(recurrent_tape_factor(tape, head, part + CHUNK_LANES * c));
        [[unroll]] for (uint c = 0u; c < CHUNK_COLUMNS; ++c) {
            const uint column = part + CHUNK_LANES * c;
            const float k = element_f32_at(tape + uint64_t(RECURRENT_TAPE_K + key * RECURRENT_W + column) * 4ul);
            state[c] = seismic_fma_rn(u, k, state[c] * factor[c]);
        }
    }
}

void chunk_store(recurrent_inputs in_, int bank, uint head, uint state_row, float state[CHUNK_MAX_COLUMNS]) {
    if (!chunk_active())
        return;
    const uint part = chunk_part();
    const uint64_t target = recurrent_state_row(in_, bank, head, state_row);
    [[unroll]] for (uint c = 0u; c < CHUNK_COLUMNS; ++c)
        element_f32_put(target + uint64_t(part + CHUNK_LANES * c) * 4ul, state[c]);
}

// Shared floats of a staged piece: row r's q at r * CHUNK_PREPARED, its k at
// + W, the block's v at + 2W; then the decay factors [SPAN][DECAYS] and beta
// [SPAN].
#define CHUNK_DECAY (RECURRENT_SPAN * CHUNK_PREPARED)
#define CHUNK_BETA (CHUNK_DECAY + RECURRENT_SPAN * RECURRENT_DECAYS)

// The piece of rows [first, first + count): each invocation's float4 items and
// gates.
struct chunk_stage {
    vec4 items[CHUNK_ITEMS_PER_THREAD];
    float gates[CHUNK_GATES_PER_THREAD];
};

chunk_stage chunk_load_piece(uint64_t inputs, uint head, uint key_head, uint block_row, int first, int count) {
    chunk_stage stage;
    [[unroll]] for (uint i = 0u; i < CHUNK_ITEMS_PER_THREAD; ++i) {
        const uint item = gl_LocalInvocationIndex + i * CHUNK_THREADS;
        stage.items[i] = vec4(0.0);
        const uint r = item / (CHUNK_PREPARED / 4u), at = 4u * (item % (CHUNK_PREPARED / 4u));
        if (item < CHUNK_ITEMS && int(r) < count) {
            const uint64_t row = uint64_t(first + int(r));
            stage.items[i] = at < 2u * RECURRENT_W
                ? element_vec4_at(chunk_rows_at(inputs, key_head, row) + uint64_t(at) * 4ul)
                : element_vec4_at(chunk_values_at(inputs, head, row) + uint64_t(block_row + at - 2u * RECURRENT_W) * 4ul);
        }
    }
    [[unroll]] for (uint i = 0u; i < CHUNK_GATES_PER_THREAD; ++i) {
        const uint item = gl_LocalInvocationIndex + i * CHUNK_THREADS;
        const bool decay = item < RECURRENT_SPAN * RECURRENT_DECAYS;
        const uint r = decay ? item / RECURRENT_DECAYS : item - RECURRENT_SPAN * RECURRENT_DECAYS;
        stage.gates[i] = 0.0;
        if (item < CHUNK_GATES && int(r) < count)
            stage.gates[i] = element_f32_at(decay
                    ? chunk_decays_at(inputs, head, uint64_t(first + int(r))) + uint64_t(item % RECURRENT_DECAYS) * 4ul
                    : chunk_betas_at(inputs, head, uint64_t(first + int(r))));
    }
    return stage;
}

void chunk_store_piece(chunk_stage stage) {
    [[unroll]] for (uint i = 0u; i < CHUNK_ITEMS_PER_THREAD; ++i) {
        const uint item = gl_LocalInvocationIndex + i * CHUNK_THREADS;
        if (item < CHUNK_ITEMS) {
            const uint r = item / (CHUNK_PREPARED / 4u), at = 4u * (item % (CHUNK_PREPARED / 4u));
            [[unroll]] for (uint e = 0u; e < 4u; ++e)
                seismic_shared_f32[r * CHUNK_PREPARED + at + e] = stage.items[i][e];
        }
    }
    [[unroll]] for (uint i = 0u; i < CHUNK_GATES_PER_THREAD; ++i) {
        const uint item = gl_LocalInvocationIndex + i * CHUNK_THREADS;
        if (item < CHUNK_GATES)
            seismic_shared_f32[CHUNK_DECAY + item] = stage.gates[i];
    }
}

// Advance the lane's state row over the slot's rows [lo, lo + stop) and
// write their raw outputs.
void chunk_scan_rows(recurrent_inputs in_, recurrent_slot slot, uint head, uint block_row, uint64_t mixed,
    inout float state[CHUNK_MAX_COLUMNS]) {
    const uint64_t inputs = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_INPUTS);
    const uint W = RECURRENT_W, L = CHUNK_LANES;
    const uint key_head = recurrent_key_head(in_, head);
    const uint local_row = chunk_row(), part = chunk_part();
    const int end = slot.lo + slot.stop;
    chunk_stage stage = chunk_load_piece(inputs, head, key_head, block_row, slot.lo, min(int(RECURRENT_SPAN), end - slot.lo));
    for (int first = slot.lo; first < end; first += int(RECURRENT_SPAN)) {
        const int count = min(int(RECURRENT_SPAN), end - first);
        // The previous piece's reads of the region finish before it is
        // restaged.
        barrier();
        chunk_store_piece(stage);
        barrier();
        const int next = first + int(RECURRENT_SPAN);
        if (next < end)
            stage = chunk_load_piece(inputs, head, key_head, block_row, next, min(int(RECURRENT_SPAN), end - next));
        for (int r = 0; r < count; ++r) {
            const uint base = uint(r) * CHUNK_PREPARED;
            const uint factors = CHUNK_DECAY + uint(r) * RECURRENT_DECAYS;
            const float beta = seismic_shared_f32[CHUNK_BETA + uint(r)];
            // One decay per value head is read once.
            float k[CHUNK_MAX_COLUMNS], factor[CHUNK_MAX_COLUMNS];
            [[unroll]] for (uint c = 0u; c < CHUNK_COLUMNS; ++c)
                factor[c] = RECURRENT_DECAYS == 1u && c > 0u ? factor[0]
                    : seismic_shared_f32[factors + RECURRENT_DECAY_COLUMN(part + L * c)];
            float remembered = 0.0;
            [[unroll]] for (uint c = 0u; c < CHUNK_COLUMNS; ++c) {
                k[c] = seismic_shared_f32[base + W + part + L * c];
                remembered = seismic_fma_rn(state[c] * factor[c], k[c], remembered);
            }
            remembered = reduce_lanes_sum(remembered, L);
            const float residual = (seismic_shared_f32[base + 2u * W + local_row] - remembered) * beta;
            float attended = 0.0;
            [[unroll]] for (uint c = 0u; c < CHUNK_COLUMNS; ++c) {
                state[c] = seismic_fma_rn(residual, k[c], state[c] * factor[c]);
                attended = seismic_fma_rn(state[c], seismic_shared_f32[base + part + L * c], attended);
            }
            attended = reduce_lanes_sum(attended, L);
            if (part == 0u && chunk_active())
                element_put(ELEMENT_ACT, mixed, recurrent_raw_index(uint64_t(first + r), head, block_row + local_row),
                    attended);
        }
    }
}

// The subgroup's ROWS state rows in the row-sequential layout (lane l holding
// columns [l CPL, (l + 1) CPL) of each), from the chunked layout (row r on
// the lanes [8r, 8r + 8), lane j holding columns j, j + 8, ...). The rows are
// the subgroup's own, so the hand-off is a subgroup exchange of registers:
// nothing is written to memory that another invocation then reads back, which
// no barrier orders on every driver (the dispatch reads through `readonly`
// references and enables no memory model; see `element.glsl`). Every lane
// calls it.
void chunk_sequential_rows(float state[CHUNK_MAX_COLUMNS], out float rows[RECURRENT_MAX_ROWS][RECURRENT_CPL]) {
    const uint lane = SEISMIC_LANE;
    [[unroll]] for (uint r = 0u; r < RECURRENT_MAX_ROWS; ++r) {
        [[unroll]] for (uint c = 0u; c < RECURRENT_CPL; ++c)
            rows[r][c] = 0.0;
        if (r < CHUNK_ROWS) {
            [[unroll]] for (uint c = 0u; c < RECURRENT_CPL; ++c) {
                const uint column = lane * RECURRENT_CPL + c;
                // Register k of the lane holding the column is the one every
                // lane sends in this exchange; the lane that wants it keeps it.
                [[unroll]] for (uint k = 0u; k < CHUNK_COLUMNS; ++k) {
                    const float sent = seismic_shuffle(state[k], r * CHUNK_LANES + column % CHUNK_LANES);
                    if (column / CHUNK_LANES == k)
                        rows[r][c] = sent;
                }
            }
        }
    }
}

// The scan launch: a slot of more than CHUNK_SEQUENTIAL_ROWS rows with a
// nonzero stop scans its rows before the stop row from the staged pieces,
// then advances the rest row-sequentially; any other slot (and the
// uncovered rows) advances row-sequentially throughout.
void chunk_scan() {
    const uint64_t slot_index = gl_WorkGroupID.z;
    if (slot_index == SEISMIC_DIM_B || !chunk_long(recurrent_slot_of(slot_index))) {
        recurrent_sequential(CHUNK_ROWS);
        return;
    }
    const recurrent_inputs in_ = recurrent_inputs_of();
    const recurrent_slot slot = recurrent_slot_of(slot_index);
    const uint64_t mixed = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_MIXED);
    const uint head = gl_WorkGroupID.x;
    const uint block_row = gl_WorkGroupID.y * CHUNK_BLOCK_ROWS;
    float state[CHUNK_MAX_COLUMNS];
    chunk_load_version(in_, slot, head, block_row + chunk_row(), state);
    recurrent_publish_window(in_, slot, head * gl_NumWorkGroups.y + gl_WorkGroupID.y, RECURRENT_NV * gl_NumWorkGroups.y);
    chunk_scan_rows(in_, slot, head, block_row, mixed, state);
    if (slot.stop == slot.hi - slot.lo) {
        chunk_store(in_, slot.target, head, block_row + chunk_row(), state);
    } else {
        // The rows after the stop row advance row-sequentially from the
        // state at the stop row, which that advance publishes (their
        // innovations feed the tape).
        const uint first_row = block_row + SEISMIC_SUBGROUP * CHUNK_ROWS;
        float rows[RECURRENT_MAX_ROWS][RECURRENT_CPL];
        chunk_sequential_rows(state, rows);
        recurrent_advance_rows(in_, slot, slot.lo + slot.stop, head, block_row, CHUNK_BLOCK_ROWS, CHUNK_ROWS, first_row,
            rows, mixed);
    }
}
