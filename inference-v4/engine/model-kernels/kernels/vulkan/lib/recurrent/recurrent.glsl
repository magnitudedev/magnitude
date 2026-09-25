// Shared device code of the Vulkan gated-delta entries (`gated_delta_step`,
// `gated_delta_chunk`; contracts in recurrent.seismic): slot, bank and
// version lookup (the tape replay), tensor addressing, the per-head gates,
// the state-arena window publication, and the row-sequential advance, which
// records the tape rows after the stop row. The counterpart of
// `cuda/lib/recurrent/recurrent.cuh`, whose arithmetic it follows.
// Activation tensors are canonical in their last axis.
#include "../core/activation.glsl"

#define RECURRENT_NK uint(SEISMIC_DIM_NK)
#define RECURRENT_NV uint(SEISMIC_DIM_NV)
#define RECURRENT_W uint(SEISMIC_DIM_W)
#define RECURRENT_C uint(SEISMIC_DIM_C)
#define RECURRENT_CPL (RECURRENT_W / 32u)
#define RECURRENT_CH ((2u * RECURRENT_NK + RECURRENT_NV) * RECURRENT_W)
// Columns of the projection's gate segments.
#define RECURRENT_ALPHA (RECURRENT_CH + RECURRENT_NV * RECURRENT_W)
#define RECURRENT_BETA (RECURRENT_ALPHA + RECURRENT_NV)
// A tape row: the innovations u [NV, W], the normalized keys k [NK, W], the
// decays d [NV].
#define RECURRENT_TAPE_U 0u
#define RECURRENT_TAPE_K (RECURRENT_NV * RECURRENT_W)
#define RECURRENT_TAPE_D (RECURRENT_NV * RECURRENT_W + RECURRENT_NK * RECURRENT_W)
// Rows the sequential advance convolves at once.
#define RECURRENT_SPAN 8u
// The largest short-convolution width the advance's register window holds.
#define RECURRENT_MAX_C 8u

struct recurrent_inputs {
    uint64_t projection;
    uint64_t convolution;
    uint64_t rate;
    uint64_t time_bias;
    uint64_t window;
    uint64_t delta;
    uint64_t tape;
    float epsilon;
    bool grouped;
};

recurrent_inputs recurrent_inputs_of() {
    return recurrent_inputs(SEISMIC_PTR(SEISMIC_BUFFER_PROJECTION), SEISMIC_PTR(SEISMIC_BUFFER_CONVOLUTION),
        SEISMIC_PTR(SEISMIC_BUFFER_RATE), SEISMIC_PTR(SEISMIC_BUFFER_TIME_BIAS), SEISMIC_PTR(SEISMIC_BUFFER_WINDOW),
        SEISMIC_PTR(SEISMIC_BUFFER_DELTA), SEISMIC_PTR(SEISMIC_BUFFER_TAPE),
        element_word_f32(SEISMIC_PARAM_NORM_EPSILON), SEISMIC_PARAM_GROUPED != 0ul);
}

// One slot's rows [lo, hi), its publication row count, the version it reads
// (bank `source` advanced by its first `taped` tape rows) and its successor.
struct recurrent_slot {
    int lo;
    int hi;
    int stop;
    int source;
    int taped;
    int target;
};

recurrent_slot recurrent_slot_of(uint64_t slot) {
    const uint64_t segments = SEISMIC_PTR(SEISMIC_BUFFER_SEGMENTS);
    return recurrent_slot(element_i32_index(segments, slot * SEISMIC_SEGMENTS_STRIDE_0),
        element_i32_index(segments, slot * SEISMIC_SEGMENTS_STRIDE_0 + SEISMIC_SEGMENTS_STRIDE_1),
        element_i32_index(SEISMIC_PTR(SEISMIC_BUFFER_STOP), slot * SEISMIC_STOP_STRIDE_0),
        element_i32_index(SEISMIC_PTR(SEISMIC_BUFFER_PREVIOUS_BANK), slot * SEISMIC_PREVIOUS_BANK_STRIDE_0),
        element_i32_index(SEISMIC_PTR(SEISMIC_BUFFER_PREVIOUS_TAPE), slot * SEISMIC_PREVIOUS_TAPE_STRIDE_0),
        element_i32_index(SEISMIC_PTR(SEISMIC_BUFFER_FOLLOWING_BANK), slot * SEISMIC_FOLLOWING_BANK_STRIDE_0));
}

// Rows the slot records in its successor's tape: those after the stop row, at
// most T.
int recurrent_tape_rows(recurrent_slot slot) {
    return min(int(SEISMIC_DIM_T), slot.hi - slot.lo - slot.stop);
}

// Byte address of tape row `entry` of `bank`.
uint64_t recurrent_tape_row(recurrent_inputs in_, int bank, int entry) {
    return in_.tape + (uint64_t(bank) * SEISMIC_TAPE_STRIDE_0 + uint64_t(entry) * SEISMIC_TAPE_STRIDE_1) * 4ul;
}

// The one value head of each key head that records the key in a tape row.
bool recurrent_records_key(recurrent_inputs in_, uint head) {
    return in_.grouped ? head % (RECURRENT_NV / RECURRENT_NK) == 0u : head < RECURRENT_NK;
}

// The first row no slot covers: slots partition a prefix of rows.
int recurrent_covered_end() {
    const uint64_t slots = SEISMIC_DIM_B;
    return slots == 0ul ? 0
                        : element_i32_index(SEISMIC_PTR(SEISMIC_BUFFER_SEGMENTS),
                              (slots - 1ul) * SEISMIC_SEGMENTS_STRIDE_0 + SEISMIC_SEGMENTS_STRIDE_1);
}

float recurrent_projection(recurrent_inputs in_, uint64_t row, uint64_t column) {
    return element_at(ELEMENT_ACT, in_.projection, row * SEISMIC_PROJECTION_STRIDE_0 + column);
}

uint64_t recurrent_window_at(int bank, int tap, uint64_t channel) {
    return uint64_t(bank) * SEISMIC_WINDOW_STRIDE_0 + uint64_t(tap) * SEISMIC_WINDOW_STRIDE_1 + channel;
}

// Byte address of state row `state_row` of value head `head` in `bank`: W
// contiguous floats.
uint64_t recurrent_state_row(recurrent_inputs in_, int bank, uint head, uint state_row) {
    return in_.delta
        + (uint64_t(bank) * SEISMIC_DELTA_STRIDE_0 + uint64_t(head) * SEISMIC_DELTA_STRIDE_1
              + uint64_t(state_row) * SEISMIC_DELTA_STRIDE_2) * 4ul;
}

// The key head of value head `head`.
uint recurrent_key_head(recurrent_inputs in_, uint head) {
    return in_.grouped ? head * RECURRENT_NK / RECURRENT_NV : head % RECURRENT_NK;
}

// beta = sigmoid(b) and the decay exp(log_decay), log_decay =
// rate * softplus(alpha + time_bias).
void recurrent_gates(recurrent_inputs in_, uint64_t row, uint head, out float beta, out float log_decay, out float decay) {
    const float alpha = recurrent_projection(in_, row, RECURRENT_ALPHA + head);
    const float beta_input = recurrent_projection(in_, row, RECURRENT_BETA + head);
    const float shifted = alpha + element_f32_at(in_.time_bias + uint64_t(head) * SEISMIC_TIME_BIAS_STRIDE_0 * 4ul);
    const float softplus = max(shifted, 0.0) + log(1.0 + exp(-abs(shifted)));
    log_decay = element_f32_at(in_.rate + uint64_t(head) * SEISMIC_RATE_STRIDE_0 * 4ul) * softplus;
    beta = seismic_div_rn(1.0, 1.0 + exp(-beta_input));
    decay = exp(log_decay);
}

// Raw input row `position` (slot-local) of a channel: the source version's
// window rows before the slot, the projection after.
float recurrent_raw_input(recurrent_inputs in_, recurrent_slot slot, int position, uint channel) {
    return position < 0
        ? element_at(ELEMENT_ACT, in_.window,
              recurrent_window_at(slot.source, slot.taped + int(RECURRENT_C) - 1 + position, channel))
        : recurrent_projection(in_, uint64_t(slot.lo + position), channel);
}

// Publish the slot's successor window: the C - 1 raw rows before the
// publication row, then the raw rows of its tape. `part` of `parts`
// cooperating workgroups writes an even share.
void recurrent_publish_window(recurrent_inputs in_, recurrent_slot slot, uint part, uint parts) {
    const uint total = (RECURRENT_C - 1u + uint(recurrent_tape_rows(slot))) * RECURRENT_CH;
    const uint per = (total + parts - 1u) / parts;
    const uint first = part * per;
    const uint last = min(total, first + per);
    for (uint index = first + gl_LocalInvocationIndex; index < last; index += gl_WorkGroupSize.x) {
        const int row = int(index / RECURRENT_CH);
        const uint channel = index % RECURRENT_CH;
        element_put(ELEMENT_ACT, in_.window, recurrent_window_at(slot.target, row, channel),
            recurrent_raw_input(in_, slot, slot.stop + row - (int(RECURRENT_C) - 1), channel));
    }
}

// A subgroup's `rows` state rows in the sequential layout: lane l holds
// columns [l W / 32, (l + 1) W / 32) of each (at most 4 rows).
#define RECURRENT_MAX_ROWS 4

void recurrent_load_rows(recurrent_inputs in_, const uint rows, int bank, uint head, uint first_row,
    out float state[RECURRENT_MAX_ROWS][RECURRENT_CPL]) {
    const uint lane = gl_SubgroupInvocationID;
    [[unroll]] for (uint r = 0u; r < RECURRENT_MAX_ROWS; ++r)
        if (r < rows)
            [[unroll]] for (uint c = 0u; c < RECURRENT_CPL; ++c)
                state[r][c] = element_f32_at(recurrent_state_row(in_, bank, head, first_row + r) + uint64_t(lane * RECURRENT_CPL + c) * 4ul);
}

void recurrent_store_rows(recurrent_inputs in_, const uint rows, int bank, uint head, uint first_row,
    float state[RECURRENT_MAX_ROWS][RECURRENT_CPL]) {
    const uint lane = gl_SubgroupInvocationID;
    [[unroll]] for (uint r = 0u; r < RECURRENT_MAX_ROWS; ++r)
        if (r < rows)
            [[unroll]] for (uint c = 0u; c < RECURRENT_CPL; ++c)
                element_f32_put(recurrent_state_row(in_, bank, head, first_row + r) + uint64_t(lane * RECURRENT_CPL + c) * 4ul,
                    state[r][c]);
}

// A subgroup's `rows` rows of the slot's source version: the bank's state
// advanced by its first `taped` tape rows with the step's update, so the bits
// equal a run that published after those rows.
void recurrent_load_version(recurrent_inputs in_, recurrent_slot slot, const uint rows, uint head, uint first_row,
    out float state[RECURRENT_MAX_ROWS][RECURRENT_CPL]) {
    recurrent_load_rows(in_, rows, slot.source, head, first_row, state);
    const uint lane = gl_SubgroupInvocationID;
    const uint key = recurrent_key_head(in_, head);
    for (int entry = 0; entry < slot.taped; ++entry) {
        const uint64_t tape = recurrent_tape_row(in_, slot.source, entry);
        const float decay = element_f32_at(tape + uint64_t(RECURRENT_TAPE_D + head) * 4ul);
        float k[RECURRENT_CPL];
        [[unroll]] for (uint c = 0u; c < RECURRENT_CPL; ++c)
            k[c] = element_f32_at(tape + uint64_t(RECURRENT_TAPE_K + key * RECURRENT_W + lane * RECURRENT_CPL + c) * 4ul);
        [[unroll]] for (uint s = 0u; s < RECURRENT_MAX_ROWS; ++s) {
            if (s < rows) {
                const float u = element_f32_at(tape + uint64_t(RECURRENT_TAPE_U + head * RECURRENT_W + first_row + s) * 4ul);
                [[unroll]] for (uint c = 0u; c < RECURRENT_CPL; ++c)
                    state[s][c] = seismic_fma_rn(u, k[c], state[s][c] * decay);
            }
        }
    }
}

// The row-sequential gated delta rule over the slot's rows from `begin` (a
// workgroup's shape never changes bits). Every invocation of the workgroup
// calls it. The workgroup owns state rows [block_row, block_row + block_rows)
// of value head `head`; each subgroup holds its `rows` rows from `first_row`
// in `state` (the state before row `begin`), advances them, writes their mixed
// outputs and publishes them after the slot's first `stop` rows (at the start
// when `begin` is the publication row). Per span of up to RECURRENT_SPAN rows the
// workgroup convolves (causal depthwise conv + SiLU) the key head's q and k
// channels and the workgroup's v channels with every load in flight at once;
// each subgroup then walks the span's rows from shared memory: L2-normalized
// q (scaled by W^-1/2) and k, S <- decay S + beta (v - decay S k) k^T,
// output S q.
//
// Shared floats: prepared [SPAN][2W + block_rows], then beta [SPAN] and decay
// [SPAN].
void recurrent_advance_rows(recurrent_inputs in_, recurrent_slot slot, int begin, uint head, uint block_row,
    const uint block_rows, const uint rows, uint first_row, inout float state[RECURRENT_MAX_ROWS][RECURRENT_CPL],
    uint64_t mixed) {
    const uint W = RECURRENT_W, C = RECURRENT_C, CPL = RECURRENT_CPL;
    const uint prepared_width = 2u * W + block_rows;
    const uint beta_at = RECURRENT_SPAN * prepared_width;
    const uint decay_at = beta_at + RECURRENT_SPAN;
    const uint lane = gl_SubgroupInvocationID, thread = gl_LocalInvocationIndex;
    const uint key_row = recurrent_key_head(in_, head);
    const int publish = slot.lo + slot.stop;
    const int taped = recurrent_tape_rows(slot);
    const bool records_keys = recurrent_records_key(in_, head);
    if (begin == publish)
        recurrent_store_rows(in_, rows, slot.target, head, first_row, state);

    const float root = inversesqrt(float(W));
    for (int first = begin; first < slot.hi; first += int(RECURRENT_SPAN)) {
        const int count = min(int(RECURRENT_SPAN), slot.hi - first);
        // Each channel's C - 1 earlier inputs and the span's inputs load once.
        for (uint index = thread; index < prepared_width; index += gl_WorkGroupSize.x) {
            const uint channel = index < W ? key_row * W + index
                : index < 2u * W           ? (RECURRENT_NK + key_row) * W + index - W
                                           : (2u * RECURRENT_NK + head) * W + block_row + index - 2u * W;
            float weights[RECURRENT_MAX_C];
            [[unroll]] for (uint tap = 0u; tap < RECURRENT_MAX_C; ++tap)
                if (tap < C)
                    weights[tap] = element_f32_at(in_.convolution
                        + (uint64_t(channel) * SEISMIC_CONVOLUTION_STRIDE_0 + uint64_t(tap) * SEISMIC_CONVOLUTION_STRIDE_1) * 4ul);
            const int local = first - slot.lo;  // slot-local position of the span
            float inputs[RECURRENT_SPAN + RECURRENT_MAX_C - 1u];
            [[unroll]] for (uint i = 0u; i < RECURRENT_SPAN + RECURRENT_MAX_C - 1u; ++i) {
                if (i < RECURRENT_SPAN + C - 1u) {
                    const int position = local + int(i) - (int(C) - 1);
                    inputs[i] = int(i) >= count + int(C) - 1 ? 0.0 : recurrent_raw_input(in_, slot, position, channel);
                }
            }
            [[unroll]] for (uint r = 0u; r < RECURRENT_SPAN; ++r) {
                float sum = weights[C - 1u] * inputs[r + C - 1u];
                [[unroll]] for (uint tap = 0u; tap + 1u < RECURRENT_MAX_C; ++tap)
                    if (tap + 1u < C)
                        sum = seismic_fma_rn(weights[tap], inputs[r + tap], sum);
                seismic_shared_f32[r * prepared_width + index] = seismic_div_rn(sum, 1.0 + exp(-sum));
            }
        }
        if (thread < uint(count)) {
            float beta, log_decay, decay;
            recurrent_gates(in_, uint64_t(first) + thread, head, beta, log_decay, decay);
            seismic_shared_f32[beta_at + thread] = beta;
            seismic_shared_f32[decay_at + thread] = decay;
        }
        barrier();

        for (int r = 0; r < count; ++r) {
            const int row = first + r;
            const uint base = uint(r) * prepared_width;
            float q[RECURRENT_CPL], k[RECURRENT_CPL];
            float q_squares = 0.0, k_squares = 0.0;
            [[unroll]] for (uint c = 0u; c < RECURRENT_CPL; ++c) {
                q[c] = seismic_shared_f32[base + lane * CPL + c];
                k[c] = seismic_shared_f32[base + W + lane * CPL + c];
                q_squares = seismic_fma_rn(q[c], q[c], q_squares);
                k_squares = seismic_fma_rn(k[c], k[c], k_squares);
            }
            const float q_inverse = inversesqrt(seismic_subgroup_sum_f32(q_squares) + in_.epsilon) * root;
            const float k_inverse = inversesqrt(seismic_subgroup_sum_f32(k_squares) + in_.epsilon);
            [[unroll]] for (uint c = 0u; c < RECURRENT_CPL; ++c) {
                q[c] *= q_inverse;
                k[c] *= k_inverse;
            }
            const float decay = seismic_shared_f32[decay_at + uint(r)];
            const float beta = seismic_shared_f32[beta_at + uint(r)];

            float remembered[RECURRENT_MAX_ROWS];
            [[unroll]] for (uint s = 0u; s < RECURRENT_MAX_ROWS; ++s) {
                if (s < rows) {
                    float sum = 0.0;
                    [[unroll]] for (uint c = 0u; c < RECURRENT_CPL; ++c)
                        sum = seismic_fma_rn(state[s][c] * decay, k[c], sum);
                    remembered[s] = seismic_subgroup_sum_f32(sum);
                }
            }
            // Rows after the stop row are recorded in the successor's tape: the
            // subgroup owning state row 0 records the decay, and the key when
            // the head records its key head's key.
            const bool taping = row >= publish && row - publish < taped;
            const uint64_t entry = taping ? recurrent_tape_row(in_, slot.target, row - publish) : 0ul;
            if (taping && first_row == 0u) {
                if (records_keys)
                    [[unroll]] for (uint c = 0u; c < RECURRENT_CPL; ++c)
                        element_f32_put(entry + uint64_t(RECURRENT_TAPE_K + key_row * W + lane * CPL + c) * 4ul, k[c]);
                if (lane == 0u)
                    element_f32_put(entry + uint64_t(RECURRENT_TAPE_D + head) * 4ul, decay);
            }
            float outputs[RECURRENT_MAX_ROWS];
            [[unroll]] for (uint s = 0u; s < RECURRENT_MAX_ROWS; ++s) {
                if (s < rows) {
                    const float v = seismic_shared_f32[base + 2u * W + first_row - block_row + s];
                    const float residual = (v - remembered[s]) * beta;
                    if (taping && lane == s)
                        element_f32_put(entry + uint64_t(RECURRENT_TAPE_U + head * W + first_row + s) * 4ul, residual);
                    float sum = 0.0;
                    [[unroll]] for (uint c = 0u; c < RECURRENT_CPL; ++c) {
                        state[s][c] = seismic_fma_rn(residual, k[c], state[s][c] * decay);
                        sum = seismic_fma_rn(state[s][c], q[c], sum);
                    }
                    outputs[s] = seismic_subgroup_sum_f32(sum);
                }
            }
            if (lane < rows) {
                float value = outputs[0];
                [[unroll]] for (uint s = 1u; s < RECURRENT_MAX_ROWS; ++s)
                    if (s < rows && lane == s)
                        value = outputs[s];
                element_put(ELEMENT_ACT, mixed,
                    uint64_t(row) * SEISMIC_RESULT_0_STRIDE_0 + uint64_t(head) * SEISMIC_RESULT_0_STRIDE_1
                        + uint64_t(first_row + lane) * SEISMIC_RESULT_0_STRIDE_2,
                    value);
            }
            if (row + 1 == publish)
                recurrent_store_rows(in_, rows, slot.target, head, first_row, state);
        }
        barrier();
    }
}

// One workgroup of the sequential advance: (value head, state-row block,
// slot); slot index B zeroes the mixed rows no slot covers. Subgroup sg owns
// `rows` state rows from block_row + sg * rows.
void recurrent_sequential(const uint rows) {
    const recurrent_inputs in_ = recurrent_inputs_of();
    const uint64_t mixed = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    const uint head = gl_WorkGroupID.x;
    const uint block_rows = rows * gl_NumSubgroups;
    const uint block_row = gl_WorkGroupID.y * block_rows;
    const uint64_t slot_index = gl_WorkGroupID.z;
    const uint lane = gl_SubgroupInvocationID;
    const uint first_row = block_row + gl_SubgroupID * rows;

    if (slot_index == SEISMIC_DIM_B) {
        const int covered = recurrent_covered_end();
        for (uint64_t row = uint64_t(covered); row < SEISMIC_DIM_M; ++row)
            if (lane < rows)
                element_put(ELEMENT_ACT, mixed,
                    row * SEISMIC_RESULT_0_STRIDE_0 + uint64_t(head) * SEISMIC_RESULT_0_STRIDE_1
                        + uint64_t(first_row + lane) * SEISMIC_RESULT_0_STRIDE_2,
                    0.0);
        return;
    }

    const recurrent_slot slot = recurrent_slot_of(slot_index);
    float state[RECURRENT_MAX_ROWS][RECURRENT_CPL];
    recurrent_load_version(in_, slot, rows, head, first_row, state);
    recurrent_publish_window(in_, slot, head * gl_NumWorkGroups.y + gl_WorkGroupID.y, RECURRENT_NV * gl_NumWorkGroups.y);
    recurrent_advance_rows(in_, slot, slot.lo, head, block_row, block_rows, rows, first_row, state, mixed);
}
