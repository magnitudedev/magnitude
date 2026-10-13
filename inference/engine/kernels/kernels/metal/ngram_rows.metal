// ngram_rows (contract and portable body in ngram.seismic): the n-gram table
// rows of each launch row from its token and its predecessors, the ones
// before the slot read from the slot's token window. A thread owns one item:
// item y < M forms row y's C * P rows ((0, 0) past the slots), item M + b
// publishes slot b's successor window. The hash is the body's u64 words, so
// every backend names the same rows. The successor window is no slot's
// source, so the publishing threads run beside the row threads.

#include "lib/recurrent/versions.h"

// The token of launch row `row`, stored as the window stores it.
inline uint stored_token(device const int *tokens, long row, constant ulong *seismic_words) {
    return uint(tokens[ulong(row) * SEISMIC_TOKENS_STRIDE_0] + 1);
}

kernel void ngram_rows(
    device const int *tokens [[buffer(SEISMIC_BUFFER_TOKENS)]],
    device const ulong *multipliers [[buffer(SEISMIC_BUFFER_MULTIPLIERS)]],
    device const ulong *sizes [[buffer(SEISMIC_BUFFER_SIZES)]],
    device const ulong *offsets [[buffer(SEISMIC_BUFFER_OFFSETS)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    device const ulong *window [[buffer(SEISMIC_BUFFER_WINDOW)]],
    device int *rows [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint position_in_grid [[thread_position_in_grid]]) {
    const ulong item = position_in_grid;
    const ulong launch_rows = SEISMIC_DIM_M;
    if (item >= launch_rows + SEISMIC_DIM_B)
        return;
    const long context = long(SEISMIC_DIM_C);
    const ulong heads_per_order = SEISMIC_DIM_P;
    const versions::Slots slots{segments, stop, previous_bank, previous_tape, following_bank};
    if (item >= launch_rows) {
        const versions::Slot slot = versions::slot_of(slots, item - launch_rows, seismic_words);
        device const uint *source = versions::bank<uint>(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words);
        device uint *target = versions::bank<uint>(window, slot.target, SEISMIC_WINDOW_STRIDE_0, seismic_words);
        const long published = context + versions::tape_rows(slot, long(SEISMIC_DIM_T));
        for (long tap = 0; tap < published; ++tap) {
            const long position = slot.stop + tap - context;
            target[ulong(tap) * SEISMIC_WINDOW_STRIDE_1] = position < 0
                ? source[ulong(slot.taped + slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1]
                : stored_token(tokens, slot.lo + position, seismic_words);
        }
        return;
    }
    const long row = long(item);
    const ulong first = ulong(row) * ulong(context) * heads_per_order;
    const ulong slot_index = versions::slot_index_of_row(slots, row, seismic_words);
    if (slot_index == SEISMIC_DIM_B) {
        for (ulong head = 0; head < ulong(context) * heads_per_order; ++head) {
            rows[(first + head) * SEISMIC_RESULT_0_STRIDE_0] = 0;
            rows[(first + head) * SEISMIC_RESULT_0_STRIDE_0 + SEISMIC_RESULT_0_STRIDE_1] = 0;
        }
        return;
    }
    const versions::Slot slot = versions::slot_of(slots, slot_index, seismic_words);
    device const uint *source = versions::bank<uint>(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words);
    const long local = row - slot.lo;
    const int reset = int(uint(SEISMIC_PARAM_RESET));
    const uint reset_stored = uint(reset + 1);
    const int status = tokens[ulong(row) * SEISMIC_TOKENS_STRIDE_0 + SEISMIC_TOKENS_STRIDE_1];
    ulong mixed = ulong(long(tokens[ulong(row) * SEISMIC_TOKENS_STRIDE_0])) * multipliers[0];
    bool cut = false;
    for (long distance = 1; distance <= context; ++distance) {
        const uint stored = local >= distance
            ? stored_token(tokens, row - distance, seismic_words)
            : source[ulong(slot.taped + context + local - distance) * SEISMIC_WINDOW_STRIDE_1];
        cut = cut || stored == 0u || stored == reset_stored;
        const ulong token = cut ? ulong(long(reset)) : ulong(stored - 1u);
        mixed ^= token * multipliers[ulong(distance) * SEISMIC_MULTIPLIERS_STRIDE_0];
        for (ulong p = 0; p < heads_per_order; ++p) {
            const ulong head = ulong(distance - 1) * heads_per_order + p;
            const ulong table_row = offsets[head * SEISMIC_OFFSETS_STRIDE_0] + mixed % sizes[head * SEISMIC_SIZES_STRIDE_0];
            rows[(first + head) * SEISMIC_RESULT_0_STRIDE_0] = int(table_row);
            rows[(first + head) * SEISMIC_RESULT_0_STRIDE_0 + SEISMIC_RESULT_0_STRIDE_1] = status;
        }
    }
}
