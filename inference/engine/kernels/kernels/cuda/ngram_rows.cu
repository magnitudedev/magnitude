// ngram_rows (contract and portable body in ngram.seismic): the n-gram table
// rows of each launch row from its token and its predecessors, the ones
// before the slot read from the slot's token window. A thread owns one item:
// item y < M forms row y's C * P rows ((0, 0) past the slots), item M + b
// publishes slot b's successor window. The hash is the body's u64 words, so
// every backend names the same rows. The successor window is no slot's
// source, so the publishing threads run beside the row threads.

#include "lib/recurrent/versions.cuh"

namespace ngram_rows_impl {

typedef versions::u32 u32;
typedef versions::u64 u64;

// The token of launch row `row`, stored as the window stores it.
__device__ __forceinline__ u32 stored_token(const int *tokens, int row, const seismic_words_t &seismic_words_value) {
    return static_cast<u32>(tokens[static_cast<u64>(row) * SEISMIC_TOKENS_STRIDE_0] + 1);
}

} // namespace ngram_rows_impl

extern "C" __global__ void ngram_rows(SEISMIC_KERNEL_PARAMS) {
    typedef versions::u32 u32;
    typedef versions::u64 u64;
    using ngram_rows_impl::stored_token;
    const u64 item = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const u64 launch_rows = SEISMIC_DIM_M;
    if (item >= launch_rows + SEISMIC_DIM_B)
        return;
    const int *tokens = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKENS));
    const u64 *multipliers = reinterpret_cast<const u64 *>(SEISMIC_PTR(SEISMIC_BUFFER_MULTIPLIERS));
    const u64 *sizes = reinterpret_cast<const u64 *>(SEISMIC_PTR(SEISMIC_BUFFER_SIZES));
    const u64 *offsets = reinterpret_cast<const u64 *>(SEISMIC_PTR(SEISMIC_BUFFER_OFFSETS));
    const u32 *window = reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW));
    int *rows = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const versions::Slots slots = VERSIONS_SLOTS();
    const int context = static_cast<int>(SEISMIC_DIM_C);
    const u64 heads_per_order = SEISMIC_DIM_P;
    const u64 bank_bytes = SEISMIC_WINDOW_STRIDE_0 * 4;
    if (item >= launch_rows) {
        const versions::Slot slot = versions::slot_of(slots, item - launch_rows, seismic_words_value);
        const u32 *source =
            reinterpret_cast<const u32 *>(versions::bank(window, slot.source, bank_bytes, seismic_words_value));
        u32 *target = reinterpret_cast<u32 *>(versions::bank(window, slot.target, bank_bytes, seismic_words_value));
        const int published = context + versions::tape_rows(slot, static_cast<int>(SEISMIC_DIM_T));
        for (int tap = 0; tap < published; ++tap) {
            const int position = slot.stop + tap - context;
            target[static_cast<u64>(tap) * SEISMIC_WINDOW_STRIDE_1] =
                position < 0 ? source[static_cast<u64>(slot.taped + slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1]
                             : stored_token(tokens, slot.lo + position, seismic_words_value);
        }
        return;
    }
    const int row = static_cast<int>(item);
    const u64 heads = static_cast<u64>(context) * heads_per_order;
    const u64 first = static_cast<u64>(row) * heads;
    const u64 slot_index = versions::slot_index_of_row(slots, row, seismic_words_value);
    if (slot_index == SEISMIC_DIM_B) {
        for (u64 head = 0; head < heads; ++head) {
            rows[(first + head) * SEISMIC_RESULT_0_STRIDE_0] = 0;
            rows[(first + head) * SEISMIC_RESULT_0_STRIDE_0 + SEISMIC_RESULT_0_STRIDE_1] = 0;
        }
        return;
    }
    const versions::Slot slot = versions::slot_of(slots, slot_index, seismic_words_value);
    const u32 *source =
        reinterpret_cast<const u32 *>(versions::bank(window, slot.source, bank_bytes, seismic_words_value));
    const int local = row - slot.lo;
    const int reset = static_cast<int>(static_cast<u32>(SEISMIC_PARAM_RESET));
    const u32 reset_stored = static_cast<u32>(reset + 1);
    const int status = tokens[static_cast<u64>(row) * SEISMIC_TOKENS_STRIDE_0 + SEISMIC_TOKENS_STRIDE_1];
    u64 mixed = static_cast<u64>(static_cast<long long>(tokens[static_cast<u64>(row) * SEISMIC_TOKENS_STRIDE_0])) *
                multipliers[0];
    bool cut = false;
    for (int distance = 1; distance <= context; ++distance) {
        const u32 stored =
            local >= distance
                ? stored_token(tokens, row - distance, seismic_words_value)
                : source[static_cast<u64>(slot.taped + context + local - distance) * SEISMIC_WINDOW_STRIDE_1];
        cut = cut || stored == 0u || stored == reset_stored;
        const u64 token = cut ? static_cast<u64>(static_cast<long long>(reset)) : static_cast<u64>(stored - 1u);
        mixed ^= token * multipliers[static_cast<u64>(distance) * SEISMIC_MULTIPLIERS_STRIDE_0];
        for (u64 p = 0; p < heads_per_order; ++p) {
            const u64 head = static_cast<u64>(distance - 1) * heads_per_order + p;
            const u64 table_row =
                offsets[head * SEISMIC_OFFSETS_STRIDE_0] + mixed % sizes[head * SEISMIC_SIZES_STRIDE_0];
            rows[(first + head) * SEISMIC_RESULT_0_STRIDE_0] = static_cast<int>(table_row);
            rows[(first + head) * SEISMIC_RESULT_0_STRIDE_0 + SEISMIC_RESULT_0_STRIDE_1] = status;
        }
    }
}
