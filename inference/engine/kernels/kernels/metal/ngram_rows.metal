// ngram_rows (contract and portable body in ngram.seismic): the n-gram table
// rows of each launch row from its token and its predecessors, the ones
// before the slot read from the slot's token window. A thread owns one item:
// item y < M forms row y's C * P rows ((-1, 0) past the slots), item M + b
// publishes slot b's successor window. The hash is the body's u64 words, so
// every backend names the same rows. The successor window is no slot's
// source, so the publishing threads run beside the row threads.

#include "lib/recurrent/versions.h"

// The token of launch row `row`, stored as the window stores it.
inline uint stored_token(device const int *tokens, long row, constant ulong *seismic_words) {
    return uint(tokens[ulong(row) * SEISMIC_TOKENS_STRIDE_0] + 1);
}

// Exact `n % d` for any `n` and nonzero `d`, given `inv = floor((2^64 - 1) / d)`.
// Because `2^64 - d <= inv * d <= 2^64 - 1`, `q = floor(n * inv / 2^64)` satisfies
// `0 <= n - q * d < 2 * d`, so one conditional subtraction yields the exact remainder.
inline ulong fast_rem_u64(ulong n, ulong d, ulong inv) {
    const ulong q = metal::mulhi(n, inv);
    ulong r = n - q * d;
    if (r >= d) r -= d;
    return r;
}

inline void publish_slot_window(
    versions::Slots slots, ulong slot_index, device const int *tokens, device const ulong *window,
    long context, constant ulong *seismic_words) {
    const versions::Slot slot = versions::slot_of(slots, slot_index, seismic_words);
    if (slot.lo == slot.hi) return;
    device const uint *source = versions::bank<uint>(window, slot.source, SEISMIC_WINDOW_STRIDE_0, seismic_words);
    device uint *target = versions::bank<uint>(window, slot.target, SEISMIC_WINDOW_STRIDE_0, seismic_words);
    const long published = context + versions::tape_rows(slot, long(SEISMIC_DIM_T));
    for (long tap = 0; tap < published; ++tap) {
        const long position = slot.stop + tap - context;
        target[ulong(tap) * SEISMIC_WINDOW_STRIDE_1] = position < 0
            ? source[ulong(slot.taped + slot.stop + tap) * SEISMIC_WINDOW_STRIDE_1]
            : stored_token(tokens, slot.lo + position, seismic_words);
    }
}

kernel void ngram_rows(
    device const int *tokens [[buffer(SEISMIC_BUFFER_TOKENS)]],
    device const ulong *multipliers [[buffer(SEISMIC_BUFFER_MULTIPLIERS)]],
    device const ulong *sizes [[buffer(SEISMIC_BUFFER_SIZES)]],
    device const ulong *offsets [[buffer(SEISMIC_BUFFER_OFFSETS)]],
    device const uint *map [[buffer(SEISMIC_BUFFER_MAP)]],
    device const int *boundary [[buffer(SEISMIC_BUFFER_BOUNDARY)]],
    device const int *segments [[buffer(SEISMIC_BUFFER_SEGMENTS)]],
    device const int *stop [[buffer(SEISMIC_BUFFER_STOP)]],
    device const int *previous_bank [[buffer(SEISMIC_BUFFER_PREVIOUS_BANK)]],
    device const int *previous_tape [[buffer(SEISMIC_BUFFER_PREVIOUS_TAPE)]],
    device const int *following_bank [[buffer(SEISMIC_BUFFER_FOLLOWING_BANK)]],
    device const ulong *window [[buffer(SEISMIC_BUFFER_WINDOW)]],
    device int *rows [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint position_in_grid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    constexpr ulong kContext = SEISMIC_DIM_C;
    constexpr ulong kHeadsPerOrder = SEISMIC_DIM_P;
    constexpr ulong kHeads = kContext * kHeadsPerOrder;
    constexpr uint kBlockThreads = 64u;

    const ulong item = position_in_grid;
    const ulong block_start = ulong(position_in_grid - tid);
    const ulong launch_rows = SEISMIC_DIM_M;
    const long context = long(kContext);
    const versions::Slots slots{segments, stop, previous_bank, previous_tape, following_bank};

    // Publish-only threadgroups need no threadgroup hash parameters or barrier.
    if (block_start >= launch_rows) {
        if (item >= launch_rows + SEISMIC_DIM_B) return;
        publish_slot_window(slots, item - launch_rows, tokens, window, context, seismic_words);
        return;
    }

    threadgroup ulong smem_multipliers[kContext + 1];
    threadgroup ulong smem_sizes[kHeads];
    threadgroup ulong smem_reciprocals[kHeads];
    threadgroup uint smem_offsets[kHeads];
    threadgroup uint smem_stored[kBlockThreads];
    threadgroup uint smem_mapped[SEISMIC_DIM_V > 0 ? kBlockThreads : 1];

    for (ulong i = tid; i <= kContext; i += threads) {
        smem_multipliers[i] = multipliers[i * SEISMIC_MULTIPLIERS_STRIDE_0];
    }
    for (ulong i = tid; i < kHeads; i += threads) {
        const ulong size = sizes[i * SEISMIC_SIZES_STRIDE_0];
        smem_sizes[i] = size;
        smem_reciprocals[i] = (~ulong(0)) / size;
        smem_offsets[i] = uint(offsets[i * SEISMIC_OFFSETS_STRIDE_0]);
    }

    const bool dense_tokens = (SEISMIC_TOKENS_STRIDE_0 == 2ul && SEISMIC_TOKENS_STRIDE_1 == 1ul);
    const bool dense_rows = (SEISMIC_RESULT_0_STRIDE_0 == 2ul && SEISMIC_RESULT_0_STRIDE_1 == 1ul);

    ulong slot_index = SEISMIC_DIM_B;
    int status = 0;
    ulong hashed = 0;
    if (item < launch_rows) {
        slot_index = versions::slot_index_of_row(slots, long(item), seismic_words);
        if (slot_index < SEISMIC_DIM_B) {
            int token;
            if (dense_tokens) {
                const int2 pair = reinterpret_cast<device const int2 *>(tokens)[item];
                token = pair.x;
                status = pair.y;
            } else {
                token = tokens[item * SEISMIC_TOKENS_STRIDE_0];
                status = tokens[item * SEISMIC_TOKENS_STRIDE_0 + SEISMIC_TOKENS_STRIDE_1];
            }
            if (SEISMIC_DIM_V > 0) {
                const uint mapped = map[ulong(token) * SEISMIC_MAP_STRIDE_0];
                hashed = ulong(mapped);
                if (tid < kBlockThreads) smem_mapped[tid] = mapped;
            } else {
                hashed = ulong(long(token));
            }
            if (tid < kBlockThreads) smem_stored[tid] = uint(token + 1);
        }
    }

    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (item >= launch_rows + SEISMIC_DIM_B) return;
    if (item >= launch_rows) {
        publish_slot_window(slots, item - launch_rows, tokens, window, context, seismic_words);
        return;
    }

    const long row = long(item);
    const ulong first = ulong(row) * kHeads;
    if (slot_index == SEISMIC_DIM_B) {
        if (dense_rows) {
            if (kHeads % 2 == 0) {
                device int4 *out4 = reinterpret_cast<device int4 *>(rows) + first / 2;
                const int4 empty4 = int4(-1, 0, -1, 0);
                for (ulong i = 0; i < kHeads / 2; ++i) {
                    out4[i] = empty4;
                }
            } else {
                device int2 *out2 = reinterpret_cast<device int2 *>(rows) + first;
                const int2 empty2 = int2(-1, 0);
                for (ulong head = 0; head < kHeads; ++head) {
                    out2[head] = empty2;
                }
            }
        } else {
            for (ulong head = 0; head < kHeads; ++head) {
                rows[(first + head) * SEISMIC_RESULT_0_STRIDE_0] = -1;
                rows[(first + head) * SEISMIC_RESULT_0_STRIDE_0 + SEISMIC_RESULT_0_STRIDE_1] = 0;
            }
        }
        return;
    }

    const long lo = segments[slot_index * SEISMIC_SEGMENTS_STRIDE_0];
    const long local = row - lo;
    device const uint *source = nullptr;
    long taped = 0;
    if (local < context) {
        const ulong source_bank = ulong(previous_bank[slot_index * SEISMIC_PREVIOUS_BANK_STRIDE_0]);
        taped = previous_tape[slot_index * SEISMIC_PREVIOUS_TAPE_STRIDE_0];
        source = versions::bank<uint>(window, source_bank, SEISMIC_WINDOW_STRIDE_0, seismic_words);
    }

    const ulong fill = ulong(long(int(uint(SEISMIC_PARAM_FILL))));
    // The boundary as the window stores it; 0 (no token) without one, which
    // cuts already.
    const uint boundary_stored = SEISMIC_DIM_NB == 0 ? 0u : uint(boundary[0] + 1);
    ulong mixed = hashed * smem_multipliers[0];
    bool cut = false;

    for (long distance = 1; distance <= context; ++distance) {
        uint stored;
        ulong value;
        if (local >= distance && tid >= uint(distance) && tid < kBlockThreads) {
            const uint prev_tid = tid - uint(distance);
            stored = smem_stored[prev_tid];
            cut = cut || stored == 0u || stored == boundary_stored;
            value = cut ? fill
                : SEISMIC_DIM_V > 0 ? ulong(smem_mapped[prev_tid]) : ulong(stored - 1u);
        } else {
            stored = local >= distance
                ? stored_token(tokens, row - distance, seismic_words)
                : source[ulong(taped + context + local - distance) * SEISMIC_WINDOW_STRIDE_1];
            cut = cut || stored == 0u || stored == boundary_stored;
            // A cut slot reads no map entry: its stored token may be none.
            value = cut ? fill
                : SEISMIC_DIM_V > 0 ? ulong(map[ulong(stored - 1u) * SEISMIC_MAP_STRIDE_0]) : ulong(stored - 1u);
        }
        mixed ^= value * smem_multipliers[distance];

        const ulong order_base = ulong(distance - 1) * kHeadsPerOrder;
        if (dense_rows) {
            if (kHeadsPerOrder % 2 == 0) {
                device int4 *out4 = reinterpret_cast<device int4 *>(rows) + (first + order_base) / 2;
                for (ulong p = 0; p + 1 < kHeadsPerOrder; p += 2) {
                    const ulong h0 = order_base + p;
                    const ulong h1 = h0 + 1;
                    const int r0 = int(smem_offsets[h0] + uint(fast_rem_u64(mixed, smem_sizes[h0], smem_reciprocals[h0])));
                    const int r1 = int(smem_offsets[h1] + uint(fast_rem_u64(mixed, smem_sizes[h1], smem_reciprocals[h1])));
                    out4[p / 2] = int4(r0, status, r1, status);
                }
            } else {
                device int2 *out2 = reinterpret_cast<device int2 *>(rows) + first + order_base;
                for (ulong p = 0; p < kHeadsPerOrder; ++p) {
                    const ulong head = order_base + p;
                    const int r = int(smem_offsets[head] + uint(fast_rem_u64(mixed, smem_sizes[head], smem_reciprocals[head])));
                    out2[p] = int2(r, status);
                }
            }
        } else {
            for (ulong p = 0; p < kHeadsPerOrder; ++p) {
                const ulong head = order_base + p;
                const int r = int(smem_offsets[head] + uint(fast_rem_u64(mixed, smem_sizes[head], smem_reciprocals[head])));
                rows[(first + head) * SEISMIC_RESULT_0_STRIDE_0] = r;
                rows[(first + head) * SEISMIC_RESULT_0_STRIDE_0 + SEISMIC_RESULT_0_STRIDE_1] = status;
            }
        }
    }
}
