// ngram_rows (contract and portable body in ngram.seismic): the n-gram table
// rows of each launch row from its token and its predecessors, the ones
// before the slot read from the slot's token window. A thread owns one item:
// item y < M forms row y's C * P rows ((-1, 0) past the slots), item M + b
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

// Exact `n % d` for any `n` and nonzero `d`, given `inv = floor((2^64 - 1) / d)`.
// Because `2^64 - d <= inv * d <= 2^64 - 1`, `q = floor(n * inv / 2^64)` satisfies
// `0 <= n - q * d < 2 * d`, so one conditional subtraction yields the exact remainder.
__device__ __forceinline__ u64 fast_rem_u64(u64 n, u64 d, u64 inv) {
    const u64 q = __umul64hi(n, inv);
    u64 r = n - q * d;
    if (r >= d) r -= d;
    return r;
}

__device__ __forceinline__ void publish_slot_window(const versions::Slots &slots, u64 slot_index, const int *tokens,
                                                    const u32 *window, int context, u64 bank_bytes,
                                                    const seismic_words_t &seismic_words_value) {
    const versions::Slot slot = versions::slot_of(slots, slot_index, seismic_words_value);
    if (slot.lo == slot.hi) return;
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
}

} // namespace ngram_rows_impl

extern "C" __global__ void ngram_rows(SEISMIC_KERNEL_PARAMS) {
    typedef versions::u32 u32;
    typedef versions::u64 u64;
    using ngram_rows_impl::fast_rem_u64;
    using ngram_rows_impl::publish_slot_window;
    using ngram_rows_impl::stored_token;

    constexpr u64 kContext = SEISMIC_DIM_C;
    constexpr u64 kHeadsPerOrder = SEISMIC_DIM_P;
    constexpr u64 kHeads = kContext * kHeadsPerOrder;
    constexpr u32 kBlockThreads = 64u;

    const u32 tid = threadIdx.x;
    const u64 block_start = static_cast<u64>(blockIdx.x) * blockDim.x;
    const u64 item = block_start + tid;
    const u64 launch_rows = SEISMIC_DIM_M;

    const int *tokens = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKENS));
    const u32 *window = reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_WINDOW));
    const versions::Slots slots = VERSIONS_SLOTS();
    const int context = static_cast<int>(kContext);
    const u64 bank_bytes = SEISMIC_WINDOW_STRIDE_0 * 4;

    // Publish-only blocks need no shared-memory hash parameters or barrier.
    if (block_start >= launch_rows) {
        if (item >= launch_rows + SEISMIC_DIM_B) return;
        publish_slot_window(slots, item - launch_rows, tokens, window, context, bank_bytes, seismic_words_value);
        return;
    }

    __shared__ u64 smem_multipliers[kContext + 1];
    __shared__ u64 smem_sizes[kHeads];
    __shared__ u64 smem_reciprocals[kHeads];
    __shared__ u32 smem_offsets[kHeads];
    __shared__ u32 smem_stored[kBlockThreads];
    __shared__ u32 smem_mapped[kBlockThreads];

    const u64 *multipliers = reinterpret_cast<const u64 *>(SEISMIC_PTR(SEISMIC_BUFFER_MULTIPLIERS));
    const u64 *sizes = reinterpret_cast<const u64 *>(SEISMIC_PTR(SEISMIC_BUFFER_SIZES));
    const u64 *offsets = reinterpret_cast<const u64 *>(SEISMIC_PTR(SEISMIC_BUFFER_OFFSETS));
    const u32 *map = reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_MAP));
    const int *boundary = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_BOUNDARY));
    int *rows = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));

    for (u64 i = tid; i <= kContext; i += blockDim.x) {
        smem_multipliers[i] = multipliers[i * SEISMIC_MULTIPLIERS_STRIDE_0];
    }
    for (u64 i = tid; i < kHeads; i += blockDim.x) {
        const u64 size = sizes[i * SEISMIC_SIZES_STRIDE_0];
        smem_sizes[i] = size;
        smem_reciprocals[i] = (~static_cast<u64>(0)) / size;
        smem_offsets[i] = static_cast<u32>(offsets[i * SEISMIC_OFFSETS_STRIDE_0]);
    }

    const bool dense_tokens = (SEISMIC_TOKENS_STRIDE_0 == 2 && SEISMIC_TOKENS_STRIDE_1 == 1);
    const bool dense_rows = (SEISMIC_RESULT_0_STRIDE_0 == 2 && SEISMIC_RESULT_0_STRIDE_1 == 1);

    u64 slot_index = SEISMIC_DIM_B;
    int status = 0;
    u64 hashed = 0;
    if (item < launch_rows) {
        slot_index = versions::slot_index_of_row(slots, static_cast<int>(item), seismic_words_value);
        if (slot_index < SEISMIC_DIM_B) {
            int token;
            if (dense_tokens) {
                const int2 pair = reinterpret_cast<const int2 *>(tokens)[item];
                token = pair.x;
                status = pair.y;
            } else {
                token = tokens[item * SEISMIC_TOKENS_STRIDE_0];
                status = tokens[item * SEISMIC_TOKENS_STRIDE_0 + SEISMIC_TOKENS_STRIDE_1];
            }
            if (SEISMIC_DIM_V > 0) {
                const u32 mapped = map[static_cast<u64>(token) * SEISMIC_MAP_STRIDE_0];
                hashed = static_cast<u64>(mapped);
                if (tid < kBlockThreads) smem_mapped[tid] = mapped;
            } else {
                hashed = static_cast<u64>(static_cast<long long>(token));
            }
            if (tid < kBlockThreads) smem_stored[tid] = static_cast<u32>(token + 1);
        }
    }

    __syncthreads();

    if (item >= launch_rows + SEISMIC_DIM_B) return;
    if (item >= launch_rows) {
        publish_slot_window(slots, item - launch_rows, tokens, window, context, bank_bytes, seismic_words_value);
        return;
    }

    const int row = static_cast<int>(item);
    const u64 first = static_cast<u64>(row) * kHeads;
    if (slot_index == SEISMIC_DIM_B) {
        if (dense_rows) {
            if (kHeads % 2 == 0) {
                int4 *out4 = reinterpret_cast<int4 *>(rows) + first / 2;
                const int4 empty4 = make_int4(-1, 0, -1, 0);
                #pragma unroll
                for (u64 i = 0; i < kHeads / 2; ++i) {
                    out4[i] = empty4;
                }
            } else {
                int2 *out2 = reinterpret_cast<int2 *>(rows) + first;
                const int2 empty2 = make_int2(-1, 0);
                #pragma unroll
                for (u64 head = 0; head < kHeads; ++head) {
                    out2[head] = empty2;
                }
            }
        } else {
            for (u64 head = 0; head < kHeads; ++head) {
                rows[(first + head) * SEISMIC_RESULT_0_STRIDE_0] = -1;
                rows[(first + head) * SEISMIC_RESULT_0_STRIDE_0 + SEISMIC_RESULT_0_STRIDE_1] = 0;
            }
        }
        return;
    }

    const int lo = slots.segments[slot_index * SEISMIC_SEGMENTS_STRIDE_0];
    const int local = row - lo;
    const u32 *source = nullptr;
    int taped = 0;
    if (local < context) {
        const int source_bank = slots.previous_bank[slot_index * SEISMIC_PREVIOUS_BANK_STRIDE_0];
        taped = slots.previous_tape[slot_index * SEISMIC_PREVIOUS_TAPE_STRIDE_0];
        source = reinterpret_cast<const u32 *>(versions::bank(window, source_bank, bank_bytes, seismic_words_value));
    }

    const u64 fill = static_cast<u64>(static_cast<long long>(static_cast<int>(static_cast<u32>(SEISMIC_PARAM_FILL))));
    // The boundary as the window stores it; 0 (no token) without one, which
    // cuts already.
    const u32 boundary_stored = SEISMIC_DIM_NB == 0 ? 0u : static_cast<u32>(boundary[0] + 1);
    u64 mixed = hashed * smem_multipliers[0];
    bool cut = false;

    for (int distance = 1; distance <= context; ++distance) {
        u32 stored;
        u64 value;
        if (local >= distance && tid >= static_cast<u32>(distance) && tid < kBlockThreads) {
            const u32 prev_tid = tid - static_cast<u32>(distance);
            stored = smem_stored[prev_tid];
            cut = cut || stored == 0u || stored == boundary_stored;
            value = cut                 ? fill
                    : SEISMIC_DIM_V > 0 ? static_cast<u64>(smem_mapped[prev_tid])
                                        : static_cast<u64>(stored - 1u);
        } else {
            stored = local >= distance
                         ? stored_token(tokens, row - distance, seismic_words_value)
                         : source[static_cast<u64>(taped + context + local - distance) * SEISMIC_WINDOW_STRIDE_1];
            cut = cut || stored == 0u || stored == boundary_stored;
            // A cut slot reads no map entry: its stored token may be none.
            value = cut                 ? fill
                    : SEISMIC_DIM_V > 0 ? static_cast<u64>(map[static_cast<u64>(stored - 1u) * SEISMIC_MAP_STRIDE_0])
                                        : static_cast<u64>(stored - 1u);
        }
        mixed ^= value * smem_multipliers[distance];

        const u64 order_base = static_cast<u64>(distance - 1) * kHeadsPerOrder;
        if (dense_rows) {
            if (kHeadsPerOrder % 2 == 0) {
                int4 *out4 = reinterpret_cast<int4 *>(rows) + (first + order_base) / 2;
                #pragma unroll
                for (u64 p = 0; p + 1 < kHeadsPerOrder; p += 2) {
                    const u64 h0 = order_base + p;
                    const u64 h1 = h0 + 1;
                    const int r0 = static_cast<int>(
                        smem_offsets[h0] + static_cast<u32>(fast_rem_u64(mixed, smem_sizes[h0], smem_reciprocals[h0])));
                    const int r1 = static_cast<int>(
                        smem_offsets[h1] + static_cast<u32>(fast_rem_u64(mixed, smem_sizes[h1], smem_reciprocals[h1])));
                    out4[p / 2] = make_int4(r0, status, r1, status);
                }
            } else {
                int2 *out2 = reinterpret_cast<int2 *>(rows) + first + order_base;
                #pragma unroll
                for (u64 p = 0; p < kHeadsPerOrder; ++p) {
                    const u64 head = order_base + p;
                    const int r = static_cast<int>(
                        smem_offsets[head] +
                        static_cast<u32>(fast_rem_u64(mixed, smem_sizes[head], smem_reciprocals[head])));
                    out2[p] = make_int2(r, status);
                }
            }
        } else {
            for (u64 p = 0; p < kHeadsPerOrder; ++p) {
                const u64 head = order_base + p;
                const int r = static_cast<int>(
                    smem_offsets[head] +
                    static_cast<u32>(fast_rem_u64(mixed, smem_sizes[head], smem_reciprocals[head])));
                rows[(first + head) * SEISMIC_RESULT_0_STRIDE_0] = r;
                rows[(first + head) * SEISMIC_RESULT_0_STRIDE_0 + SEISMIC_RESULT_0_STRIDE_1] = status;
            }
        }
    }
}
