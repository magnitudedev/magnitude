// draft_top_k: one threadgroup of 256 threads per row selects the row's K
// highest logits in (value descending, token ascending) order, ignoring NaNs.
//
// Tiles of <= 4096 elements are loaded once into registers (coalesced
// packed_float4 loads when unit-strided) and selected via a 4-pass 8-bit
// threadgroup radix select over order-preserving float keys (`order::finite_key`),
// with early exit on exact boundaries and exact lowest-token tie-breaking via
// a secondary radix select on `~token` only when ties split the K-th rank.
// Large vocabularies (`V > 8192`) run `draft_top_k_parts` (unsorted K per
// 4096-element partition into compact 4-byte token + 4-byte value scratch)
// followed by `draft_top_k_merge`, which selects and sorts the global top K via
// an in-threadgroup-memory bitonic sort (simdgroup-only with zero threadgroup
// barriers for K <= 64).
#include <seismic/element.h>
#include "lib/core/order.h"
#include "lib/core/reduce.h"

namespace {

constant constexpr uint SORT_MAX_K = 1024u;

inline uint draft_key(float value) {
    return metal::isnan(value) ? 0u : order::finite_key(value);
}

inline ulong pack_sort_key(uint key, int token) {
    return (ulong(key) << 32) | ulong(~uint(token));
}

inline uint next_pow2(uint v) {
    if (v <= 1u) return 1u;
    return 1u << (32u - metal::clz(v - 1u));
}

inline int tile_token_offset(int slot, uint thread_index) {
    return (slot >> 2) * 1024 + int(thread_index) * 4 + (slot & 3);
}

inline ushort load_logit_tile(
    device const float *part_line,
    ulong valid,
    ulong stride,
    uint thread_index,
    thread float (&vals)[16],
    thread uint (&keys)[16]) {
    const bool vec_ok = stride == 1ul;
    ushort active = 0u;
    for (uint c = 0u; c < 4u; ++c) {
        const ulong local_base = ulong(c) * 1024ul + ulong(thread_index) * 4ul;
        if (vec_ok && local_base + 4ul <= valid) {
            const float4 v = float4(*reinterpret_cast<device const packed_float4 *>(part_line + local_base));
            const float f[4] = {v.x, v.y, v.z, v.w};
            for (uint j = 0u; j < 4u; ++j) {
                const uint k = draft_key(f[j]);
                vals[c * 4u + j] = f[j];
                keys[c * 4u + j] = k;
                if (k != 0u) active |= ushort(1u << (c * 4u + j));
            }
        } else {
            for (uint j = 0u; j < 4u; ++j) {
                const ulong offset = local_base + ulong(j);
                if (offset < valid) {
                    const float f = part_line[offset * stride];
                    const uint k = draft_key(f);
                    vals[c * 4u + j] = f;
                    keys[c * 4u + j] = k;
                    if (k != 0u) active |= ushort(1u << (c * 4u + j));
                } else {
                    vals[c * 4u + j] = 0.0f;
                    keys[c * 4u + j] = 0u;
                }
            }
        }
    }
    return active;
}

// One threadgroup atomic per distinct digit of a simdgroup.
inline void simd_histogram_add(
    threadgroup uint *bins,
    uint key,
    uint shift,
    bool pending,
    uint lane) {
    const uint value = (key >> shift) & 255u;
    uint waiting = uint(uint64_t(simd_ballot(pending)));
    while (waiting != 0u) {
        const uint leader = metal::ctz(waiting);
        const uint leading = simd_shuffle(value, ushort(leader));
        const uint same = uint(uint64_t(simd_ballot(pending && value == leading)));
        if (lane == leader) {
            atomic_fetch_add_explicit(
                reinterpret_cast<threadgroup atomic_uint *>(&bins[leading]),
                metal::popcount(same),
                memory_order_relaxed);
        }
        pending = pending && value != leading;
        waiting &= ~same;
    }
}

// 4-pass 8-bit threadgroup radix select over up to 4096 keys held in registers
// (16 per thread, 256 threads). On return:
// - `above_mask` marks items strictly in the top-K (or all valid items if
//   fewer than `k` exist, or exact boundary hit).
// - `equal_mask` marks items tied at the K-th key when `total_equal > take`;
//   otherwise `equal_mask == 0` and `take == 0`.
inline void block_radix_select_tile(
    thread const uint (&keys)[16],
    ushort active,
    uint k,
    uint tid,
    uint sg,
    uint lane,
    threadgroup uint *bins,
    threadgroup uint *warp_sums,
    threadgroup uint *state,
    thread ushort &above_mask,
    thread ushort &equal_mask,
    thread uint &take,
    thread uint &total_equal) {
    bins[tid] = 0u;
    if (tid == 0u) {
        state[0] = 0u;
        state[1] = k;
        state[2] = 0u;
        state[3] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ushort match = active;
    above_mask = 0u;

    for (int digit = 3; digit >= 0; --digit) {
        const uint shift = uint(digit) * 8u;
        const uint remaining = state[1];

        if (digit == 3 && simd_all(match == 0xffffu)) {
            for (int i = 0; i < 16; ++i) {
                simd_histogram_add(bins, keys[i], 24u, true, lane);
            }
        } else {
            uint m = uint(match);
            while (simd_any(m != 0u)) {
                const bool has = m != 0u;
                const uint idx = has ? metal::ctz(m) : 0u;
                if (has) m &= m - 1u;
                simd_histogram_add(bins, has ? keys[idx] : 0u, shift, has, lane);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Thread tid owns bin 255 - tid (descending order).
        const uint bin = 255u - tid;
        const uint count = bins[bin];
        const uint warp_inc = simd_prefix_inclusive_sum(count);
        if (lane == 31u) warp_sums[sg] = warp_inc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint before_warp = 0u;
        for (uint w = 0u; w < 8u; ++w) {
            if (w < sg) before_warp += warp_sums[w];
        }
        const uint above = before_warp + warp_inc - count;
        const uint total = before_warp + warp_inc;
        if (tid == 255u && total < remaining) {
            state[3] = 2u;
        }
        if (above < remaining && remaining <= above + count) {
            state[0] = bin;
            state[1] = remaining - above;
            state[2] = count;
            if (above + count == remaining) {
                state[3] = 1u;
            }
        }
        bins[tid] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint done = state[3];
        if (done == 2u) {
            above_mask |= match;
            equal_mask = 0u;
            take = 0u;
            total_equal = 0u;
            return;
        }
        const uint win_byte = state[0];
        ushort next_match = 0u;
        uint m = uint(match);
        while (m != 0u) {
            const uint idx = metal::ctz(m);
            m &= m - 1u;
            const uint b = (keys[idx] >> shift) & 255u;
            if (b > win_byte) {
                above_mask |= ushort(1u << idx);
            } else if (b == win_byte) {
                next_match |= ushort(1u << idx);
            }
        }
        if (done == 1u) {
            above_mask |= next_match;
            equal_mask = 0u;
            take = 0u;
            total_equal = 0u;
            return;
        }
        match = next_match;
    }

    equal_mask = match;
    take = state[1];
    total_equal = state[2];
}

// Select exact top-k mask from a register tile, breaking any threshold ties by
// lowest token index (unique tokens guarantee exact selection).
template <typename TokenOf>
inline ushort select_tile_topk_mask(
    thread const uint (&keys)[16],
    ushort active,
    uint k,
    uint tid,
    uint sg,
    uint lane,
    TokenOf token_of,
    threadgroup uint *bins,
    threadgroup uint *warp_sums,
    threadgroup uint *state) {
    ushort above_mask = 0u;
    ushort equal_mask = 0u;
    uint take = 0u;
    uint total_equal = 0u;
    block_radix_select_tile(
        keys, active, k, tid, sg, lane, bins, warp_sums, state,
        above_mask, equal_mask, take, total_equal);
    if (total_equal > take && take > 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tie_keys[16];
        for (int i = 0; i < 16; ++i) tie_keys[i] = 0u;
        uint m = uint(equal_mask);
        while (m != 0u) {
            const uint idx = metal::ctz(m);
            m &= m - 1u;
            tie_keys[idx] = ~uint(token_of(int(idx)));
        }
        ushort tie_above = 0u;
        ushort tie_eq = 0u;
        uint tie_take = 0u;
        uint tie_total = 0u;
        block_radix_select_tile(
            tie_keys, equal_mask, take, tid, sg, lane, bins, warp_sums, state,
            tie_above, tie_eq, tie_take, tie_total);
        above_mask |= tie_above | tie_eq;
    }
    return above_mask;
}

// Exclusive prefix sum over 256 threads using 8 simdgroup sums (1 barrier;
// caller must ensure warp_sums is not live from a prior use).
inline uint block_exclusive_add_256(
    uint value,
    uint sg,
    uint lane,
    threadgroup uint *warp_sums,
    thread uint &total) {
    const uint warp_inc = simd_prefix_inclusive_sum(value);
    if (lane == 31u) warp_sums[sg] = warp_inc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before_warp = 0u;
    uint sum = 0u;
    for (uint w = 0u; w < 8u; ++w) {
        const uint ws = warp_sums[w];
        if (w < sg) before_warp += ws;
        sum += ws;
    }
    total = sum;
    return before_warp + warp_inc - value;
}

// Sort `count <= k <= SORT_MAX_K` packed (key, ~token) entries in `sort_keys`
// / `sort_vals` descending and write `candidates` and `unary` for `row`.
inline void bitonic_sort_and_emit(
    threadgroup ulong *sort_keys,
    threadgroup float *sort_vals,
    uint count,
    uint k,
    ulong row,
    uint tid,
    device int *candidates,
    ulong cand_stride_0,
    ulong cand_stride_1,
    const element::Out<element::F32> unary,
    ulong unary_stride_0,
    ulong unary_stride_1) {
    const uint P = next_pow2(k);
    for (uint i = count + tid; i < P; i += 256u) {
        sort_keys[i] = 0ul;
        sort_vals[i] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (P <= 64u) {
        if (tid < 32u) {
            const uint half_p = P >> 1u;
            for (uint size = 2u; size <= P; size <<= 1u) {
                for (uint stride = size >> 1u; stride > 0u; stride >>= 1u) {
                    if (tid < half_p) {
                        const uint pos = 2u * tid - (tid & (stride - 1u));
                        const uint partner = pos + stride;
                        const bool desc = (pos & size) == 0u;
                        const ulong ka = sort_keys[pos];
                        const ulong kb = sort_keys[partner];
                        if (desc ? (ka < kb) : (ka > kb)) {
                            sort_keys[pos] = kb;
                            sort_keys[partner] = ka;
                            const float va = sort_vals[pos];
                            sort_vals[pos] = sort_vals[partner];
                            sort_vals[partner] = va;
                        }
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            for (uint rank = tid; rank < k; rank += 32u) {
                const ulong sk = sort_keys[rank];
                const int token = sk != 0ul ? int(~uint(sk)) : reduce::LowerIndex::none;
                const ulong at = row * ulong(k) + ulong(rank);
                candidates[at * cand_stride_0] = token;
                candidates[at * cand_stride_0 + cand_stride_1] = 0;
                unary.put(row * unary_stride_0 + ulong(rank) * unary_stride_1, sort_vals[rank]);
            }
        }
        return;
    }

    const uint half_p = P >> 1u;
    for (uint size = 2u; size <= P; size <<= 1u) {
        for (uint stride = size >> 1u; stride > 0u; stride >>= 1u) {
            for (uint idx = tid; idx < half_p; idx += 256u) {
                const uint pos = 2u * idx - (idx & (stride - 1u));
                const uint partner = pos + stride;
                const bool desc = (pos & size) == 0u;
                const ulong ka = sort_keys[pos];
                const ulong kb = sort_keys[partner];
                if (desc ? (ka < kb) : (ka > kb)) {
                    sort_keys[pos] = kb;
                    sort_keys[partner] = ka;
                    const float va = sort_vals[pos];
                    sort_vals[pos] = sort_vals[partner];
                    sort_vals[partner] = va;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint rank = tid; rank < k; rank += 256u) {
        const ulong sk = sort_keys[rank];
        const int token = sk != 0ul ? int(~uint(sk)) : reduce::LowerIndex::none;
        const ulong at = row * ulong(k) + ulong(rank);
        candidates[at * cand_stride_0] = token;
        candidates[at * cand_stride_0 + cand_stride_1] = 0;
        unary.put(row * unary_stride_0 + ulong(rank) * unary_stride_1, sort_vals[rank]);
    }
}

// Streaming 4-pass radix select over `n` items accessed via
// `fetch(i, val, key, token)` (setting `key = 0` for invalid/NaN items),
// compacting the top `k <= SORT_MAX_K` into `sort_keys`/`sort_vals` and
// returning the number compacted.
template <typename Fetch>
inline uint stream_radix_select(
    ulong n,
    uint k,
    uint tid,
    uint sg,
    uint lane,
    Fetch fetch,
    threadgroup uint *bins,
    threadgroup uint *warp_sums,
    threadgroup uint *state,
    threadgroup ulong *sort_keys,
    threadgroup float *sort_vals) {
    bins[tid] = 0u;
    if (tid == 0u) {
        state[0] = 0u; // prefix_bits
        state[1] = k;  // remaining
        state[2] = 0u; // win_count
        state[3] = 0u; // done (2 = fewer than k valid items)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint mask = 0u;
    for (int digit = 3; digit >= 0; --digit) {
        const uint prefix_bits = state[0];
        const uint remaining = state[1];
        const uint shift = uint(digit) * 8u;

        for (ulong base = 0ul; base < n; base += 256ul) {
            const ulong i = base + ulong(tid);
            float val = 0.0f;
            uint key = 0u;
            int token = -1;
            const bool valid = i < n && fetch(i, val, key, token) && key != 0u;
            const bool match = valid && (key & mask) == prefix_bits;
            simd_histogram_add(bins, key, shift, match, lane);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint bin = 255u - tid;
        const uint count = bins[bin];
        const uint warp_inc = simd_prefix_inclusive_sum(count);
        if (lane == 31u) warp_sums[sg] = warp_inc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint before_warp = 0u;
        for (uint w = 0u; w < 8u; ++w) {
            if (w < sg) before_warp += warp_sums[w];
        }
        const uint above = before_warp + warp_inc - count;
        const uint total = before_warp + warp_inc;
        if (tid == 255u && total < remaining) {
            state[3] = 2u;
        }
        if (above < remaining && remaining <= above + count) {
            state[0] = prefix_bits | (bin << shift);
            state[1] = remaining - above;
            state[2] = count;
        }
        bins[tid] = 0u;
        mask |= 255u << shift;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (state[3] == 2u) break;
    }

    const uint done = state[3];
    const uint threshold = state[0];
    const uint take = state[1];
    const uint total_equal = state[2];

    // If ties split the K-th rank, find the exact `take`-th smallest token
    // among items with `key == threshold` via radix select on `~token`.
    uint tie_inv_threshold = 0u;
    const bool need_tie = (done != 2u) && (total_equal > take) && (take > 0u);
    if (need_tie) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            state[0] = 0u;
            state[1] = take;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tmask = 0u;
        for (int digit = 3; digit >= 0; --digit) {
            const uint prefix_bits = state[0];
            const uint remaining = state[1];
            const uint shift = uint(digit) * 8u;
            for (ulong base = 0ul; base < n; base += 256ul) {
                const ulong i = base + ulong(tid);
                float val = 0.0f;
                uint key = 0u;
                int token = -1;
                const bool valid = i < n && fetch(i, val, key, token) && key == threshold;
                const uint inv_tok = ~uint(token);
                const bool match = valid && (inv_tok & tmask) == prefix_bits;
                simd_histogram_add(bins, inv_tok, shift, match, lane);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint bin = 255u - tid;
            const uint count = bins[bin];
            const uint warp_inc = simd_prefix_inclusive_sum(count);
            if (lane == 31u) warp_sums[sg] = warp_inc;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint before_warp = 0u;
            for (uint w = 0u; w < 8u; ++w) {
                if (w < sg) before_warp += warp_sums[w];
            }
            const uint above = before_warp + warp_inc - count;
            if (above < remaining && remaining <= above + count) {
                state[0] = prefix_bits | (bin << shift);
                state[1] = remaining - above;
            }
            bins[tid] = 0u;
            tmask |= 255u << shift;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        tie_inv_threshold = state[0];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) state[0] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ulong base = 0ul; base < n; base += 256ul) {
        const ulong i = base + ulong(tid);
        float val = 0.0f;
        uint key = 0u;
        int token = -1;
        if (i < n && fetch(i, val, key, token) && key != 0u) {
            const bool picked = (done == 2u) || (key > threshold) ||
                                (key == threshold && (!need_tie || (~uint(token)) >= tie_inv_threshold));
            if (picked) {
                const uint slot = atomic_fetch_add_explicit(
                    reinterpret_cast<threadgroup atomic_uint *>(&state[0]),
                    1u,
                    memory_order_relaxed);
                if (slot < k) {
                    sort_keys[slot] = pack_sort_key(key, token);
                    sort_vals[slot] = val;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint written = state[0];
    return written < k ? written : k;
}

// Updates (best, best_token) with (value, token) if it orders strictly after
// (previous, previous_token) and beats the current best. NaNs are excluded.
inline void consider_after(
    float value,
    int token,
    float previous,
    int previous_token,
    thread float &best,
    thread int &best_token) {
    const bool after = value < previous || (value == previous && token > previous_token);
    if (after && (value > best || (value == best && token < best_token))) {
        best = value;
        best_token = token;
    }
}

struct TileTokenOf {
    int begin;
    uint tid;
    inline int operator()(int idx) const {
        return begin + tile_token_offset(idx, tid);
    }
};

struct ArrayTokenOf {
    thread const int *toks;
    inline int operator()(int idx) const {
        return toks[idx];
    }
};

struct LineFetch {
    device const float *line;
    ulong stride1;
    inline bool operator()(ulong i, thread float &val, thread uint &key, thread int &token) const {
        val = line[i * stride1];
        key = draft_key(val);
        token = int(i);
        return true;
    }
};

struct PartialFetch {
    device const int *partial_ids;
    device const float *partial_values;
    ulong base;
    inline bool operator()(ulong i, thread float &val, thread uint &key, thread int &token) const {
        const ulong at = base + i;
        token = partial_ids[at];
        if (token < 0) return false;
        val = partial_values[at];
        key = draft_key(val);
        return key != 0u;
    }
};

} // namespace

kernel void draft_top_k(
    device const float *logits [[buffer(SEISMIC_BUFFER_LOGITS)]],
    device int *candidates [[buffer(SEISMIC_BUFFER_CANDIDATES)]],
    SEISMIC_OUT_UNARY_PARAM(unary_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (SEISMIC_DIM_K == 0ul) return;
    threadgroup uint bins[256];
    threadgroup uint warp_sums[8];
    threadgroup uint state[4];
    threadgroup ulong sort_keys[SORT_MAX_K];
    threadgroup float sort_vals[SORT_MAX_K];
    const element::Out<element::F32> unary(unary_cells);
    const ulong vocabulary = SEISMIC_DIM_V;
    device const float *line = logits + ulong(row) * SEISMIC_LOGITS_STRIDE_0;

    if (SEISMIC_DIM_K <= SORT_MAX_K) {
        const uint k = uint(SEISMIC_DIM_K);
        if (vocabulary <= 4096ul) {
            float vals[16];
            uint keys[16];
            const ushort active = load_logit_tile(
                line, vocabulary, SEISMIC_LOGITS_STRIDE_1, thread_index, vals, keys);
            const ushort picked = select_tile_topk_mask(
                keys, active, k, thread_index, sg, lane,
                TileTokenOf{0, thread_index},
                bins, warp_sums, state);
            uint total_picked = 0u;
            uint pos = block_exclusive_add_256(
                uint(metal::popcount(uint(picked))), sg, lane, warp_sums, total_picked);
            uint m = uint(picked);
            while (m != 0u) {
                const uint idx = metal::ctz(m);
                m &= m - 1u;
                if (pos < k) {
                    sort_keys[pos] = pack_sort_key(keys[idx], tile_token_offset(int(idx), thread_index));
                    sort_vals[pos] = vals[idx];
                }
                ++pos;
            }
            const uint count = total_picked < k ? total_picked : k;
            bitonic_sort_and_emit(
                sort_keys, sort_vals, count, k, ulong(row), thread_index,
                candidates, SEISMIC_CANDIDATES_STRIDE_0, SEISMIC_CANDIDATES_STRIDE_1,
                unary, SEISMIC_UNARY_STRIDE_0, SEISMIC_UNARY_STRIDE_1);
            return;
        }
        const uint count = stream_radix_select(
            vocabulary, k, thread_index, sg, lane,
            LineFetch{line, SEISMIC_LOGITS_STRIDE_1},
            bins, warp_sums, state, sort_keys, sort_vals);
        bitonic_sort_and_emit(
            sort_keys, sort_vals, count, k, ulong(row), thread_index,
            candidates, SEISMIC_CANDIDATES_STRIDE_0, SEISMIC_CANDIDATES_STRIDE_1,
            unary, SEISMIC_UNARY_STRIDE_0, SEISMIC_UNARY_STRIDE_1);
        return;
    }

    // Fallback for K > SORT_MAX_K.
    threadgroup reduce::Candidate partials[8];
    const bool vec_ok = SEISMIC_LOGITS_STRIDE_1 == 1ul;
    const ulong vec_end = vocabulary & ~3ul;
    float previous = INFINITY;
    int previous_token = -1;
    for (ulong rank = 0ul; rank < SEISMIC_DIM_K; ++rank) {
        float best = -INFINITY;
        int best_token = reduce::LowerIndex::none;
        if (vec_ok) {
            for (ulong base = ulong(thread_index) * 4ul; base < vec_end; base += 1024ul) {
                const float4 v = float4(*reinterpret_cast<device const packed_float4 *>(line + base));
                consider_after(v.x, int(base + 0ul), previous, previous_token, best, best_token);
                consider_after(v.y, int(base + 1ul), previous, previous_token, best, best_token);
                consider_after(v.z, int(base + 2ul), previous, previous_token, best, best_token);
                consider_after(v.w, int(base + 3ul), previous, previous_token, best, best_token);
            }
            const ulong tail_token = vec_end + ulong(thread_index);
            if (tail_token < vocabulary) {
                consider_after(line[tail_token], int(tail_token), previous, previous_token, best, best_token);
            }
        } else {
            for (ulong token = thread_index; token < vocabulary; token += 256ul) {
                consider_after(
                    line[token * SEISMIC_LOGITS_STRIDE_1], int(token), previous, previous_token, best, best_token);
            }
        }
        const reduce::Candidate winner = reduce::group_argmax<reduce::LowerIndex, 8>(
            reduce::Candidate{best, best_token}, partials, sg, lane);
        previous = winner.score;
        previous_token = winner.index;
        if (thread_index == 0u) {
            const ulong at = ulong(row) * SEISMIC_DIM_K + rank;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0] = previous_token;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0 + SEISMIC_CANDIDATES_STRIDE_1] = 0;
            unary.put(ulong(row) * SEISMIC_UNARY_STRIDE_0 + rank * SEISMIC_UNARY_STRIDE_1, previous);
        }
    }
}

// Exact vocabulary partitions: each threadgroup of 256 threads loads its
// 4096-element partition once into registers, selects the exact top K via
// block radix select (without sorting), and emits compact (token, value) pairs
// into scratch.
kernel void draft_top_k_parts(
    device const float *logits [[buffer(SEISMIC_BUFFER_LOGITS)]],
    device int *partial_tokens [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIAL_TOKENS)]],
    device float *partial_values [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIAL_VALUES)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (SEISMIC_DIM_K == 0ul) return;
    threadgroup uint bins[256];
    threadgroup uint warp_sums[8];
    threadgroup uint state[4];
    const ulong row = group.x;
    const ulong part = group.y;
    const ulong parts = (SEISMIC_DIM_V + 4095ul) / 4096ul;
    const ulong begin = part * 4096ul;
    const ulong vocabulary = min(SEISMIC_DIM_V, begin + 4096ul);
    const ulong valid = vocabulary - begin;
    const ulong count = min(SEISMIC_DIM_K, valid);
    device const float *line = logits + row * SEISMIC_LOGITS_STRIDE_0;
    device const float *part_line = line + begin * SEISMIC_LOGITS_STRIDE_1;
    const ulong out_base = (row * parts + part) * SEISMIC_DIM_K;

    float vals[16];
    uint keys[16];
    const ushort active = load_logit_tile(
        part_line, valid, SEISMIC_LOGITS_STRIDE_1, thread_index, vals, keys);
    const ushort picked = (count == valid)
        ? active
        : select_tile_topk_mask(
              keys, active, uint(count), thread_index, sg, lane,
              TileTokenOf{int(begin), thread_index},
              bins, warp_sums, state);

    uint total_picked = 0u;
    uint pos = block_exclusive_add_256(
        uint(metal::popcount(uint(picked))), sg, lane, warp_sums, total_picked);
    uint m = uint(picked);
    while (m != 0u) {
        const uint idx = metal::ctz(m);
        m &= m - 1u;
        if (ulong(pos) < count) {
            partial_tokens[out_base + ulong(pos)] = int(begin) + tile_token_offset(int(idx), thread_index);
            partial_values[out_base + ulong(pos)] = vals[idx];
        }
        ++pos;
    }
    // Pad any short final partition or NaN-depleted slots with -1.
    const ulong written = ulong(total_picked) < count ? ulong(total_picked) : count;
    for (ulong rank = written + ulong(thread_index); rank < SEISMIC_DIM_K; rank += 256ul) {
        partial_tokens[out_base + rank] = -1;
        partial_values[out_base + rank] = -INFINITY;
    }
}

// The global top K is contained in the union of the partitions' top K.
// Select and sort the exact global top K via 4-pass radix select + bitonic sort.
kernel void draft_top_k_merge(
    device const int *partial_ids [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIAL_TOKENS)]],
    device const float *partial_values [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIAL_VALUES)]],
    device int *candidates [[buffer(SEISMIC_BUFFER_CANDIDATES)]],
    SEISMIC_OUT_UNARY_PARAM(unary_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (SEISMIC_DIM_K == 0ul) return;
    threadgroup uint bins[256];
    threadgroup uint warp_sums[8];
    threadgroup uint state[4];
    threadgroup ulong sort_keys[SORT_MAX_K];
    threadgroup float sort_vals[SORT_MAX_K];
    const element::Out<element::F32> unary(unary_cells);
    const ulong parts = (SEISMIC_DIM_V + 4095ul) / 4096ul;
    const ulong count = parts * SEISMIC_DIM_K;
    const ulong base = ulong(row) * count;

    if (SEISMIC_DIM_K <= SORT_MAX_K) {
        const uint k = uint(SEISMIC_DIM_K);
        if (count <= 4096ul) {
            float vals[16];
            uint keys[16];
            int toks[16];
            ushort active = 0u;
            for (uint slot = 0u; slot < 16u; ++slot) {
                const ulong i = ulong(slot) * 256ul + ulong(thread_index);
                if (i < count) {
                    const ulong at = base + i;
                    const int token = partial_ids[at];
                    if (token >= 0) {
                        const float value = partial_values[at];
                        const uint key = draft_key(value);
                        if (key != 0u) {
                            vals[slot] = value;
                            keys[slot] = key;
                            toks[slot] = token;
                            active |= ushort(1u << slot);
                            continue;
                        }
                    }
                }
                vals[slot] = 0.0f;
                keys[slot] = 0u;
                toks[slot] = -1;
            }
            const ushort picked = select_tile_topk_mask(
                keys, active, k, thread_index, sg, lane,
                ArrayTokenOf{toks},
                bins, warp_sums, state);
            uint total_picked = 0u;
            uint pos = block_exclusive_add_256(
                uint(metal::popcount(uint(picked))), sg, lane, warp_sums, total_picked);
            uint m = uint(picked);
            while (m != 0u) {
                const uint idx = metal::ctz(m);
                m &= m - 1u;
                if (pos < k) {
                    sort_keys[pos] = pack_sort_key(keys[idx], toks[idx]);
                    sort_vals[pos] = vals[idx];
                }
                ++pos;
            }
            const uint written = total_picked < k ? total_picked : k;
            bitonic_sort_and_emit(
                sort_keys, sort_vals, written, k, ulong(row), thread_index,
                candidates, SEISMIC_CANDIDATES_STRIDE_0, SEISMIC_CANDIDATES_STRIDE_1,
                unary, SEISMIC_UNARY_STRIDE_0, SEISMIC_UNARY_STRIDE_1);
            return;
        }
        const uint written = stream_radix_select(
            count, k, thread_index, sg, lane,
            PartialFetch{partial_ids, partial_values, base},
            bins, warp_sums, state, sort_keys, sort_vals);
        bitonic_sort_and_emit(
            sort_keys, sort_vals, written, k, ulong(row), thread_index,
            candidates, SEISMIC_CANDIDATES_STRIDE_0, SEISMIC_CANDIDATES_STRIDE_1,
            unary, SEISMIC_UNARY_STRIDE_0, SEISMIC_UNARY_STRIDE_1);
        return;
    }

    // Fallback for K > SORT_MAX_K.
    threadgroup reduce::Candidate partials[8];
    float previous = INFINITY;
    int previous_token = -1;
    for (ulong rank = 0ul; rank < SEISMIC_DIM_K; ++rank) {
        float best = -INFINITY;
        int best_token = reduce::LowerIndex::none;
        for (ulong i = thread_index; i < count; i += 256ul) {
            const ulong at = base + i;
            const int token = partial_ids[at];
            if (token < 0) continue;
            consider_after(partial_values[at], token, previous, previous_token, best, best_token);
        }
        const reduce::Candidate winner = reduce::group_argmax<reduce::LowerIndex, 8>(
            reduce::Candidate{best, best_token}, partials, sg, lane);
        previous = winner.score;
        previous_token = winner.index;
        if (thread_index == 0u) {
            const ulong at = ulong(row) * SEISMIC_DIM_K + rank;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0] = previous_token;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0 + SEISMIC_CANDIDATES_STRIDE_1] = 0;
            unary.put(ulong(row) * SEISMIC_UNARY_STRIDE_0 + rank * SEISMIC_UNARY_STRIDE_1, previous);
        }
    }
}
