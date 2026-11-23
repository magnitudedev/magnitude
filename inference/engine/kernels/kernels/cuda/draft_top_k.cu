// draft_top_k: one block of 256 threads per row selects the row's K highest
// logits in (value descending, token ascending) order, ignoring NaNs.
//
// Tiles of <= 4096 elements are loaded once into registers (coalesced float4
// loads when unit-strided and 16-byte aligned) and selected via a 4-pass 8-bit
// block radix select over order-preserving float keys (`order::finite_key`),
// with early exit on exact boundaries and exact lowest-token tie-breaking via
// a secondary radix select on `~token` only when ties split the K-th rank.
// Large vocabularies (`V > 8192`) run `draft_top_k_parts` (unsorted K per
// 4096-element partition into compact 4-byte token + 4-byte value scratch)
// followed by `draft_top_k_merge`, which selects and sorts the global top K via
// an in-shared-memory bitonic sort (warp-only with zero block barriers for
// K <= 64).
#include <seismic/element.cuh>
#include "lib/core/order.cuh"
#include "lib/core/reduce.cuh"

namespace {

using u32 = unsigned int;
using u64 = unsigned long long;

constexpr u32 SORT_MAX_K = 1024u;

__device__ __forceinline__ u32 draft_key(float value) {
    return isnan(value) ? 0u : order::finite_key(value);
}

__device__ __forceinline__ u64 pack_sort_key(u32 key, int token) {
    return (static_cast<u64>(key) << 32) | static_cast<u64>(~static_cast<u32>(token));
}

__device__ __forceinline__ u32 next_pow2(u32 v) {
    if (v <= 1u) return 1u;
    return 1u << (32u - __clz(v - 1u));
}

__device__ __forceinline__ int tile_token_offset(int slot) {
    return (slot >> 2) * 1024 + static_cast<int>(threadIdx.x) * 4 + (slot & 3);
}

__device__ __forceinline__ unsigned short load_logit_tile(
    const float *part_line, u64 valid, u64 stride, float (&vals)[16], u32 (&keys)[16]) {
    const bool vec_ok = stride == 1ull && (reinterpret_cast<u64>(part_line) & 15ull) == 0ull;
    unsigned short active = 0u;
#pragma unroll
    for (unsigned c = 0; c < 4; ++c) {
        const u64 local_base = static_cast<u64>(c) * 1024ull + static_cast<u64>(threadIdx.x) * 4ull;
        if (vec_ok && local_base + 4ull <= valid) {
            const float4 v = *reinterpret_cast<const float4 *>(part_line + local_base);
            const float f[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const u32 k = draft_key(f[j]);
                vals[c * 4u + j] = f[j];
                keys[c * 4u + j] = k;
                if (k != 0u) active |= static_cast<unsigned short>(1u << (c * 4u + j));
            }
        } else {
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const u64 offset = local_base + static_cast<u64>(j);
                if (offset < valid) {
                    const float f = part_line[offset * stride];
                    const u32 k = draft_key(f);
                    vals[c * 4u + j] = f;
                    keys[c * 4u + j] = k;
                    if (k != 0u) active |= static_cast<unsigned short>(1u << (c * 4u + j));
                } else {
                    vals[c * 4u + j] = 0.0f;
                    keys[c * 4u + j] = 0u;
                }
            }
        }
    }
    return active;
}

__device__ __forceinline__ void warp_histogram_add(u32 *bins, u32 key, u32 shift, bool match, int lane) {
    const u32 active = __ballot_sync(0xffffffffu, match);
    if (match) {
        const u32 bin = (key >> shift) & 255u;
        const u32 peers = __match_any_sync(active, bin);
        if (__ffs(peers) - 1 == lane)
            atomicAdd(&bins[bin], static_cast<u32>(__popc(peers)));
    }
}

// 4-pass 8-bit block radix select over up to 4096 keys held in registers
// (16 per thread, 256 threads). On return:
// - `above_mask` marks items strictly in the top-K (or all valid items if
//   fewer than `k` exist, or exact boundary hit).
// - `equal_mask` marks items tied at the K-th key when `total_equal > take`;
//   otherwise `equal_mask == 0` and `take == 0`.
__device__ __forceinline__ void block_radix_select_tile(
    const u32 (&keys)[16], unsigned short active, u32 k,
    u32 *bins, u32 *warp_sums, u32 *state,
    unsigned short &above_mask, unsigned short &equal_mask,
    u32 &take, u32 &total_equal) {
    const u32 tid = threadIdx.x;
    const u32 warp = tid >> 5;
    const int lane = static_cast<int>(tid & 31u);

    bins[tid] = 0u;
    if (tid == 0u) {
        state[0] = 0u;
        state[1] = k;
        state[2] = 0u;
        state[3] = 0u;
    }
    __syncthreads();

    unsigned short match = active;
    above_mask = 0u;

    for (int digit = 3; digit >= 0; --digit) {
        const u32 shift = static_cast<u32>(digit) * 8u;
        const u32 remaining = state[1];

        if (digit == 3 && __all_sync(0xffffffffu, match == 0xffffu)) {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                warp_histogram_add(bins, keys[i], 24u, true, lane);
        } else {
            u32 m = match;
            while (__any_sync(0xffffffffu, m != 0u)) {
                const bool has = m != 0u;
                const int idx = has ? (__ffs(m) - 1) : 0;
                if (has) m &= m - 1u;
                warp_histogram_add(bins, has ? keys[idx] : 0u, shift, has, lane);
            }
        }
        __syncthreads();

        // Thread tid owns bin 255 - tid (descending order).
        const u32 bin = 255u - tid;
        const u32 count = bins[bin];
        const u32 warp_inc = reduce::lanes_inclusive_add(count, lane);
        if (lane == 31) warp_sums[warp] = warp_inc;
        __syncthreads();

        u32 before_warp = 0u;
#pragma unroll
        for (u32 w = 0u; w < 8u; ++w) {
            if (w < warp) before_warp += warp_sums[w];
        }
        const u32 above = before_warp + warp_inc - count;
        const u32 total = before_warp + warp_inc;
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
        __syncthreads();

        const u32 done = state[3];
        if (done == 2u) {
            above_mask |= match;
            equal_mask = 0u;
            take = 0u;
            total_equal = 0u;
            return;
        }
        const u32 win_byte = state[0];
        unsigned short next_match = 0u;
        u32 m = match;
        while (m != 0u) {
            const int idx = __ffs(m) - 1;
            m &= m - 1u;
            const u32 b = (keys[idx] >> shift) & 255u;
            if (b > win_byte) {
                above_mask |= static_cast<unsigned short>(1u << idx);
            } else if (b == win_byte) {
                next_match |= static_cast<unsigned short>(1u << idx);
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
template <class TokenOf>
__device__ __forceinline__ unsigned short select_tile_topk_mask(
    const u32 (&keys)[16], unsigned short active, u32 k, TokenOf token_of,
    u32 *bins, u32 *warp_sums, u32 *state) {
    unsigned short above_mask = 0u;
    unsigned short equal_mask = 0u;
    u32 take = 0u;
    u32 total_equal = 0u;
    block_radix_select_tile(keys, active, k, bins, warp_sums, state,
                            above_mask, equal_mask, take, total_equal);
    if (total_equal > take && take > 0u) {
        __syncthreads();
        u32 tie_keys[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) tie_keys[i] = 0u;
        u32 m = equal_mask;
        while (m != 0u) {
            const int idx = __ffs(m) - 1;
            m &= m - 1u;
            tie_keys[idx] = ~static_cast<u32>(token_of(idx));
        }
        unsigned short tie_above = 0u;
        unsigned short tie_eq = 0u;
        u32 tie_take = 0u;
        u32 tie_total = 0u;
        block_radix_select_tile(tie_keys, equal_mask, take, bins, warp_sums, state,
                                tie_above, tie_eq, tie_take, tie_total);
        above_mask |= tie_above | tie_eq;
    }
    return above_mask;
}

// Exclusive prefix sum over 256 threads using 8 warp sums (1 barrier; caller
// must ensure warp_sums is not live from a prior use).
__device__ __forceinline__ u32 block_exclusive_add_256(u32 value, u32 *warp_sums, u32 &total) {
    const u32 tid = threadIdx.x;
    const u32 warp = tid >> 5;
    const int lane = static_cast<int>(tid & 31u);
    const u32 warp_inc = reduce::lanes_inclusive_add(value, lane);
    if (lane == 31) warp_sums[warp] = warp_inc;
    __syncthreads();
    u32 before_warp = 0u;
    u32 sum = 0u;
#pragma unroll
    for (u32 w = 0u; w < 8u; ++w) {
        const u32 ws = warp_sums[w];
        if (w < warp) before_warp += ws;
        sum += ws;
    }
    total = sum;
    return before_warp + warp_inc - value;
}

// Sort `count <= k <= SORT_MAX_K` packed (key, ~token) entries in `sort_keys`
// / `sort_vals` descending and write `candidates` and `unary` for `row`.
template <class UnaryOut>
__device__ __forceinline__ void bitonic_sort_and_emit(
    u64 *sort_keys, float *sort_vals, u32 count, u32 k, u64 row, float infinity,
    int *candidates, u64 cand_stride_0, u64 cand_stride_1,
    const UnaryOut &unary, u64 unary_stride_0, u64 unary_stride_1) {
    const u32 P = next_pow2(k);
    for (u32 i = count + threadIdx.x; i < P; i += blockDim.x) {
        sort_keys[i] = 0ull;
        sort_vals[i] = -infinity;
    }
    __syncthreads();

    if (P <= 64u) {
        if (threadIdx.x < 32u) {
            const u32 half = P >> 1u;
            for (u32 size = 2u; size <= P; size <<= 1u) {
                for (u32 stride = size >> 1u; stride > 0u; stride >>= 1u) {
                    if (threadIdx.x < half) {
                        const u32 pos = 2u * threadIdx.x - (threadIdx.x & (stride - 1u));
                        const u32 partner = pos + stride;
                        const bool desc = (pos & size) == 0u;
                        const u64 ka = sort_keys[pos];
                        const u64 kb = sort_keys[partner];
                        if (desc ? (ka < kb) : (ka > kb)) {
                            sort_keys[pos] = kb;
                            sort_keys[partner] = ka;
                            const float va = sort_vals[pos];
                            sort_vals[pos] = sort_vals[partner];
                            sort_vals[partner] = va;
                        }
                    }
                    __syncwarp();
                }
            }
            for (u32 rank = threadIdx.x; rank < k; rank += 32u) {
                const u64 sk = sort_keys[rank];
                const int token = sk != 0ull ? static_cast<int>(~static_cast<u32>(sk)) : reduce::LowerIndex::none;
                const u64 at = row * static_cast<u64>(k) + rank;
                candidates[at * cand_stride_0] = token;
                candidates[at * cand_stride_0 + cand_stride_1] = 0;
                unary.put(row * unary_stride_0 + rank * unary_stride_1, sort_vals[rank]);
            }
        }
        return;
    }

    const u32 half = P >> 1u;
    for (u32 size = 2u; size <= P; size <<= 1u) {
        for (u32 stride = size >> 1u; stride > 0u; stride >>= 1u) {
            for (u32 idx = threadIdx.x; idx < half; idx += blockDim.x) {
                const u32 pos = 2u * idx - (idx & (stride - 1u));
                const u32 partner = pos + stride;
                const bool desc = (pos & size) == 0u;
                const u64 ka = sort_keys[pos];
                const u64 kb = sort_keys[partner];
                if (desc ? (ka < kb) : (ka > kb)) {
                    sort_keys[pos] = kb;
                    sort_keys[partner] = ka;
                    const float va = sort_vals[pos];
                    sort_vals[pos] = sort_vals[partner];
                    sort_vals[partner] = va;
                }
            }
            __syncthreads();
        }
    }
    for (u32 rank = threadIdx.x; rank < k; rank += blockDim.x) {
        const u64 sk = sort_keys[rank];
        const int token = sk != 0ull ? static_cast<int>(~static_cast<u32>(sk)) : reduce::LowerIndex::none;
        const u64 at = row * static_cast<u64>(k) + rank;
        candidates[at * cand_stride_0] = token;
        candidates[at * cand_stride_0 + cand_stride_1] = 0;
        unary.put(row * unary_stride_0 + rank * unary_stride_1, sort_vals[rank]);
    }
}

// Streaming 4-pass radix select over `n` items accessed via `fetch(i, val, key, token)`
// (setting `key = 0` for invalid/NaN items), compacting the top `k <= SORT_MAX_K`
// into `sort_keys`/`sort_vals` and returning the number compacted.
template <class Fetch>
__device__ __forceinline__ u32 stream_radix_select(
    u64 n, u32 k, Fetch fetch,
    u32 *bins, u32 *warp_sums, u32 *state,
    u64 *sort_keys, float *sort_vals) {
    const u32 tid = threadIdx.x;
    const u32 warp = tid >> 5;
    const int lane = static_cast<int>(tid & 31u);

    bins[tid] = 0u;
    if (tid == 0u) {
        state[0] = 0u; // prefix_bits
        state[1] = k;  // remaining
        state[2] = 0u; // win_count
        state[3] = 0u; // done (2 = fewer than k valid items)
    }
    __syncthreads();

    u32 mask = 0u;
    for (int digit = 3; digit >= 0; --digit) {
        const u32 prefix_bits = state[0];
        const u32 remaining = state[1];
        const u32 shift = static_cast<u32>(digit) * 8u;

        for (u64 base = 0ull; base < n; base += 256ull) {
            const u64 i = base + tid;
            float val = 0.0f;
            u32 key = 0u;
            int token = -1;
            const bool valid = i < n && fetch(i, val, key, token) && key != 0u;
            const bool match = valid && (key & mask) == prefix_bits;
            warp_histogram_add(bins, key, shift, match, lane);
        }
        __syncthreads();

        const u32 bin = 255u - tid;
        const u32 count = bins[bin];
        const u32 warp_inc = reduce::lanes_inclusive_add(count, lane);
        if (lane == 31) warp_sums[warp] = warp_inc;
        __syncthreads();

        u32 before_warp = 0u;
#pragma unroll
        for (u32 w = 0u; w < 8u; ++w) {
            if (w < warp) before_warp += warp_sums[w];
        }
        const u32 above = before_warp + warp_inc - count;
        const u32 total = before_warp + warp_inc;
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
        __syncthreads();
        if (state[3] == 2u) break;
    }

    const u32 done = state[3];
    const u32 threshold = state[0];
    const u32 take = state[1];
    const u32 total_equal = state[2];

    // If ties split the K-th rank, find the exact `take`-th smallest token
    // among items with `key == threshold` via radix select on `~token`.
    u32 tie_inv_threshold = 0u;
    const bool need_tie = (done != 2u) && (total_equal > take) && (take > 0u);
    if (need_tie) {
        __syncthreads();
        if (tid == 0u) {
            state[0] = 0u;
            state[1] = take;
        }
        __syncthreads();
        u32 tmask = 0u;
        for (int digit = 3; digit >= 0; --digit) {
            const u32 prefix_bits = state[0];
            const u32 remaining = state[1];
            const u32 shift = static_cast<u32>(digit) * 8u;
            for (u64 base = 0ull; base < n; base += 256ull) {
                const u64 i = base + tid;
                float val = 0.0f;
                u32 key = 0u;
                int token = -1;
                const bool valid = i < n && fetch(i, val, key, token) && key == threshold;
                const u32 inv_tok = ~static_cast<u32>(token);
                const bool match = valid && (inv_tok & tmask) == prefix_bits;
                warp_histogram_add(bins, inv_tok, shift, match, lane);
            }
            __syncthreads();

            const u32 bin = 255u - tid;
            const u32 count = bins[bin];
            const u32 warp_inc = reduce::lanes_inclusive_add(count, lane);
            if (lane == 31) warp_sums[warp] = warp_inc;
            __syncthreads();

            u32 before_warp = 0u;
#pragma unroll
            for (u32 w = 0u; w < 8u; ++w) {
                if (w < warp) before_warp += warp_sums[w];
            }
            const u32 above = before_warp + warp_inc - count;
            if (above < remaining && remaining <= above + count) {
                state[0] = prefix_bits | (bin << shift);
                state[1] = remaining - above;
            }
            bins[tid] = 0u;
            tmask |= 255u << shift;
            __syncthreads();
        }
        tie_inv_threshold = state[0];
    }

    __syncthreads();
    if (tid == 0u) state[0] = 0u;
    __syncthreads();

    for (u64 base = 0ull; base < n; base += 256ull) {
        const u64 i = base + tid;
        float val = 0.0f;
        u32 key = 0u;
        int token = -1;
        if (i < n && fetch(i, val, key, token) && key != 0u) {
            const bool picked = (done == 2u) || (key > threshold) ||
                                (key == threshold && (!need_tie || (~static_cast<u32>(token)) >= tie_inv_threshold));
            if (picked) {
                const u32 slot = atomicAdd(&state[0], 1u);
                if (slot < k) {
                    sort_keys[slot] = pack_sort_key(key, token);
                    sort_vals[slot] = val;
                }
            }
        }
    }
    __syncthreads();
    const u32 written = state[0];
    return written < k ? written : k;
}

__device__ __forceinline__ void consider_streaming(
    float value, int token, float previous, int previous_token, float &best, int &best_token) {
    const bool after = value < previous || (value == previous && token > previous_token);
    if (after && (value > best || (value == best && token < best_token))) {
        best = value;
        best_token = token;
    }
}

} // namespace

extern "C" __global__ void draft_top_k(SEISMIC_KERNEL_PARAMS) {
    if (SEISMIC_DIM_K == 0ull) return;
    __shared__ u32 bins[256];
    __shared__ u32 warp_sums[8];
    __shared__ u32 state[4];
    __shared__ u64 sort_keys[SORT_MAX_K];
    __shared__ float sort_vals[SORT_MAX_K];
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    int *candidates = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_CANDIDATES));
    const auto unary = SEISMIC_OUT_UNARY;
    const float infinity = __int_as_float(0x7f800000);
    const u64 row = blockIdx.x;
    const float *line = logits + row * SEISMIC_LOGITS_STRIDE_0;

    if (SEISMIC_DIM_K <= SORT_MAX_K) {
        const u32 k = static_cast<u32>(SEISMIC_DIM_K);
        if (SEISMIC_DIM_V <= 4096ull) {
            float vals[16];
            u32 keys[16];
            const unsigned short active = load_logit_tile(line, SEISMIC_DIM_V, SEISMIC_LOGITS_STRIDE_1, vals, keys);
            const unsigned short picked = select_tile_topk_mask(
                keys, active, k,
                [](int idx) { return tile_token_offset(idx); },
                bins, warp_sums, state);
            u32 total_picked = 0u;
            u32 pos = block_exclusive_add_256(static_cast<u32>(__popc(picked)), warp_sums, total_picked);
            u32 m = picked;
            while (m != 0u) {
                const int idx = __ffs(m) - 1;
                m &= m - 1u;
                if (pos < k) {
                    sort_keys[pos] = pack_sort_key(keys[idx], tile_token_offset(idx));
                    sort_vals[pos] = vals[idx];
                }
                ++pos;
            }
            const u32 count = total_picked < k ? total_picked : k;
            bitonic_sort_and_emit(
                sort_keys, sort_vals, count, k, row, infinity,
                candidates, SEISMIC_CANDIDATES_STRIDE_0, SEISMIC_CANDIDATES_STRIDE_1,
                unary, SEISMIC_UNARY_STRIDE_0, SEISMIC_UNARY_STRIDE_1);
            return;
        }
        const u64 stride1 = SEISMIC_LOGITS_STRIDE_1;
        const u32 count = stream_radix_select(
            SEISMIC_DIM_V, k,
            [&](u64 i, float &val, u32 &key, int &token) {
                val = line[i * stride1];
                key = draft_key(val);
                token = static_cast<int>(i);
                return true;
            },
            bins, warp_sums, state, sort_keys, sort_vals);
        bitonic_sort_and_emit(
            sort_keys, sort_vals, count, k, row, infinity,
            candidates, SEISMIC_CANDIDATES_STRIDE_0, SEISMIC_CANDIDATES_STRIDE_1,
            unary, SEISMIC_UNARY_STRIDE_0, SEISMIC_UNARY_STRIDE_1);
        return;
    }

    // Fallback for K > SORT_MAX_K.
    __shared__ reduce::Candidate partials[32];
    const bool vec_ok = SEISMIC_LOGITS_STRIDE_1 == 1ull && (reinterpret_cast<u64>(line) & 15ull) == 0ull;
    const u64 vec_end = SEISMIC_DIM_V & ~3ull;
    float previous = infinity;
    int previous_token = -1;
    for (u64 rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -infinity;
        int best_token = reduce::LowerIndex::none;
        if (vec_ok) {
            for (u64 base = static_cast<u64>(threadIdx.x) * 4ull; base < vec_end; base += 1024ull) {
                const float4 v = *reinterpret_cast<const float4 *>(line + base);
                const int tok = static_cast<int>(base);
                consider_streaming(v.x, tok + 0, previous, previous_token, best, best_token);
                consider_streaming(v.y, tok + 1, previous, previous_token, best, best_token);
                consider_streaming(v.z, tok + 2, previous, previous_token, best, best_token);
                consider_streaming(v.w, tok + 3, previous, previous_token, best, best_token);
            }
            const u64 tail = vec_end + threadIdx.x;
            if (tail < SEISMIC_DIM_V) {
                consider_streaming(line[tail], static_cast<int>(tail), previous, previous_token, best, best_token);
            }
        } else {
            for (u64 token = threadIdx.x; token < SEISMIC_DIM_V; token += blockDim.x) {
                consider_streaming(
                    line[token * SEISMIC_LOGITS_STRIDE_1], static_cast<int>(token),
                    previous, previous_token, best, best_token);
            }
        }
        const reduce::Candidate winner = reduce::group_argmax<reduce::LowerIndex>(
            reduce::Candidate{best, best_token}, partials);
        previous = winner.score;
        previous_token = winner.index;
        if (threadIdx.x == 0) {
            const u64 at = row * SEISMIC_DIM_K + rank;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0] = previous_token;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0 + SEISMIC_CANDIDATES_STRIDE_1] = 0;
            unary.put(row * SEISMIC_UNARY_STRIDE_0 + rank * SEISMIC_UNARY_STRIDE_1, previous);
        }
    }
}

// Exact vocabulary partitions: each block of 256 threads loads its 4096-element
// partition once into registers, selects the exact top K via block radix select
// (without sorting), and emits compact (token, value) pairs into scratch.
extern "C" __global__ void draft_top_k_parts(SEISMIC_KERNEL_PARAMS) {
    if (SEISMIC_DIM_K == 0ull) return;
    __shared__ u32 bins[256];
    __shared__ u32 warp_sums[8];
    __shared__ u32 state[4];
    const float infinity = __int_as_float(0x7f800000);
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    int *partial_tokens = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL_TOKENS));
    float *partial_values = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL_VALUES));
    const u64 row = blockIdx.x;
    const u64 part = blockIdx.y;
    const u64 parts = (SEISMIC_DIM_V + 4095ull) / 4096ull;
    const u64 begin = part * 4096ull;
    const u64 vocabulary = SEISMIC_DIM_V < begin + 4096ull ? SEISMIC_DIM_V : begin + 4096ull;
    const u64 valid = vocabulary - begin;
    const u64 count = SEISMIC_DIM_K < valid ? SEISMIC_DIM_K : valid;
    const float *line = logits + row * SEISMIC_LOGITS_STRIDE_0;
    const float *part_line = line + begin * SEISMIC_LOGITS_STRIDE_1;
    const u64 out_base = (row * parts + part) * SEISMIC_DIM_K;

    float vals[16];
    u32 keys[16];
    const unsigned short active = load_logit_tile(part_line, valid, SEISMIC_LOGITS_STRIDE_1, vals, keys);
    const unsigned short picked = (count == valid)
        ? active
        : select_tile_topk_mask(
              keys, active, static_cast<u32>(count),
              [&](int idx) { return static_cast<int>(begin) + tile_token_offset(idx); },
              bins, warp_sums, state);

    u32 total_picked = 0u;
    u32 pos = block_exclusive_add_256(static_cast<u32>(__popc(picked)), warp_sums, total_picked);
    u32 m = picked;
    while (m != 0u) {
        const int idx = __ffs(m) - 1;
        m &= m - 1u;
        if (pos < count) {
            partial_tokens[out_base + pos] = static_cast<int>(begin) + tile_token_offset(idx);
            partial_values[out_base + pos] = vals[idx];
        }
        ++pos;
    }
    // Pad any short final partition or NaN-depleted slots with -1.
    const u64 written = total_picked < count ? static_cast<u64>(total_picked) : count;
    for (u64 rank = written + threadIdx.x; rank < SEISMIC_DIM_K; rank += 256ull) {
        partial_tokens[out_base + rank] = -1;
        partial_values[out_base + rank] = -infinity;
    }
}

extern "C" __global__ void draft_top_k_merge(SEISMIC_KERNEL_PARAMS) {
    if (SEISMIC_DIM_K == 0ull) return;
    __shared__ u32 bins[256];
    __shared__ u32 warp_sums[8];
    __shared__ u32 state[4];
    __shared__ u64 sort_keys[SORT_MAX_K];
    __shared__ float sort_vals[SORT_MAX_K];
    const float infinity = __int_as_float(0x7f800000);
    const int *partial_ids = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL_TOKENS));
    const float *partial_values = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIAL_VALUES));
    int *candidates = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_CANDIDATES));
    const auto unary = SEISMIC_OUT_UNARY;
    const u64 row = blockIdx.x;
    const u64 parts = (SEISMIC_DIM_V + 4095ull) / 4096ull;
    const u64 count = parts * SEISMIC_DIM_K;
    const u64 base = row * count;

    if (SEISMIC_DIM_K <= SORT_MAX_K) {
        const u32 k = static_cast<u32>(SEISMIC_DIM_K);
        if (count <= 4096ull) {
            float vals[16];
            u32 keys[16];
            int toks[16];
            unsigned short active = 0u;
#pragma unroll
            for (int slot = 0; slot < 16; ++slot) {
                const u64 i = static_cast<u64>(slot) * 256ull + threadIdx.x;
                if (i < count) {
                    const u64 at = base + i;
                    const int token = partial_ids[at];
                    if (token >= 0) {
                        const float value = partial_values[at];
                        const u32 key = draft_key(value);
                        if (key != 0u) {
                            vals[slot] = value;
                            keys[slot] = key;
                            toks[slot] = token;
                            active |= static_cast<unsigned short>(1u << slot);
                            continue;
                        }
                    }
                }
                vals[slot] = 0.0f;
                keys[slot] = 0u;
                toks[slot] = -1;
            }
            const unsigned short picked = select_tile_topk_mask(
                keys, active, k,
                [&](int idx) { return toks[idx]; },
                bins, warp_sums, state);
            u32 total_picked = 0u;
            u32 pos = block_exclusive_add_256(static_cast<u32>(__popc(picked)), warp_sums, total_picked);
            u32 m = picked;
            while (m != 0u) {
                const int idx = __ffs(m) - 1;
                m &= m - 1u;
                if (pos < k) {
                    sort_keys[pos] = pack_sort_key(keys[idx], toks[idx]);
                    sort_vals[pos] = vals[idx];
                }
                ++pos;
            }
            const u32 written = total_picked < k ? total_picked : k;
            bitonic_sort_and_emit(
                sort_keys, sort_vals, written, k, row, infinity,
                candidates, SEISMIC_CANDIDATES_STRIDE_0, SEISMIC_CANDIDATES_STRIDE_1,
                unary, SEISMIC_UNARY_STRIDE_0, SEISMIC_UNARY_STRIDE_1);
            return;
        }
        const u32 written = stream_radix_select(
            count, k,
            [&](u64 i, float &val, u32 &key, int &token) {
                const u64 at = base + i;
                token = partial_ids[at];
                if (token < 0) return false;
                val = partial_values[at];
                key = draft_key(val);
                return key != 0u;
            },
            bins, warp_sums, state, sort_keys, sort_vals);
        bitonic_sort_and_emit(
            sort_keys, sort_vals, written, k, row, infinity,
            candidates, SEISMIC_CANDIDATES_STRIDE_0, SEISMIC_CANDIDATES_STRIDE_1,
            unary, SEISMIC_UNARY_STRIDE_0, SEISMIC_UNARY_STRIDE_1);
        return;
    }

    // Fallback for K > SORT_MAX_K.
    __shared__ reduce::Candidate partials[32];
    float previous = infinity;
    int previous_token = -1;
    for (u64 rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -infinity;
        int best_token = reduce::LowerIndex::none;
        for (u64 i = threadIdx.x; i < count; i += 256ull) {
            const u64 at = base + i;
            const int token = partial_ids[at];
            if (token < 0) continue;
            consider_streaming(partial_values[at], token, previous, previous_token, best, best_token);
        }
        const reduce::Candidate winner = reduce::group_argmax<reduce::LowerIndex>(
            reduce::Candidate{best, best_token}, partials);
        previous = winner.score;
        previous_token = winner.index;
        if (threadIdx.x == 0) {
            const u64 at = row * SEISMIC_DIM_K + rank;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0] = previous_token;
            candidates[at * SEISMIC_CANDIDATES_STRIDE_0 + SEISMIC_CANDIDATES_STRIDE_1] = 0;
            unary.put(row * SEISMIC_UNARY_STRIDE_0 + rank * SEISMIC_UNARY_STRIDE_1, previous);
        }
    }
}
