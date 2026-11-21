// shape_rows: penalties and temperature, then top-k (ties at the cutoff kept),
// min-p against the row maximum, and top-p over the survivors in
// (value descending, token ascending) order.
//   shape_rows_prepare (PARTS x Sx): penalized/tempered values in place over
//     `logits` (-inf where a constrained row's mask rejects a finite value;
//     writeback gated on modification) and the partition maxima.
//   shape_rows_select (Sx): the cutoffs in one 1024-thread threadgroup per row.
//     Top-k is a count-weighted radix select (with a bitonic-sort compact fast
//     path when candidates fit in COMPACT_CAP); top-p is solved directly on the
//     sorted compact set or via a mass-weighted 64-bit radix select and tie
//     rank in ascending token order. Masses are exp(v - max) in 2^-32 fixed
//     point, so every sum is order-independent and exact.
//   shape_rows_apply (PARTS x Sx): writes -inf over every removed token.
// A zero temperature leaves the penalized values unshaped, as does a tempered
// NaN or +inf anywhere in the row.

#include <seismic/element.h>
#include "lib/core/order.h"

constant constexpr uint shape_threads = 256u;
constant constexpr uint shape_select_threads = 1024u;
constant constexpr uint shape_bins = 2048u;
constant constexpr uint shape_compact_cap = 2048u;
constant constexpr uint shape_count_cap = 4096u;

// Partition "maximum" of a partition holding NaN or +inf, above every finite
// key: the row is left unshaped (sampling reports it).
constant constexpr uint shape_unshaped = 0xFFFFFFFFu;

// Scratch per row: PARTS partition-maximum keys, then the selection state.
struct Selection {
    uint maximum;     // ordered key of the row maximum
    uint top_k_key;   // keys below are removed (0: none)
    uint top_p_key;   // removed: key below, or equal with token above top_p_token
    uint top_p_token;
    uint top_p_mode;  // 0 inactive, 1 cutoff, 2 every token removed
};

inline bool finite_f32(float value) { return value > -INFINITY && value < INFINITY; }

inline device uint *row_scratch(device uint *states, uint row) {
    return states + ulong(row) * (SEISMIC_TUNE_PARTS + 8u);
}

inline ulong simd_shuffle_up_u64(ulong v, ushort delta) {
    const uint lo = simd_shuffle_up(uint(v), delta);
    const uint hi = simd_shuffle_up(uint(v >> 32), delta);
    return (ulong(hi) << 32) | ulong(lo);
}

inline ulong simd_shuffle_xor_u64(ulong v, ushort mask) {
    const uint lo = simd_shuffle_xor(uint(v), mask);
    const uint hi = simd_shuffle_xor(uint(v >> 32), mask);
    return (ulong(hi) << 32) | ulong(lo);
}

// ceil(top_p * total) exactly: top_p = m * 2^-shift (m a 24-bit integer), so
// the product is a 128-bit integer (high, low) shifted right with rounding
// up. A token survives top-p when the mass before it is below this target
// (the portable `preceding < top_p`), bit for bit as on CUDA and Vulkan.
inline ulong shape_top_p_target(float top_p, ulong total) {
    if (!(top_p > 0.0f) || total == 0ul)
        return 0ul;
    const uint bits = as_type<uint>(top_p);
    const uint biased = bits >> 23;
    const ulong m = biased == 0u ? ulong(bits & 0x7fffffu) : ulong((bits & 0x7fffffu) | 0x800000u);
    const int shift = biased == 0u ? 149 : 150 - int(biased);
    const ulong low_part = m * (total & 0xfffffffful);
    const ulong mid = m * (total >> 32);
    const ulong low = low_part + (mid << 32);
    const ulong high = (mid >> 32) + (low < low_part ? 1ul : 0ul);
    if (shift <= 0)
        return low << uint(-shift);
    if (shift >= 128)
        return 1ul;
    ulong quotient;
    bool remainder;
    if (shift >= 64) {
        quotient = shift == 64 ? high : high >> uint(shift - 64);
        remainder = low != 0ul || (shift > 64 && (high & ((1ul << uint(shift - 64)) - 1ul)) != 0ul);
    } else {
        quotient = (low >> uint(shift)) | (high << uint(64 - shift));
        remainder = (low & ((1ul << uint(shift)) - 1ul)) != 0ul;
    }
    return quotient + (remainder ? 1ul : 0ul);
}

#define SHAPE_ARGUMENTS                                                                 \
    SEISMIC_OUT_LOGITS_PARAM(logits_cells),                                             \
    device const uint *mask [[buffer(SEISMIC_BUFFER_MASK)]],                            \
    device const int *constrained [[buffer(SEISMIC_BUFFER_CONSTRAINED)]],               \
    device const float *params [[buffer(SEISMIC_BUFFER_PARAMS)]],                       \
    device const int *history [[buffer(SEISMIC_BUFFER_HISTORY)]],                       \
    device uint *states [[buffer(SEISMIC_BUFFER_SCRATCH_STATE)]],                       \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define SHAPE_PARTITION                                                                 \
    const uint vocabulary = uint(SEISMIC_DIM_V);                                        \
    const uint span = (vocabulary + SEISMIC_TUNE_PARTS - 1u) / SEISMIC_TUNE_PARTS;      \
    const uint begin = metal::min(part * span, vocabulary);                             \
    const uint end = metal::min(begin + span, vocabulary);                              \
    const element::Out<element::F32> values =                                           \
        element::Out<element::F32>(logits_cells).offset(ulong(row) * SEISMIC_LOGITS_STRIDE_0)

kernel void shape_rows_prepare(SHAPE_ARGUMENTS,
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    // 4096 * 4 + 8 * 4 = 16416 bytes <= 32768 bytes threadgroup ceiling.
    threadgroup atomic_uint counts[shape_count_cap];
    threadgroup uint simd_maxima[shape_threads / 32u];
    const uint part = group.x, row = group.y;
    SHAPE_PARTITION;
    const uint slice_len = end - begin;
    device const float *p = params + ulong(row) * SEISMIC_PARAMS_STRIDE_0;
    const ulong p1 = SEISMIC_PARAMS_STRIDE_1;
    const float temperature = p[0];
    const float repetition = p[4u * p1];
    const float presence = p[5u * p1];
    const float frequency = p[6u * p1];
    const uint hn = uint(SEISMIC_DIM_HN);
    device const int *row_history = history + ulong(row) * SEISMIC_HISTORY_STRIDE_0;
    const bool penalized = hn > 0u && (repetition != 1.0f || presence != 0.0f || frequency != 0.0f);
    const bool use_counts = penalized && slice_len <= shape_count_cap;
    if (use_counts) {
        for (uint i = thread_index; i < slice_len; i += shape_threads)
            atomic_store_explicit(&counts[i], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint h = thread_index; h < hn; h += shape_threads) {
            const int tok = row_history[ulong(h) * SEISMIC_HISTORY_STRIDE_1];
            if (tok >= 0) {
                const uint u = uint(tok);
                if (u >= begin && u < end)
                    atomic_fetch_add_explicit(&counts[u - begin], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // A constrained row removes every rejected value but NaN and +inf, which
    // keep the row unshaped for sampling to report.
    const bool masked = constrained[ulong(row) * SEISMIC_CONSTRAINED_STRIDE_0] != 0;
    device const uint *words = mask + ulong(row) * SEISMIC_MASK_STRIDE_0;
    const bool scale_temp = temperature != 0.0f && temperature != 1.0f;
    const bool write_back = penalized || masked || scale_temp;
    uint maximum = 0u;
    for (uint token = begin + thread_index; token < end; token += shape_threads) {
        const ulong slot = ulong(token) * SEISMIC_LOGITS_STRIDE_1;
        float value = values.at(slot);
        if (penalized) {
            uint count = 0u;
            if (use_counts) {
                count = atomic_load_explicit(&counts[token - begin], memory_order_relaxed);
            } else {
                for (uint h = 0u; h < hn; ++h)
                    count += uint(row_history[ulong(h) * SEISMIC_HISTORY_STRIDE_1] == int(token));
            }
            if (count > 0u) {
                value = value < 0.0f ? value * repetition : value / repetition;
                value = value - presence - frequency * float(count);
            }
        }
        if (masked && ((words[ulong(token / 32u) * SEISMIC_MASK_STRIDE_1] >> (token % 32u)) & 1u) == 0u
            && value < INFINITY)
            value = -INFINITY;
        if (scale_temp)
            value = value / temperature;
        if (write_back)
            values.put(slot, value);
        if (finite_f32(value))
            maximum = metal::max(maximum, order::finite_key(value));
        else if (value != -INFINITY)
            maximum = shape_unshaped;
    }
    maximum = simd_max(maximum);
    if (lane == 0u)
        simd_maxima[sg] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < 32u) {
        maximum = thread_index < shape_threads / 32u ? simd_maxima[thread_index] : 0u;
        maximum = simd_max(maximum);
        if (thread_index == 0u)
            row_scratch(states, row)[part] = maximum;
    }
}

// ---------------------------------------------------------------------------
// Selection (one threadgroup of shape_select_threads = 1024 per row).

struct Survivor {
    element::Out<element::F32> line;
    ulong stride;
    float maximum;
    uint top_k_key;
    float min_p;

    inline bool at(ulong token, thread float &value, thread uint &key) const {
        value = line.at(token * stride);
        if (!finite_f32(value))
            return false;
        key = order::finite_key(value);
        return key >= top_k_key && !(min_p > 0.0f && metal::exp(value - maximum) < min_p);
    }
    inline bool at_mass(ulong token, thread uint &key, thread ulong &mass_out) const {
        const float value = line.at(token * stride);
        if (!finite_f32(value))
            return false;
        key = order::finite_key(value);
        if (key < top_k_key)
            return false;
        const float e = metal::exp(value - maximum);
        if (min_p > 0.0f && e < min_p)
            return false;
        mass_out = e >= 1.0f ? (1ul << 32) : ulong(e * 4294967296.0f);
        return true;
    }
    inline ulong mass(float value) const {
        const float e = metal::exp(value - maximum);
        return e >= 1.0f ? (1ul << 32) : ulong(e * 4294967296.0f);
    }
};

inline bool comes_before(uint ka, uint ta, uint kb, uint tb) {
    return ka > kb || (ka == kb && ta < tb);
}

inline void hist_clear(threadgroup uint *hist_lo, threadgroup uint *hist_hi, uint bins, uint tid) {
    for (uint bin = tid; bin < bins; bin += shape_select_threads) {
        hist_lo[bin] = 0u;
        hist_hi[bin] = 0u;
    }
}

inline ulong hist_get(threadgroup const uint *hist_lo, threadgroup const uint *hist_hi, uint bin) {
    return (ulong(hist_hi[bin]) << 32) | ulong(hist_lo[bin]);
}

// One threadgroup atomic per distinct bin across a simdgroup (uniform control flow).
inline void hist_add_count_simd(threadgroup uint *hist_lo, uint bin, bool active, uint lane) {
    uint waiting = uint(uint64_t(simd_ballot(active)));
    while (waiting != 0u) {
        const uint leader = metal::ctz(waiting);
        const uint leading = simd_shuffle(bin, ushort(leader));
        const uint same = uint(uint64_t(simd_ballot(active && bin == leading)));
        if (lane == leader)
            atomic_fetch_add_explicit(
                reinterpret_cast<threadgroup atomic_uint *>(&hist_lo[leading]),
                metal::popcount(same), memory_order_relaxed);
        active = active && bin != leading;
        waiting &= ~same;
    }
}

inline void hist_add_u64(threadgroup uint *hist_lo, threadgroup uint *hist_hi, uint bin, ulong w) {
    const uint low = uint(w);
    uint high = uint(w >> 32);
    const uint old = atomic_fetch_add_explicit(
        reinterpret_cast<threadgroup atomic_uint *>(&hist_lo[bin]), low, memory_order_relaxed);
    high += uint(old + low < old);
    if (high != 0u)
        atomic_fetch_add_explicit(
            reinterpret_cast<threadgroup atomic_uint *>(&hist_hi[bin]), high, memory_order_relaxed);
}

// Parallel bitonic sort of `count <= shape_compact_cap` (key, token) pairs in
// threadgroup memory in (key descending, token ascending) order.
inline void bitonic_sort_compact(uint count, threadgroup uint *keys, threadgroup uint *tokens, uint tid) {
    if (count <= 1u)
        return;
    uint n = 1u;
    while (n < count)
        n <<= 1;
    for (uint i = count + tid; i < n; i += shape_select_threads) {
        keys[i] = 0u;
        tokens[i] = 0xFFFFFFFFu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_n = n >> 1;
    for (uint k = 2u; k <= n; k <<= 1) {
        for (uint j = k >> 1; j > 0u; j >>= 1) {
            if (tid < half_n) {
                const uint low_mask = j - 1u;
                const uint i = ((tid & ~low_mask) << 1) | (tid & low_mask);
                const uint p = i | j;
                const uint ki = keys[i];
                const uint ti = tokens[i];
                const uint kp = keys[p];
                const uint tp = tokens[p];
                const bool swap = (i & k) == 0u ? comes_before(kp, tp, ki, ti) : comes_before(ki, ti, kp, tp);
                if (swap) {
                    keys[i] = kp;
                    tokens[i] = tp;
                    keys[p] = ki;
                    tokens[p] = ti;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Top-p cutoff directly from `count <= shape_compact_cap` candidates sorted in
// (key descending, token ascending) order.
inline void solve_top_p_sorted(uint count, thread const Survivor &survivor, float top_p,
                               threadgroup const uint *keys, threadgroup const uint *tokens,
                               threadgroup ulong *reduce, threadgroup uint &shared_key,
                               device Selection *selection, thread uint &mode, thread uint &top_p_key,
                               uint tid, uint sg, uint lane) {
    const uint i0 = 2u * tid;
    const uint i1 = i0 + 1u;
    ulong m0 = 0ul, m1 = 0ul;
    if (i0 < count) {
        const uint k = keys[i0];
        if (k >= survivor.top_k_key) {
            const float e = metal::exp(order::value(k) - survivor.maximum);
            if (!(survivor.min_p > 0.0f && e < survivor.min_p))
                m0 = e >= 1.0f ? (1ul << 32) : ulong(e * 4294967296.0f);
        }
    }
    if (i1 < count) {
        const uint k = keys[i1];
        if (k >= survivor.top_k_key) {
            const float e = metal::exp(order::value(k) - survivor.maximum);
            if (!(survivor.min_p > 0.0f && e < survivor.min_p))
                m1 = e >= 1.0f ? (1ul << 32) : ulong(e * 4294967296.0f);
        }
    }
    const ulong thread_sum = m0 + m1;
    ulong warp_inc = thread_sum;
    for (ushort offset = 1; offset < 32; offset <<= 1) {
        const ulong n = simd_shuffle_up_u64(warp_inc, offset);
        if (lane >= uint(offset))
            warp_inc += n;
    }
    if (lane == 31u)
        reduce[sg] = warp_inc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        ulong wv = reduce[lane];
        for (ushort offset = 1; offset < 32; offset <<= 1) {
            const ulong n = simd_shuffle_up_u64(wv, offset);
            if (lane >= uint(offset))
                wv += n;
        }
        reduce[lane] = wv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong total = reduce[31];
    const ulong target = shape_top_p_target(top_p, total);
    if (target == 0ul) {
        mode = 2u;
        return;
    }
    const ulong warp_base = sg > 0u ? reduce[sg - 1u] : 0ul;
    const ulong excl_i0 = warp_base + (warp_inc - thread_sum);
    const ulong incl_i0 = excl_i0 + m0;
    const ulong incl_i1 = incl_i0 + m1;
    if (i0 < count && excl_i0 < target && target <= incl_i0) {
        selection->top_p_token = tokens[i0];
        shared_key = keys[i0];
    } else if (i1 < count && incl_i0 < target && target <= incl_i1) {
        selection->top_p_token = tokens[i1];
        shared_key = keys[i1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    top_p_key = shared_key;
    mode = 1u;
}

// Descending digit of a weighted histogram at which the running weight
// `above` + bins reaches `target`; adds the weight of higher bins to `above`.
// Returns shape_bins when the whole histogram stays below the target.
inline uint descend(threadgroup const uint *hist_lo, threadgroup const uint *hist_hi,
                    uint bins, ulong target, thread ulong &above,
                    threadgroup ulong *lane_totals,
                    threadgroup uint &shared_digit, threadgroup ulong &shared_base,
                    uint tid) {
    // Simdgroup 0: lane l owns bins [bins - (l + 1) * per, bins - l * per), highest first.
    if (tid < 32u) {
        const uint lane = tid;
        const uint per = bins / 32u;
        ulong total = 0ul;
        for (uint i = 0u; i < per; ++i)
            total += hist_get(hist_lo, hist_hi, bins - 1u - (lane * per + i));
        lane_totals[lane] = total;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            ulong running = above;
            uint digit = shape_bins;
            for (uint l = 0u; l < 32u; ++l) {
                if (running + lane_totals[l] >= target) {
                    for (uint i = 0u; i < per; ++i) {
                        const uint bin = bins - 1u - (l * per + i);
                        const ulong h = hist_get(hist_lo, hist_hi, bin);
                        if (running + h >= target) {
                            digit = bin;
                            break;
                        }
                        running += h;
                    }
                    break;
                }
                running += lane_totals[l];
            }
            shared_digit = digit;
            shared_base = running;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    above = shared_base;
    return shared_digit;
}

// Radix select over the keys of survivors (weight 1 or mass) starting from
// `start_level` with established `prefix`/`mask`/`above`: finds the key K with
// weight(key > K) < target <= weight(key >= K). Returns false when the total
// weight is below the target; `above` receives weight(key > K).
template <bool MASS>
inline bool select_key_from(thread const Survivor &survivor, ulong vocabulary, ulong target,
                            int start_level, uint prefix, uint mask, thread ulong &above,
                            thread uint &key_out,
                            threadgroup uint *hist_lo, threadgroup uint *hist_hi,
                            threadgroup ulong *lane_totals,
                            threadgroup uint &shared_digit, threadgroup ulong &shared_base,
                            uint tid, uint lane) {
    const uint shifts[3] = {21u, 10u, 0u};
    const uint widths[3] = {11u, 11u, 10u};
    for (int level = start_level; level < 3; ++level) {
        const uint bins = 1u << widths[level];
        hist_clear(hist_lo, hist_hi, bins, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!MASS) {
            for (ulong base = 0ul; base < vocabulary; base += shape_select_threads) {
                const ulong token = base + tid;
                bool hit = false;
                uint bin = 0u;
                if (token < vocabulary) {
                    float value;
                    uint key;
                    if (survivor.at(token, value, key) && (key & mask) == prefix) {
                        hit = true;
                        bin = (key >> shifts[level]) & (bins - 1u);
                    }
                }
                hist_add_count_simd(hist_lo, bin, hit, lane);
            }
        } else {
            for (ulong token = tid; token < vocabulary; token += shape_select_threads) {
                uint key;
                ulong w;
                if (survivor.at_mass(token, key, w) && (key & mask) == prefix)
                    hist_add_u64(hist_lo, hist_hi, (key >> shifts[level]) & (bins - 1u), w);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint digit = descend(hist_lo, hist_hi, bins, target, above, lane_totals, shared_digit, shared_base, tid);
        if (digit == shape_bins)
            return false;
        prefix |= digit << shifts[level];
        mask |= (bins - 1u) << shifts[level];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    key_out = prefix;
    return true;
}

kernel void shape_rows_select(SHAPE_ARGUMENTS,
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    // Threadgroup memory: 8192 + 8192 + 256 + 256 + 4096 + 20 = 21012 bytes <= 32768 bytes.
    // buf_a / buf_b alias (hist_lo, hist_hi) during radix passes and (keys, tokens) during compact sort.
    threadgroup uint buf_a[shape_bins];
    threadgroup uint buf_b[shape_bins];
    threadgroup ulong lane_totals[32];
    threadgroup ulong reduce[32];
    threadgroup uint chunk_counts[shape_select_threads];
    threadgroup atomic_uint cand_count;
    threadgroup uint shared_digit;
    threadgroup ulong shared_base;
    threadgroup uint shared_key;

    device const float *p = params + ulong(row) * SEISMIC_PARAMS_STRIDE_0;
    const ulong p1 = SEISMIC_PARAMS_STRIDE_1;
    if (p[0] == 0.0f)
        return;

    device uint *scratch = row_scratch(states, row);
    device Selection *selection = reinterpret_cast<device Selection *>(scratch + SEISMIC_TUNE_PARTS);
    const int top_k = int(p[p1]);
    const float top_p = p[2u * p1];
    const float min_p = p[3u * p1];
    const ulong vocabulary = SEISMIC_DIM_V;

    uint maximum = 0u;
    for (uint part = 0u; part < SEISMIC_TUNE_PARTS; ++part)
        maximum = metal::max(maximum, scratch[part]);
    if (maximum == shape_unshaped) {
        if (tid == 0u)
            selection->maximum = shape_unshaped;
        return;
    }

    Survivor survivor{
        element::Out<element::F32>(logits_cells).offset(ulong(row) * SEISMIC_LOGITS_STRIDE_0),
        SEISMIC_LOGITS_STRIDE_1,
        order::value(maximum),
        0u,
        0.0f
    };

    uint top_k_key = 0u;
    ulong above = 0ul;
    bool compacted = false;
    uint compacted_count = 0u;
    if (top_k > 0 && ulong(top_k) < vocabulary && maximum != 0u) {
        hist_clear(buf_a, buf_b, shape_bins, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ulong base = 0ul; base < vocabulary; base += shape_select_threads) {
            const ulong token = base + tid;
            bool hit = false;
            uint bin = 0u;
            if (token < vocabulary) {
                const float value = survivor.line.at(token * survivor.stride);
                if (finite_f32(value)) {
                    hit = true;
                    bin = order::finite_key(value) >> 21;
                }
            }
            hist_add_count_simd(buf_a, bin, hit, lane);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint digit0 = descend(buf_a, buf_b, shape_bins, ulong(top_k), above, lane_totals,
                                    shared_digit, shared_base, tid);
        if (digit0 != shape_bins) {
            const ulong in_bin0 = hist_get(buf_a, buf_b, digit0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (above + in_bin0 <= shape_compact_cap) {
                if (tid == 0u)
                    atomic_store_explicit(&cand_count, 0u, memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (ulong token = tid; token < vocabulary; token += shape_select_threads) {
                    const float value = survivor.line.at(token * survivor.stride);
                    if (finite_f32(value)) {
                        const uint key = order::finite_key(value);
                        if ((key >> 21) >= digit0) {
                            const uint idx = atomic_fetch_add_explicit(&cand_count, 1u, memory_order_relaxed);
                            buf_a[idx] = key;
                            buf_b[idx] = uint(token);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                compacted_count = atomic_load_explicit(&cand_count, memory_order_relaxed);
                bitonic_sort_compact(compacted_count, buf_a, buf_b, tid);
                top_k_key = buf_a[top_k - 1];
                compacted = true;
            } else {
                uint key;
                if (select_key_from<false>(survivor, vocabulary, ulong(top_k), 1, digit0 << 21, 0x7FFu << 21,
                                           above, key, buf_a, buf_b, lane_totals, shared_digit, shared_base,
                                           tid, lane))
                    top_k_key = key;
            }
        }
    }

    survivor.top_k_key = top_k_key;
    survivor.min_p = min_p;
    uint mode = 0u, top_p_key = 0u;
    if (top_p < 1.0f && maximum != 0u) {
        if (compacted) {
            solve_top_p_sorted(compacted_count, survivor, top_p, buf_a, buf_b, reduce, shared_key,
                               selection, mode, top_p_key, tid, sg, lane);
        } else {
            if (tid == 0u)
                atomic_store_explicit(&cand_count, 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ulong total = 0ul;
            for (ulong token = tid; token < vocabulary; token += shape_select_threads) {
                uint key;
                ulong w;
                if (survivor.at_mass(token, key, w)) {
                    total += w;
                    if (atomic_load_explicit(&cand_count, memory_order_relaxed) <= shape_compact_cap) {
                        const uint idx = atomic_fetch_add_explicit(&cand_count, 1u, memory_order_relaxed);
                        if (idx < shape_compact_cap) {
                            buf_a[idx] = key;
                            buf_b[idx] = uint(token);
                        }
                    }
                }
            }
            for (ushort offset = 16; offset > 0; offset >>= 1)
                total += simd_shuffle_xor_u64(total, offset);
            if (lane == 0u)
                reduce[sg] = total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            total = 0ul;
            for (uint w = 0u; w < shape_select_threads / 32u; ++w)
                total += reduce[w];
            const uint count = atomic_load_explicit(&cand_count, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const ulong target = shape_top_p_target(top_p, total);
            if (target == 0ul) {
                mode = 2u;
            } else if (count <= shape_compact_cap) {
                bitonic_sort_compact(count, buf_a, buf_b, tid);
                solve_top_p_sorted(count, survivor, top_p, buf_a, buf_b, reduce, shared_key,
                                   selection, mode, top_p_key, tid, sg, lane);
            } else {
                uint key;
                above = 0ul;
                select_key_from<true>(survivor, vocabulary, target, 0, 0u, 0u, above, key,
                                      buf_a, buf_b, lane_totals, shared_digit, shared_base, tid, lane);
                // Rank among equal keys: the tied token whose inclusive mass
                // first reaches the target, counted in ascending token order.
                const ulong each = survivor.mass(order::value(key));
                // above < target <= above + ties * each, so each > 0 and rank < ties.
                const ulong rank = (target - above + each - 1ul) / each - 1ul;
                const ulong chunk = (vocabulary + shape_select_threads - 1ul) / shape_select_threads;
                const ulong first = ulong(tid) * chunk;
                uint tie_count = 0u;
                for (ulong token = first; token < first + chunk && token < vocabulary; ++token) {
                    float value;
                    uint candidate;
                    tie_count += survivor.at(token, value, candidate) && candidate == key ? 1u : 0u;
                }
                chunk_counts[tid] = tie_count;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                ulong before = 0ul;
                for (uint t = 0u; t < tid; ++t)
                    before += chunk_counts[t];
                if (rank >= before && rank < before + tie_count) {
                    ulong seen = before;
                    for (ulong token = first; token < first + chunk && token < vocabulary; ++token) {
                        float value;
                        uint candidate;
                        if (survivor.at(token, value, candidate) && candidate == key) {
                            if (seen == rank)
                                selection->top_p_token = uint(token);
                            ++seen;
                        }
                    }
                }
                mode = 1u;
                top_p_key = key;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        selection->maximum = maximum;
        selection->top_k_key = top_k_key;
        selection->top_p_key = top_p_key;
        selection->top_p_mode = mode;
    }
}

kernel void shape_rows_apply(SHAPE_ARGUMENTS,
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint part = group.x, row = group.y;
    SHAPE_PARTITION;
    device const float *p = params + ulong(row) * SEISMIC_PARAMS_STRIDE_0;
    const ulong p1 = SEISMIC_PARAMS_STRIDE_1;
    if (p[0] == 0.0f)
        return;
    const Selection selection =
        *reinterpret_cast<device const Selection *>(row_scratch(states, row) + SEISMIC_TUNE_PARTS);
    if (selection.maximum == shape_unshaped)
        return;
    const float min_p = p[3u * p1];
    if (selection.top_k_key == 0u && !(min_p > 0.0f) && selection.top_p_mode == 0u)
        return;
    const float maximum = order::value(selection.maximum);
    for (uint token = begin + thread_index; token < end; token += shape_threads) {
        const ulong slot = ulong(token) * SEISMIC_LOGITS_STRIDE_1;
        const float value = values.at(slot);
        if (!finite_f32(value))
            continue;
        const uint key = order::finite_key(value);
        const bool removed =
            key < selection.top_k_key || (min_p > 0.0f && metal::exp(value - maximum) < min_p) ||
            selection.top_p_mode == 2u ||
            (selection.top_p_mode == 1u &&
             (key < selection.top_p_key || (key == selection.top_p_key && token > selection.top_p_token)));
        if (removed)
            values.put(slot, -INFINITY);
    }
}
