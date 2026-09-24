// shape_rows (K5): penalties, temperature, top-k, min-p and top-p per row,
// with each row's vocabulary split into PARTS contiguous partitions.
//
// prepare (PARTS x rows): penalties and temperature into `out`; partition
//         maxima and non-finite flags.
// count   (PARTS x rows): a radix histogram of the active search's current
//         digit over the elements matching its prefix: counts for top-k,
//         counts and fixed-point masses exp(v - max) for top-p.
// select  (rows): merges the histogram and narrows the search by one digit
//         (11, 11, then 10 bits of the order-preserving key).
// Three count/select rounds find the top-k threshold (the k-th largest value;
// every tie of it is kept), three more the top-p cutoff over the tokens that
// survive top-k and min-p. At the exact cutoff key, the top-p tie rule
// (ascending token index) keeps the first `n` ties:
// ties    (PARTS x rows): per-partition counts of cutoff-key survivors.
// apply   (PARTS x rows): writes -inf over every removed token.
//
// Masses are exp(v - max) quantized to 2^-32 and summed as 64-bit integers,
// so every decision is independent of PARTS and of thread order. A row whose
// tempered values contain NaN or +inf, or whose maximum is -inf, is left
// unfiltered (sampling reports it as status 2 or 1 regardless).

constant constexpr uint shape_threads = 256;
constant constexpr uint shape_buckets = 2048;
constant constexpr uint shape_history_capacity = 2048;
constant constexpr uint shape_bitmap_bits = 32768;

// Row state words (scratch `state`, 16 words per row).
constant constexpr uint word_flags = 0;        // flag_* bits
constant constexpr uint word_step = 1;         // count/select round, 0..6
constant constexpr uint word_max = 2;          // row maximum (float bits)
constant constexpr uint word_min_p = 3;        // min-p (float bits)
constant constexpr uint word_top_p = 4;        // top-p (float bits)
constant constexpr uint word_topk_prefix = 5;
constant constexpr uint word_topk_need = 6;
constant constexpr uint word_topk_key = 7;     // keep key >= this
constant constexpr uint word_topp_prefix = 8;
constant constexpr uint word_topp_above_lo = 9;
constant constexpr uint word_topp_above_hi = 10;
constant constexpr uint word_den_lo = 11;
constant constexpr uint word_den_hi = 12;
constant constexpr uint word_topp_key = 13;    // keep key > this; key == this per the tie rule
constant constexpr uint word_topp_keep = 14;   // ties of word_topp_key kept (keep_all: every one)
constant constexpr uint shape_state_words = 16;

constant constexpr uint flag_topk = 1u, flag_minp = 2u, flag_topp = 4u, flag_skip = 8u;
constant constexpr uint flag_topk_done = 16u, flag_topp_done = 32u, flag_topp_none = 64u;
constant constexpr uint keep_all = 0xffffffffu;

inline uint shape_key(float value) {
    uint bits = as_type<uint>(value + 0.0f);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

inline float shape_value(uint key) {
    return as_type<float>((key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
}

inline uint digit_shift(uint level) { return level == 0 ? 21u : (level == 1 ? 10u : 0u); }
inline uint digit_mask(uint level) { return level == 2 ? 0x3ffu : 0x7ffu; }

inline bool shape_matches(uint key, uint level, uint prefix) {
    return level == 0 || (key >> (level == 1 ? 21u : 10u)) == prefix;
}

inline uint shape_narrow(uint prefix, uint bucket, uint level) {
    return level == 0 ? bucket : (prefix << (level == 1 ? 11u : 10u)) | bucket;
}

inline uint shape_lower_bound(uint prefix, uint bucket, uint level) {
    uint narrowed = shape_narrow(prefix, bucket, level);
    return level == 0 ? narrowed << 21 : (level == 1 ? narrowed << 10 : narrowed);
}

// exp(v - max) as a 64-bit 2^-32 fixed-point value.
inline ulong shape_weight(float value, float maximum) {
    float w = metal::exp(value - maximum);
    return w >= 1.0f ? (1ul << 32) : ulong(w * 4294967296.0f);
}

// ceil(top_p * total) exactly: top_p = m * 2^-shift (m a 24-bit integer), so
// the product is a 128-bit integer (high, low) shifted right with rounding
// up. A token survives top-p when the mass before it is below this target
// (the portable `preceding < top_p`), bit for bit as on CUDA and Vulkan.
inline ulong shape_top_p_target(float top_p, ulong total) {
    if (!(top_p > 0.0f) || total == 0)
        return 0;
    const uint bits = as_type<uint>(top_p);
    const uint biased = bits >> 23;
    const ulong m = biased == 0 ? ulong(bits & 0x7fffffu) : ulong((bits & 0x7fffffu) | 0x800000u);
    const int shift = biased == 0 ? 149 : 150 - int(biased);
    const ulong low_part = m * (total & 0xfffffffful);
    const ulong mid = m * (total >> 32);
    const ulong low = low_part + (mid << 32);
    const ulong high = (mid >> 32) + (low < low_part ? 1ul : 0ul);
    if (shift <= 0)
        return low << uint(-shift);
    if (shift >= 128)
        return 1;
    ulong quotient;
    bool remainder;
    if (shift >= 64) {
        quotient = shift == 64 ? high : high >> uint(shift - 64);
        remainder = low != 0 || (shift > 64 && (high & ((1ul << uint(shift - 64)) - 1ul)) != 0);
    } else {
        quotient = (low >> uint(shift)) | (high << uint(64 - shift));
        remainder = (low & ((1ul << uint(shift)) - 1ul)) != 0;
    }
    return quotient + (remainder ? 1ul : 0ul);
}

// Survival of top-k and min-p (the population top-p ranks).
inline bool shape_survives(uint key, float value, device const uint *state) {
    uint flags = state[word_flags];
    if ((flags & flag_topk) && key < state[word_topk_key])
        return false;
    if ((flags & flag_minp)
        && metal::exp(value - as_type<float>(state[word_max])) < as_type<float>(state[word_min_p]))
        return false;
    return true;
}

#define SHAPE_ARGUMENTS                                                                 \
    device const float *logits [[buffer(SEISMIC_BUFFER_LOGITS)]],                       \
    device const float *params [[buffer(SEISMIC_BUFFER_PARAMS)]],                       \
    device const int *history [[buffer(SEISMIC_BUFFER_HISTORY)]],                       \
    device float *out [[buffer(SEISMIC_BUFFER_OUT)]],                                   \
    device uint *states [[buffer(SEISMIC_BUFFER_SCRATCH_STATE)]],                       \
    device uint *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],                  \
    device uint *histograms [[buffer(SEISMIC_BUFFER_SCRATCH_HISTOGRAM)]],               \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

#define SHAPE_PARTITION                                                                 \
    const uint vocabulary = uint(SEISMIC_DIM_V);                                        \
    const uint span = (vocabulary + SEISMIC_TUNE_PARTS - 1) / SEISMIC_TUNE_PARTS;       \
    const uint begin = min(part * span, vocabulary), end = min(begin + span, vocabulary); \
    device float *values = out + ulong(row) * SEISMIC_OUT_STRIDE_0;                     \
    device uint *state = states + ulong(row) * shape_state_words

kernel void shape_rows_prepare(SHAPE_ARGUMENTS,
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup int recent[shape_history_capacity];
    threadgroup atomic_uint marked[shape_bitmap_bits / 32u];
    threadgroup float maxima[shape_threads / 32];
    threadgroup uint flags[shape_threads / 32];
    const uint part = group.x, row = group.y;
    SHAPE_PARTITION;
    device const float *p = params + ulong(row) * SEISMIC_PARAMS_STRIDE_0;
    const ulong p1 = SEISMIC_PARAMS_STRIDE_1;
    const float temperature = p[0], top_p = p[2 * p1], min_p = p[3 * p1];
    const int top_k = int(p[p1]);
    const float repetition = p[4 * p1], presence = p[5 * p1], frequency = p[6 * p1];
    if (part == 0) {
        if (thread_index == 0) {
            uint row_flags = 0;
            if (temperature != 0.0f) {
                row_flags |= top_k > 0 ? flag_topk : 0u;
                row_flags |= min_p > 0.0f ? flag_minp : 0u;
                row_flags |= top_p < 1.0f ? flag_topp : 0u;
            }
            state[word_flags] = row_flags;
            state[word_step] = 0;
            state[word_min_p] = as_type<uint>(min_p);
            state[word_top_p] = as_type<uint>(top_p);
            state[word_topk_prefix] = 0;
            state[word_topk_need] = uint(max(top_k, 0));
            state[word_topk_key] = 0;
            state[word_topp_prefix] = 0;
            state[word_topp_above_lo] = 0;
            state[word_topp_above_hi] = 0;
            state[word_den_lo] = 0;
            state[word_den_hi] = 0;
            state[word_topp_key] = 0;
            state[word_topp_keep] = keep_all;
        }
        device uint *histogram = histograms + ulong(row) * shape_buckets * 3u;
        for (uint i = thread_index; i < shape_buckets * 3u; i += shape_threads)
            histogram[i] = 0;
    }
    // History tokens inside this partition are marked in a bitmap, so only
    // marked tokens count their occurrences.
    const uint hn = uint(SEISMIC_DIM_HN);
    device const int *row_history = history + ulong(row) * SEISMIC_HISTORY_STRIDE_0;
    const bool staged = hn <= shape_history_capacity && span <= shape_bitmap_bits;
    if (staged) {
        for (uint w = thread_index; w < shape_bitmap_bits / 32u; w += shape_threads)
            atomic_store_explicit(&marked[w], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint h = thread_index; h < hn; h += shape_threads) {
            int token = row_history[ulong(h) * SEISMIC_HISTORY_STRIDE_1];
            recent[h] = token;
            if (token >= int(begin) && token < int(end)) {
                uint local = uint(token) - begin;
                atomic_fetch_or_explicit(&marked[local / 32u], 1u << (local % 32u), memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = -INFINITY;
    uint nonfinite = 0;
    for (uint token = begin + thread_index; token < end; token += shape_threads) {
        uint count = 0;
        if (staged) {
            uint local = token - begin;
            if ((atomic_load_explicit(&marked[local / 32u], memory_order_relaxed) >> (local % 32u)) & 1u)
                for (uint h = 0; h < hn; ++h)
                    count += uint(recent[h] == int(token));
        } else {
            for (uint h = 0; h < hn; ++h)
                count += uint(row_history[ulong(h) * SEISMIC_HISTORY_STRIDE_1] == int(token));
        }
        float value = logits[ulong(row) * SEISMIC_LOGITS_STRIDE_0 + ulong(token) * SEISMIC_LOGITS_STRIDE_1];
        if (count > 0) {
            value = value < 0.0f ? value * repetition : value / repetition;
            value = value - presence - frequency * float(count);
        }
        if (temperature != 0.0f)
            value = value / temperature;
        values[ulong(token) * SEISMIC_OUT_STRIDE_1] = value;
        maximum = metal::max(maximum, value);
        nonfinite |= uint(metal::isnan(value) || value == INFINITY);
    }
    maximum = simd_max(maximum);
    nonfinite = simd_max(nonfinite);
    if (lane == 0) {
        maxima[sg] = maximum;
        flags[sg] = nonfinite;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        for (uint i = 1; i < shape_threads / 32; ++i) {
            maximum = metal::max(maxima[0], maxima[i]);
            maxima[0] = maximum;
            flags[0] |= flags[i];
        }
        device uint *partial = partials + (ulong(row) * SEISMIC_TUNE_PARTS + part) * 3u;
        partial[0] = as_type<uint>(maxima[0]);
        partial[1] = flags[0];
    }
}

kernel void shape_rows_count(SHAPE_ARGUMENTS,
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint counts[shape_buckets];
    threadgroup atomic_uint mass_lo[shape_buckets];
    threadgroup atomic_uint mass_hi[shape_buckets];
    const uint part = group.x, row = group.y;
    SHAPE_PARTITION;
    const uint step = state[word_step];
    const uint row_flags = state[word_flags];
    const bool topp = step >= 3;
    const uint level = step % 3;
    if ((row_flags & flag_skip)
        || (!topp && (!(row_flags & flag_topk) || (row_flags & flag_topk_done)))
        || (topp && (!(row_flags & flag_topp) || (row_flags & (flag_topp_done | flag_topp_none)))))
        return;
    const uint prefix = state[topp ? word_topp_prefix : word_topk_prefix];
    const float maximum = as_type<float>(state[word_max]);
    for (uint b = thread_index; b < shape_buckets; b += shape_threads) {
        atomic_store_explicit(&counts[b], 0u, memory_order_relaxed);
        atomic_store_explicit(&mass_lo[b], 0u, memory_order_relaxed);
        atomic_store_explicit(&mass_hi[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint token = begin + thread_index; token < end; token += shape_threads) {
        float value = values[ulong(token) * SEISMIC_OUT_STRIDE_1];
        uint key = shape_key(value);
        if (!shape_matches(key, level, prefix))
            continue;
        if (topp && !shape_survives(key, value, state))
            continue;
        uint bucket = (key >> digit_shift(level)) & digit_mask(level);
        atomic_fetch_add_explicit(&counts[bucket], 1u, memory_order_relaxed);
        if (topp) {
            ulong w = shape_weight(value, maximum);
            uint low = uint(w), high = uint(w >> 32);
            uint old = atomic_fetch_add_explicit(&mass_lo[bucket], low, memory_order_relaxed);
            high += uint(old + low < old);
            if (high != 0)
                atomic_fetch_add_explicit(&mass_hi[bucket], high, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device atomic_uint *histogram =
        reinterpret_cast<device atomic_uint *>(histograms + ulong(row) * shape_buckets * 3u);
    for (uint b = thread_index; b < shape_buckets; b += shape_threads) {
        uint count = atomic_load_explicit(&counts[b], memory_order_relaxed);
        if (count == 0)
            continue;
        atomic_fetch_add_explicit(&histogram[3u * b], count, memory_order_relaxed);
        if (topp) {
            uint low = atomic_load_explicit(&mass_lo[b], memory_order_relaxed);
            uint high = atomic_load_explicit(&mass_hi[b], memory_order_relaxed);
            uint old = atomic_fetch_add_explicit(&histogram[3u * b + 1u], low, memory_order_relaxed);
            high += uint(old + low < old);
            if (high != 0)
                atomic_fetch_add_explicit(&histogram[3u * b + 2u], high, memory_order_relaxed);
        }
    }
}

kernel void shape_rows_select(SHAPE_ARGUMENTS,
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup uint local_counts[shape_threads];
    threadgroup ulong local_mass[shape_threads];
    threadgroup atomic_uint chosen;
    threadgroup uint chosen_count;
    threadgroup ulong chosen_before_mass;
    threadgroup uint chosen_before_count;
    threadgroup ulong total_mass;
    threadgroup uint total_count;
    device uint *state = states + ulong(row) * shape_state_words;
    device uint *histogram = histograms + ulong(row) * shape_buckets * 3u;
    const uint step = state[word_step];
    if (step == 0 && thread_index == 0) {
        float maximum = -INFINITY;
        uint nonfinite = 0;
        for (uint part = 0; part < SEISMIC_TUNE_PARTS; ++part) {
            device const uint *partial = partials + (ulong(row) * SEISMIC_TUNE_PARTS + part) * 3u;
            maximum = metal::max(maximum, as_type<float>(partial[0]));
            nonfinite |= partial[1];
        }
        state[word_max] = as_type<uint>(maximum);
        if (nonfinite != 0 || maximum == -INFINITY)
            state[word_flags] |= flag_skip;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    const uint row_flags = state[word_flags];
    const bool topp = step >= 3;
    const uint level = step % 3;
    const bool active = !(row_flags & flag_skip)
        && (topp ? ((row_flags & flag_topp) && !(row_flags & (flag_topp_done | flag_topp_none)))
                 : ((row_flags & flag_topk) && !(row_flags & flag_topk_done)));
    // Thread t owns buckets 2047 - 8t down to 2040 - 8t (descending key order).
    uint counts[8];
    ulong masses[8];
    uint my_count = 0;
    ulong my_mass = 0;
    for (uint i = 0; i < 8; ++i) {
        uint b = shape_buckets - 1u - (8u * thread_index + i);
        counts[i] = histogram[3u * b];
        masses[i] = (ulong(histogram[3u * b + 2u]) << 32) | ulong(histogram[3u * b + 1u]);
        histogram[3u * b] = 0;
        histogram[3u * b + 1u] = 0;
        histogram[3u * b + 2u] = 0;
        my_count += counts[i];
        my_mass += masses[i];
    }
    if (!active) {
        if (thread_index == 0)
            state[word_step] = step + 1;
        return;
    }
    local_counts[thread_index] = my_count;
    local_mass[thread_index] = my_mass;
    if (thread_index == 0)
        atomic_store_explicit(&chosen, 0xffffffffu, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        uint count = 0;
        ulong mass = 0;
        for (uint t = 0; t < shape_threads; ++t) {
            uint c = local_counts[t];
            ulong m = local_mass[t];
            local_counts[t] = count;
            local_mass[t] = mass;
            count += c;
            mass += m;
        }
        total_count = count;
        total_mass = mass;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before_count = local_counts[thread_index];
    ulong before_mass = local_mass[thread_index];
    if (!topp) {
        uint need = state[word_topk_need];
        for (uint i = 0; i < 8; ++i) {
            if (counts[i] != 0 && before_count < need && need <= before_count + counts[i]) {
                uint b = shape_buckets - 1u - (8u * thread_index + i);
                atomic_store_explicit(&chosen, b, memory_order_relaxed);
                chosen_count = counts[i];
                chosen_before_count = before_count;
            }
            before_count += counts[i];
        }
    } else {
        ulong den = level == 0 ? total_mass
            : ((ulong(state[word_den_hi]) << 32) | ulong(state[word_den_lo]));
        ulong above = (ulong(state[word_topp_above_hi]) << 32) | ulong(state[word_topp_above_lo]);
        const ulong target = shape_top_p_target(as_type<float>(state[word_top_p]), den);
        for (uint i = 0; i < 8; ++i) {
            if (counts[i] != 0 && above + before_mass < target) {
                uint b = shape_buckets - 1u - (8u * thread_index + i);
                atomic_fetch_min_explicit(&chosen, b, memory_order_relaxed);
            }
            before_mass += masses[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint b = atomic_load_explicit(&chosen, memory_order_relaxed);
        before_mass = local_mass[thread_index];
        for (uint i = 0; i < 8; ++i) {
            if (shape_buckets - 1u - (8u * thread_index + i) == b) {
                chosen_count = counts[i];
                chosen_before_mass = before_mass;
            }
            before_mass += masses[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index != 0)
        return;
    uint b = atomic_load_explicit(&chosen, memory_order_relaxed);
    uint flags_now = state[word_flags];
    if (!topp) {
        uint prefix = state[word_topk_prefix];
        uint need = state[word_topk_need];
        if (total_count < need || b == 0xffffffffu) {
            state[word_topk_key] = 0;
            flags_now |= flag_topk_done;
        } else if (chosen_before_count + chosen_count == need || level == 2) {
            state[word_topk_key] = shape_lower_bound(prefix, b, level);
            flags_now |= flag_topk_done;
        } else {
            state[word_topk_prefix] = shape_narrow(prefix, b, level);
            state[word_topk_need] = need - chosen_before_count;
        }
    } else {
        uint prefix = state[word_topp_prefix];
        ulong above = (ulong(state[word_topp_above_hi]) << 32) | ulong(state[word_topp_above_lo]);
        if (level == 0) {
            state[word_den_lo] = uint(total_mass);
            state[word_den_hi] = uint(total_mass >> 32);
        }
        ulong den = (ulong(state[word_den_hi]) << 32) | ulong(state[word_den_lo]);
        if (b == 0xffffffffu) {
            flags_now |= flag_topp_none;
        } else {
            ulong reached = above + chosen_before_mass;
            if (chosen_count == 1 || level == 2) {
                state[word_topp_key] = shape_lower_bound(prefix, b, level);
                uint keep = keep_all;
                if (level == 2 && chosen_count > 1) {
                    // Ties in ascending token order: the first whose
                    // inclusive mass reaches the target is the last kept
                    // (reached < target, as the bucket was chosen).
                    const ulong w = shape_weight(shape_value(shape_lower_bound(prefix, b, level)),
                        as_type<float>(state[word_max]));
                    const ulong target = shape_top_p_target(as_type<float>(state[word_top_p]), den);
                    const ulong n = w == 0 ? ulong(chosen_count) : (target - reached + w - 1ul) / w;
                    keep = n >= chosen_count ? keep_all : uint(max(n, 1ul));
                }
                state[word_topp_keep] = keep;
                flags_now |= flag_topp_done;
            } else {
                state[word_topp_prefix] = shape_narrow(prefix, b, level);
                state[word_topp_above_lo] = uint(reached);
                state[word_topp_above_hi] = uint(reached >> 32);
            }
        }
    }
    state[word_flags] = flags_now;
    state[word_step] = step + 1;
}

// Whether the top-p cutoff applies an index tie rule.
inline bool shape_tie_rule(uint row_flags, device const uint *state) {
    return (row_flags & flag_topp) && (row_flags & flag_topp_done) && !(row_flags & flag_skip)
        && state[word_topp_keep] != keep_all;
}

kernel void shape_rows_ties(SHAPE_ARGUMENTS,
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint totals[shape_threads / 32];
    const uint part = group.x, row = group.y;
    SHAPE_PARTITION;
    const uint row_flags = state[word_flags];
    if (!shape_tie_rule(row_flags, state))
        return;
    const uint cut = state[word_topp_key];
    uint count = 0;
    for (uint token = begin + thread_index; token < end; token += shape_threads) {
        float value = values[ulong(token) * SEISMIC_OUT_STRIDE_1];
        uint key = shape_key(value);
        count += uint(key == cut && shape_survives(key, value, state));
    }
    count = simd_sum(count);
    if (lane == 0)
        totals[sg] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        uint total = 0;
        for (uint i = 0; i < shape_threads / 32; ++i)
            total += totals[i];
        partials[(ulong(row) * SEISMIC_TUNE_PARTS + part) * 3u + 2u] = total;
    }
}

kernel void shape_rows_apply(SHAPE_ARGUMENTS,
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint totals[shape_threads / 32];
    const uint part = group.x, row = group.y;
    SHAPE_PARTITION;
    const uint row_flags = state[word_flags];
    if ((row_flags & flag_skip) || !(row_flags & (flag_topk | flag_minp | flag_topp)))
        return;
    const bool topp = row_flags & flag_topp;
    const bool none = row_flags & flag_topp_none;
    const uint cut = state[word_topp_key];
    const bool tie_rule = shape_tie_rule(row_flags, state);
    const uint keep = state[word_topp_keep];
    uint rank = 0;
    if (tie_rule)
        for (uint q = 0; q < part; ++q)
            rank += partials[(ulong(row) * SEISMIC_TUNE_PARTS + q) * 3u + 2u];
    for (uint base = begin; base < end; base += shape_threads) {
        uint token = base + thread_index;
        bool present = token < end;
        float value = present ? values[ulong(token) * SEISMIC_OUT_STRIDE_1] : 0.0f;
        uint key = shape_key(value);
        bool kept = present && shape_survives(key, value, state);
        bool tie = false;
        if (topp && kept) {
            if (none || key < cut)
                kept = false;
            else if (key == cut && tie_rule)
                tie = true;
        }
        if (tie_rule) {
            uint before = simd_prefix_exclusive_sum(uint(tie));
            uint in_group = simd_sum(uint(tie));
            if (lane == 0)
                totals[sg] = in_group;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint earlier = rank;
            for (uint i = 0; i < sg; ++i)
                earlier += totals[i];
            if (tie)
                kept = earlier + before < keep;
            for (uint i = sg; i < shape_threads / 32; ++i)
                earlier += totals[i];
            rank = earlier;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (present && !kept)
            values[ulong(token) * SEISMIC_OUT_STRIDE_1] = -INFINITY;
    }
}
