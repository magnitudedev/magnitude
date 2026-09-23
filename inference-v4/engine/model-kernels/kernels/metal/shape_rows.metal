inline ulong offset2(ulong row, ulong column, ulong stride0, ulong stride1) {
    return row * stride0 + column * stride1;
}

inline float load_f32(device const uchar *base, ulong offset) {
    return reinterpret_cast<device const float *>(base)[offset];
}

inline int load_i32(device const uchar *base, ulong offset) {
    return reinterpret_cast<device const int *>(base)[offset];
}

inline uint ordered_float_key(float value) {
    // Numeric equality treats both zero encodings as one tie.
    if (value == 0.0f) value = 0.0f;
    uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
}

inline uint group_sum_uint(uint value, threadgroup uint *partials,
    threadgroup uint &result, uint lane, uint simdgroup, uint simdgroups) {
    uint partial = simd_sum(value);
    if (lane == 0) partials[simdgroup] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        uint combined = lane < simdgroups ? partials[lane] : 0u;
        combined = simd_sum(combined);
        if (lane == 0) result = combined;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

inline float group_sum_float(float value, threadgroup float *partials,
    threadgroup float &result, uint lane, uint simdgroup, uint simdgroups) {
    float partial = simd_sum(value);
    if (lane == 0) partials[simdgroup] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        float combined = lane < simdgroups ? partials[lane] : 0.0f;
        combined = simd_sum(combined);
        if (lane == 0) result = combined;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

inline float group_max_float(float value, threadgroup float *partials,
    threadgroup float &result, uint lane, uint simdgroup, uint simdgroups) {
    float partial = simd_max(value);
    if (lane == 0) partials[simdgroup] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        float combined = lane < simdgroups ? partials[lane] : -INFINITY;
        combined = simd_max(combined);
        if (lane == 0) result = combined;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

kernel void shape_rows(
    device const uchar *logits [[buffer(SEISMIC_BUFFER_LOGITS)]],
    device const uchar *params [[buffer(SEISMIC_BUFFER_PARAMS)]],
    device const uchar *history [[buffer(SEISMIC_BUFFER_HISTORY)]],
    device uchar *out [[buffer(SEISMIC_BUFFER_OUT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]])
{
#if !defined(SEISMIC_LOGITS_REPRESENTATION_F32) || !defined(SEISMIC_PARAMS_REPRESENTATION_F32) || !defined(SEISMIC_HISTORY_REPRESENTATION_I32) || !defined(SEISMIC_OUT_REPRESENTATION_F32)
#error "shape_rows requires f32 logits/params/output and i32 history"
#endif
    ulong row = ulong(row_index);
    if (row >= SEISMIC_DIM_SX) return;
    uint simdgroups = 256u / simd_width;
    threadgroup uint uint_partials[32];
    threadgroup float float_partials[32];
    threadgroup uint reduced_uint;
    threadgroup float reduced_float;

    float temperature = load_f32(params, offset2(row, 0, SEISMIC_PARAMS_STRIDE_0, SEISMIC_PARAMS_STRIDE_1));
    int top_k = int(load_f32(params, offset2(row, 1, SEISMIC_PARAMS_STRIDE_0, SEISMIC_PARAMS_STRIDE_1)));
    float top_p = load_f32(params, offset2(row, 2, SEISMIC_PARAMS_STRIDE_0, SEISMIC_PARAMS_STRIDE_1));
    float min_p = load_f32(params, offset2(row, 3, SEISMIC_PARAMS_STRIDE_0, SEISMIC_PARAMS_STRIDE_1));
    float repetition = load_f32(params, offset2(row, 4, SEISMIC_PARAMS_STRIDE_0, SEISMIC_PARAMS_STRIDE_1));
    float presence = load_f32(params, offset2(row, 5, SEISMIC_PARAMS_STRIDE_0, SEISMIC_PARAMS_STRIDE_1));
    float frequency = load_f32(params, offset2(row, 6, SEISMIC_PARAMS_STRIDE_0, SEISMIC_PARAMS_STRIDE_1));

    // The output row is temporary integer storage until every history
    // occurrence has been counted. This changes penalty work from V*Hn to V+Hn.
    device atomic_uint *counts = reinterpret_cast<device atomic_uint *>(out);
    for (ulong token = ulong(tid); token < SEISMIC_DIM_V; token += 256ul) {
        atomic_store_explicit(&counts[offset2(row, token,
            SEISMIC_OUT_STRIDE_0, SEISMIC_OUT_STRIDE_1)], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (ulong h = ulong(tid); h < SEISMIC_DIM_HN; h += 256ul) {
        int token = load_i32(history, offset2(row, h,
            SEISMIC_HISTORY_STRIDE_0, SEISMIC_HISTORY_STRIDE_1));
        if (token >= 0 && ulong(token) < SEISMIC_DIM_V) {
            atomic_fetch_add_explicit(&counts[offset2(row, ulong(token),
                SEISMIC_OUT_STRIDE_0, SEISMIC_OUT_STRIDE_1)], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    device float *shaped = reinterpret_cast<device float *>(out);
    for (ulong token = ulong(tid); token < SEISMIC_DIM_V; token += 256ul) {
        ulong output_index = offset2(row, token, SEISMIC_OUT_STRIDE_0, SEISMIC_OUT_STRIDE_1);
        uint count = atomic_load_explicit(&counts[output_index], memory_order_relaxed);
        float value = load_f32(logits, offset2(row, token,
            SEISMIC_LOGITS_STRIDE_0, SEISMIC_LOGITS_STRIDE_1));
        if (count != 0u) {
            value = value < 0.0f ? value * repetition : value / repetition;
            value -= presence + frequency * float(count);
        }
        shaped[output_index] = temperature == 0.0f ? value : value / temperature;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (temperature == 0.0f) return;

    // Select the kth-largest IEEE value in 32 cooperative radix passes. Values
    // tied at that cutoff are retained, matching the semantic rank predicate.
    if (top_k > 0 && ulong(top_k) < SEISMIC_DIM_V) {
        uint prefix = 0u;
        uint mask = 0u;
        uint rank = uint(top_k);
        for (int bit_index = 31; bit_index >= 0; --bit_index) {
            uint bit = 1u << uint(bit_index);
            uint local = 0u;
            for (ulong token = ulong(tid); token < SEISMIC_DIM_V; token += 256ul) {
                float value = shaped[offset2(row, token,
                    SEISMIC_OUT_STRIDE_0, SEISMIC_OUT_STRIDE_1)];
                uint key = ordered_float_key(value);
                local += value > -INFINITY && value < INFINITY
                    && (key & mask) == prefix && (key & bit) != 0u;
            }
            uint ones = group_sum_uint(local, uint_partials, reduced_uint,
                lane, simdgroup, simdgroups);
            mask |= bit;
            if (rank <= ones) {
                prefix |= bit;
            } else {
                rank -= ones;
            }
        }
        uint cutoff_key = prefix;
        for (ulong token = ulong(tid); token < SEISMIC_DIM_V; token += 256ul) {
            ulong output_index = offset2(row, token,
                SEISMIC_OUT_STRIDE_0, SEISMIC_OUT_STRIDE_1);
            float value = shaped[output_index];
            if (value > -INFINITY && value < INFINITY
                && ordered_float_key(value) < cutoff_key) {
                shaped[output_index] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    float local_maximum = -INFINITY;
    for (ulong token = ulong(tid); token < SEISMIC_DIM_V; token += 256ul) {
        float value = shaped[offset2(row, token,
            SEISMIC_OUT_STRIDE_0, SEISMIC_OUT_STRIDE_1)];
        if (value > -INFINITY && value < INFINITY)
            local_maximum = metal::max(local_maximum, value);
    }
    float maximum = group_max_float(local_maximum, float_partials, reduced_float,
        lane, simdgroup, simdgroups);
    for (ulong token = ulong(tid); token < SEISMIC_DIM_V; token += 256ul) {
        ulong output_index = offset2(row, token,
            SEISMIC_OUT_STRIDE_0, SEISMIC_OUT_STRIDE_1);
        float value = shaped[output_index];
        if (value > -INFINITY && value < INFINITY
            && min_p > 0.0f && metal::exp(value - maximum) < min_p) {
            shaped[output_index] = -INFINITY;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (top_p >= 1.0f) return;

    local_maximum = -INFINITY;
    for (ulong token = ulong(tid); token < SEISMIC_DIM_V; token += 256ul) {
        float value = shaped[offset2(row, token,
            SEISMIC_OUT_STRIDE_0, SEISMIC_OUT_STRIDE_1)];
        if (value > -INFINITY && value < INFINITY)
            local_maximum = metal::max(local_maximum, value);
    }
    float final_maximum = group_max_float(local_maximum, float_partials, reduced_float,
        lane, simdgroup, simdgroups);
    float local_denominator = 0.0f;
    for (ulong token = ulong(tid); token < SEISMIC_DIM_V; token += 256ul) {
        float value = shaped[offset2(row, token,
            SEISMIC_OUT_STRIDE_0, SEISMIC_OUT_STRIDE_1)];
        if (value > -INFINITY && value < INFINITY)
            local_denominator += metal::exp(value - final_maximum);
    }
    float denominator = group_sum_float(local_denominator, float_partials, reduced_float,
        lane, simdgroup, simdgroups);
    if (denominator == 0.0f) return;
    float target = top_p * denominator;

    // Weighted radix selection finds the score bucket containing the nucleus
    // boundary without sorting V values or comparing every pair.
    uint cutoff_key = 0u;
    uint mask = 0u;
    float preceding = 0.0f;
    for (int bit_index = 31; bit_index >= 0; --bit_index) {
        uint bit = 1u << uint(bit_index);
        float local_mass = 0.0f;
        for (ulong token = ulong(tid); token < SEISMIC_DIM_V; token += 256ul) {
            float value = shaped[offset2(row, token,
                SEISMIC_OUT_STRIDE_0, SEISMIC_OUT_STRIDE_1)];
            uint key = ordered_float_key(value);
            if (value > -INFINITY && value < INFINITY
                && (key & mask) == cutoff_key && (key & bit) != 0u) {
                local_mass += metal::exp(value - final_maximum);
            }
        }
        float high_mass = group_sum_float(local_mass, float_partials, reduced_float,
            lane, simdgroup, simdgroups);
        mask |= bit;
        if (preceding + high_mass >= target) {
            cutoff_key |= bit;
        } else {
            preceding += high_mass;
        }
    }

    // Equal scores use ascending token index. A second radix walk over token
    // indices identifies the first tied token whose inclusive mass reaches p.
    uint cutoff_token = 0u;
    uint token_mask = 0u;
    for (int bit_index = 31; bit_index >= 0; --bit_index) {
        uint bit = 1u << uint(bit_index);
        float local_mass = 0.0f;
        for (ulong token = ulong(tid); token < SEISMIC_DIM_V; token += 256ul) {
            float value = shaped[offset2(row, token,
                SEISMIC_OUT_STRIDE_0, SEISMIC_OUT_STRIDE_1)];
            uint token_u32 = uint(token);
            if (value > -INFINITY && value < INFINITY
                && ordered_float_key(value) == cutoff_key
                && (token_u32 & token_mask) == cutoff_token
                && (token_u32 & bit) == 0u) {
                local_mass += metal::exp(value - final_maximum);
            }
        }
        float low_mass = group_sum_float(local_mass, float_partials, reduced_float,
            lane, simdgroup, simdgroups);
        token_mask |= bit;
        if (preceding + low_mass < target) {
            preceding += low_mass;
            cutoff_token |= bit;
        }
    }

    for (ulong token = ulong(tid); token < SEISMIC_DIM_V; token += 256ul) {
        ulong output_index = offset2(row, token,
            SEISMIC_OUT_STRIDE_0, SEISMIC_OUT_STRIDE_1);
        uint key = ordered_float_key(shaped[output_index]);
        float value = shaped[output_index];
        if (value > -INFINITY && value < INFINITY
            && (key < cutoff_key || (key == cutoff_key && token > ulong(cutoff_token)))) {
            shaped[output_index] = -INFINITY;
        }
    }
}
