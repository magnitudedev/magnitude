// attention_rel_bias (contract and portable body in attention.seismic):
// attention with in-kernel relative-position bias, LSE-fused static/dynamic
// per-token sinks, and optional output gate.
// Uses 1 warp (32 threads) per block for barrier-free warp-shuffle reduction,
// pre-scaled Q, and 32-wide coalesced register layout.

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"
#include <seismic/slab.cuh>

namespace rel_bias_attn {

using element::u64;
using element::u8;
constexpr int MAX_REGS = (SEISMIC_CONST_REL_BIAS_MAX_WIDTH + 31) / 32;
static_assert(MAX_REGS * 32 >= SEISMIC_CONST_REL_BIAS_MAX_WIDTH,
    "MAX_REGS must cover SEISMIC_CONST_REL_BIAS_MAX_WIDTH");

__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

__device__ __forceinline__ float gated_column(float attended, float g, int gate_function) {
    if (gate_function != 0)
        return attended * (fmaxf(g, 0.0f) + log1pf(expf(-fabsf(g))));
    return attended / (1.0f + expf(-g));
}

__device__ __forceinline__ long long count_total_keys(
    const int *visible, const int *fresh, u64 row, u64 R,
    const seismic_words_t &seismic_words_value) {
    long long total = 0;
    for (u64 span = 0; span < R; ++span) {
        const long long lo = visible[row * SEISMIC_VISIBLE_STRIDE_0 + span * SEISMIC_VISIBLE_STRIDE_1];
        const long long hi = visible[row * SEISMIC_VISIBLE_STRIDE_0 + span * SEISMIC_VISIBLE_STRIDE_1
            + SEISMIC_VISIBLE_STRIDE_2];
        if (hi > lo && lo >= 0)
            total += hi - lo;
    }
    const long long flo = fresh[row * SEISMIC_FRESH_STRIDE_0];
    const long long fhi = fresh[row * SEISMIC_FRESH_STRIDE_0 + SEISMIC_FRESH_STRIDE_1];
    if (fhi > flo && flo >= 0)
        total += fhi - flo;
    return total;
}

__device__ __forceinline__ void append_kv(
    const u8 *key, const u8 *value, const int *destinations,
    const element::SlabOut<activation::Act> &history_key,
    const element::SlabOut<activation::Act> &history_value,
    u64 row, u64 kv_head, u64 W,
    const seismic_words_t &seismic_words_value) {
    const int dest = destinations[row * SEISMIC_DESTINATIONS_STRIDE_0];
    if (dest < 0)
        return;
    const u64 slab_rows = SEISMIC_PARAM_SLAB_ROWS;
    const auto hk = history_key.row(static_cast<u64>(dest), slab_rows, SEISMIC_HISTORY_KEY_STRIDE_0);
    const auto hv = history_value.row(static_cast<u64>(dest), slab_rows, SEISMIC_HISTORY_VALUE_STRIDE_0);
    const u64 k_base = row * SEISMIC_KEY_STRIDE_0 + kv_head * SEISMIC_KEY_STRIDE_1;
    const u64 v_base = row * SEISMIC_VALUE_STRIDE_0 + kv_head * SEISMIC_VALUE_STRIDE_1;
    for (u64 c = threadIdx.x; c < W; c += 32u) {
        hk.put(kv_head * SEISMIC_HISTORY_KEY_STRIDE_1 + c * SEISMIC_HISTORY_KEY_STRIDE_2,
            element::at<activation::Act>(key, k_base + c * SEISMIC_KEY_STRIDE_2));
        hv.put(kv_head * SEISMIC_HISTORY_VALUE_STRIDE_1 + c * SEISMIC_HISTORY_VALUE_STRIDE_2,
            element::at<activation::Act>(value, v_base + c * SEISMIC_VALUE_STRIDE_2));
    }
}

__device__ __forceinline__ void scan_range(
    const float (&q_reg)[MAX_REGS],
    float (&acc)[MAX_REGS],
    float &maximum,
    float &denominator,
    const u8 *key, const u8 *value, const float *rel_bias,
    const int *visible, const int *fresh,
    const element::SlabOut<activation::Act> &history_key,
    const element::SlabOut<activation::Act> &history_value,
    u64 row, u64 q_head, u64 kv_head, u64 W, u64 R, u64 EXT,
    long long total_keys, long long lo_k, long long hi_k,
    const seismic_words_t &seismic_words_value) {
    const u64 slab_rows = SEISMIC_PARAM_SLAB_ROWS;
    const u64 rb_base = row * SEISMIC_REL_BIAS_STRIDE_0 + q_head * SEISMIC_REL_BIAS_STRIDE_1;
    const int regs = static_cast<int>((W + 31u) / 32u);
    long long ordinal = 0;

    for (u64 span = 0; span < R; ++span) {
        const long long lo = visible[row * SEISMIC_VISIBLE_STRIDE_0 + span * SEISMIC_VISIBLE_STRIDE_1];
        const long long hi = visible[row * SEISMIC_VISIBLE_STRIDE_0 + span * SEISMIC_VISIBLE_STRIDE_1
            + SEISMIC_VISIBLE_STRIDE_2];
        if (hi <= lo || lo < 0)
            continue;
        const long long span_len = hi - lo;
        const long long start = max(0LL, lo_k - ordinal);
        const long long end = min(span_len, hi_k - ordinal);
        if (start < end) {
            const u64 hk_s0 = SEISMIC_HISTORY_KEY_STRIDE_0;
            const u64 hk_s1 = SEISMIC_HISTORY_KEY_STRIDE_1;
            const u64 hk_s2 = SEISMIC_HISTORY_KEY_STRIDE_2;
            const u64 hv_s0 = SEISMIC_HISTORY_VALUE_STRIDE_0;
            const u64 hv_s1 = SEISMIC_HISTORY_VALUE_STRIDE_1;
            const u64 hv_s2 = SEISMIC_HISTORY_VALUE_STRIDE_2;
            const bool fast128 = (W == 128u && hk_s2 == 1u && hv_s2 == 1u);
            const u64 tok0 = static_cast<u64>(lo + start);
            u64 slab_idx = tok0 / slab_rows;
            u64 in_slab = tok0 % slab_rows;
            auto hk_slab = history_key.view_region(slab_idx);
            auto hv_slab = history_value.view_region(slab_idx);
            const u64 c_base = static_cast<u64>(threadIdx.x) * 4u;

            for (long long idx = start; idx < end; ++idx) {
                if (in_slab == slab_rows) {
                    in_slab = 0u;
                    ++slab_idx;
                    hk_slab = history_key.view_region(slab_idx);
                    hv_slab = history_value.view_region(slab_idx);
                }
                const u64 hk_row = in_slab * hk_s0 + kv_head * hk_s1;
                const u64 hv_row = in_slab * hv_s0 + kv_head * hv_s1;
                float dot = 0.0f;
                if (fast128) {
                    const u64 ko = hk_row + c_base;
                    float kv[4];
                    hk_slab.template span<4>(ko, kv);
                    dot = seismic_fma_rn(q_reg[0], kv[0], dot);
                    dot = seismic_fma_rn(q_reg[1], kv[1], dot);
                    dot = seismic_fma_rn(q_reg[2], kv[2], dot);
                    dot = seismic_fma_rn(q_reg[3], kv[3], dot);
                } else {
                    for (int r_i = 0; r_i < regs; ++r_i) {
                        const u64 c = static_cast<u64>(threadIdx.x) + static_cast<u64>(r_i) * 32u;
                        if (c < W) {
                            const float kv = hk_slab.at(hk_row + c * hk_s2);
                            dot = seismic_fma_rn(q_reg[r_i], kv, dot);
                        }
                    }
                }
                float score = warp_sum(dot);
                const long long d = (total_keys - 1LL) - (ordinal + idx);
                if (EXT > 0 && d >= 0 && static_cast<u64>(d) < EXT)
                    score += rel_bias[rb_base + static_cast<u64>(d) * SEISMIC_REL_BIAS_STRIDE_2];
                const float next = fmaxf(maximum, score);
                const float p = expf(score - next);
                const float carry = expf(maximum - next);
                denominator = seismic_fma_rn(denominator, carry, p);
                if (fast128) {
                    const u64 vo = hv_row + c_base;
                    float vv[4];
                    hv_slab.template span<4>(vo, vv);
                    acc[0] = seismic_fma_rn(p, vv[0], acc[0] * carry);
                    acc[1] = seismic_fma_rn(p, vv[1], acc[1] * carry);
                    acc[2] = seismic_fma_rn(p, vv[2], acc[2] * carry);
                    acc[3] = seismic_fma_rn(p, vv[3], acc[3] * carry);
                } else {
                    for (int r_i = 0; r_i < regs; ++r_i) {
                        const u64 c = static_cast<u64>(threadIdx.x) + static_cast<u64>(r_i) * 32u;
                        if (c < W) {
                            const float vv = hv_slab.at(hv_row + c * hv_s2);
                            acc[r_i] = seismic_fma_rn(p, vv, acc[r_i] * carry);
                        }
                    }
                }
                maximum = next;
                ++in_slab;
            }
        }
        ordinal += span_len;
    }

    const long long flo = fresh[row * SEISMIC_FRESH_STRIDE_0];
    const long long fhi = fresh[row * SEISMIC_FRESH_STRIDE_0 + SEISMIC_FRESH_STRIDE_1];
    if (fhi > flo && flo >= 0) {
        const long long span_len = fhi - flo;
        const long long start = max(0LL, lo_k - ordinal);
        const long long end = min(span_len, hi_k - ordinal);
        const bool fast128 = (W == 128u && SEISMIC_HISTORY_KEY_STRIDE_2 == 1u && SEISMIC_HISTORY_VALUE_STRIDE_2 == 1u);
        const bool fast128_fresh_k = fast128 && (SEISMIC_KEY_STRIDE_2 == 1u);
        const bool fast128_fresh_v = fast128 && (SEISMIC_VALUE_STRIDE_2 == 1u);
        const u64 c_base = static_cast<u64>(threadIdx.x) * 4u;
        for (long long idx = start; idx < end; ++idx) {
            const u64 tok = static_cast<u64>(flo + idx);
            if (tok >= SEISMIC_DIM_M)
                continue;
            const u64 k_base = tok * SEISMIC_KEY_STRIDE_0 + kv_head * SEISMIC_KEY_STRIDE_1;
            const u64 v_base = tok * SEISMIC_VALUE_STRIDE_0 + kv_head * SEISMIC_VALUE_STRIDE_1;
            float dot = 0.0f;
            if (fast128_fresh_k) {
                float kv[4];
                element::span<activation::Act, 4, false>(key, k_base + c_base, kv);
                dot = seismic_fma_rn(q_reg[0], kv[0], dot);
                dot = seismic_fma_rn(q_reg[1], kv[1], dot);
                dot = seismic_fma_rn(q_reg[2], kv[2], dot);
                dot = seismic_fma_rn(q_reg[3], kv[3], dot);
            } else if (fast128) {
                for (int r_i = 0; r_i < 4; ++r_i) {
                    const u64 c = c_base + static_cast<u64>(r_i);
                    const float kv = element::at<activation::Act>(key, k_base + c * SEISMIC_KEY_STRIDE_2);
                    dot = seismic_fma_rn(q_reg[r_i], kv, dot);
                }
            } else {
                for (int r_i = 0; r_i < regs; ++r_i) {
                    const u64 c = static_cast<u64>(threadIdx.x) + static_cast<u64>(r_i) * 32u;
                    if (c < W) {
                        const float kv = element::at<activation::Act>(key, k_base + c * SEISMIC_KEY_STRIDE_2);
                        dot = seismic_fma_rn(q_reg[r_i], kv, dot);
                    }
                }
            }
            float score = warp_sum(dot);
            const long long d = (total_keys - 1LL) - (ordinal + idx);
            if (EXT > 0 && d >= 0 && static_cast<u64>(d) < EXT)
                score += rel_bias[rb_base + static_cast<u64>(d) * SEISMIC_REL_BIAS_STRIDE_2];
            const float next = fmaxf(maximum, score);
            const float p = expf(score - next);
            const float carry = expf(maximum - next);
            denominator = seismic_fma_rn(denominator, carry, p);
            if (fast128_fresh_v) {
                float vv[4];
                element::span<activation::Act, 4, false>(value, v_base + c_base, vv);
                acc[0] = seismic_fma_rn(p, vv[0], acc[0] * carry);
                acc[1] = seismic_fma_rn(p, vv[1], acc[1] * carry);
                acc[2] = seismic_fma_rn(p, vv[2], acc[2] * carry);
                acc[3] = seismic_fma_rn(p, vv[3], acc[3] * carry);
            } else if (fast128) {
                for (int r_i = 0; r_i < 4; ++r_i) {
                    const u64 c = c_base + static_cast<u64>(r_i);
                    const float vv = element::at<activation::Act>(value, v_base + c * SEISMIC_VALUE_STRIDE_2);
                    acc[r_i] = seismic_fma_rn(p, vv, acc[r_i] * carry);
                }
            } else {
                for (int r_i = 0; r_i < regs; ++r_i) {
                    const u64 c = static_cast<u64>(threadIdx.x) + static_cast<u64>(r_i) * 32u;
                    if (c < W) {
                        const float vv = element::at<activation::Act>(value, v_base + c * SEISMIC_VALUE_STRIDE_2);
                        acc[r_i] = seismic_fma_rn(p, vv, acc[r_i] * carry);
                    }
                }
            }
            maximum = next;
        }
    }
}

} // namespace rel_bias_attn

extern "C" __global__ void attention_rel_bias_partial(SEISMIC_KERNEL_PARAMS) {
    using rel_bias_attn::u64;
    using rel_bias_attn::u8;
    const u64 q_head = blockIdx.x;
    const u64 part = blockIdx.y;
    const u64 row = blockIdx.z;
    const u64 G = SEISMIC_DIM_G;
    const u64 kv_head = q_head / G;
    const u64 W = SEISMIC_DIM_W;
    const u64 R = SEISMIC_DIM_R;
    const u64 EXT = SEISMIC_DIM_EXT;
    const u64 parts = SEISMIC_CONST_REL_BIAS_PARTS;
    const float scale = __uint_as_float((unsigned)SEISMIC_PARAM_SCALE);

    const u8 *query = SEISMIC_PTR(SEISMIC_BUFFER_QUERY);
    const u8 *key = SEISMIC_PTR(SEISMIC_BUFFER_KEY);
    const u8 *value = SEISMIC_PTR(SEISMIC_BUFFER_VALUE);
    const float *rel_bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_REL_BIAS));
    const int *visible = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_VISIBLE));
    const int *fresh = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_FRESH));
    const int *destinations = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_DESTINATIONS));
    const auto history_key = SEISMIC_OUT_HISTORY_KEY;
    const auto history_value = SEISMIC_OUT_HISTORY_VALUE;

    if (part == 0 && (q_head % G) == 0) {
        rel_bias_attn::append_kv(key, value, destinations, history_key, history_value,
            row, kv_head, W, seismic_words_value);
    }

    const long long total_keys = rel_bias_attn::count_total_keys(visible, fresh, row, R, seismic_words_value);
    const long long chunk = max(32LL, (total_keys + static_cast<long long>(parts) - 1LL) / static_cast<long long>(parts));
    const long long lo_k = static_cast<long long>(part) * chunk;
    const long long hi_k = min(lo_k + chunk, total_keys);

    float q_reg[rel_bias_attn::MAX_REGS];
    float acc[rel_bias_attn::MAX_REGS];
    const u64 q_base = row * SEISMIC_QUERY_STRIDE_0 + q_head * SEISMIC_QUERY_STRIDE_1;
    const int regs = static_cast<int>((W + 31u) / 32u);
    const bool fast128 = (W == 128u && SEISMIC_HISTORY_KEY_STRIDE_2 == 1u && SEISMIC_HISTORY_VALUE_STRIDE_2 == 1u);
    const u64 c_base = static_cast<u64>(threadIdx.x) * 4u;
    if (fast128) {
        for (int r_i = 0; r_i < 4; ++r_i) {
            const u64 c = c_base + static_cast<u64>(r_i);
            q_reg[r_i] = element::at<activation::Act>(query, q_base + c * SEISMIC_QUERY_STRIDE_2) * scale;
            acc[r_i] = 0.0f;
        }
    } else {
        for (int r_i = 0; r_i < regs; ++r_i) {
            const u64 c = static_cast<u64>(threadIdx.x) + static_cast<u64>(r_i) * 32u;
            q_reg[r_i] = c < W ? element::at<activation::Act>(query, q_base + c * SEISMIC_QUERY_STRIDE_2) * scale : 0.0f;
            acc[r_i] = 0.0f;
        }
    }
    float maximum = -__int_as_float(0x7f800000);
    float denominator = 0.0f;

    if (lo_k < hi_k) {
        rel_bias_attn::scan_range(q_reg, acc, maximum, denominator,
            key, value, rel_bias, visible, fresh, history_key, history_value,
            row, q_head, kv_head, W, R, EXT, total_keys, lo_k, hi_k, seismic_words_value);
    }

    float *partials_out = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    float *stats_out = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS));
    const u64 slot = (row * (SEISMIC_DIM_KV * G) + q_head) * parts + part;
    if (threadIdx.x == 0) {
        stats_out[slot * 2] = maximum;
        stats_out[slot * 2 + 1] = denominator;
    }
    if (fast128) {
        for (int r_i = 0; r_i < 4; ++r_i) {
            partials_out[slot * W + c_base + static_cast<u64>(r_i)] = acc[r_i];
        }
    } else {
        for (int r_i = 0; r_i < regs; ++r_i) {
            const u64 c = static_cast<u64>(threadIdx.x) + static_cast<u64>(r_i) * 32u;
            if (c < W)
                partials_out[slot * W + c] = acc[r_i];
        }
    }
}

extern "C" __global__ void attention_rel_bias_merge(SEISMIC_KERNEL_PARAMS) {
    using rel_bias_attn::u64;
    using rel_bias_attn::u8;
    const u64 q_head = blockIdx.x;
    const u64 row = blockIdx.y;
    const u64 W = SEISMIC_DIM_W;
    const u64 U = SEISMIC_DIM_U;
    const u64 SK = SEISMIC_DIM_SK;
    const u64 parts = SEISMIC_CONST_REL_BIAS_PARTS;
    const int gate_function = static_cast<int>(SEISMIC_PARAM_GATE_FUNCTION);
    const u64 base_slot = (row * (SEISMIC_DIM_KV * SEISMIC_DIM_G) + q_head) * parts;
    const int regs = static_cast<int>((W + 31u) / 32u);

    const u8 *gate = SEISMIC_PTR(SEISMIC_BUFFER_GATE);
    const float *sinks = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SINKS));
    const float *partials_in = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    const float *stats_in = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS));

    const float neg_inf = -__int_as_float(0x7f800000);
    float overall_m = neg_inf;
    for (u64 p = 0; p < parts; ++p)
        overall_m = fmaxf(overall_m, stats_in[(base_slot + p) * 2]);

    float total = 0.0f;
    float acc[rel_bias_attn::MAX_REGS];
    for (int r_i = 0; r_i < regs; ++r_i)
        acc[r_i] = 0.0f;

    for (u64 p = 0; p < parts; ++p) {
        const float pm = stats_in[(base_slot + p) * 2];
        const float pl = stats_in[(base_slot + p) * 2 + 1];
        if (pl > 0.0f) {
            const float w = expf(pm - overall_m);
            total = seismic_fma_rn(pl, w, total);
            const u64 p_off = (base_slot + p) * W;
            for (int r_i = 0; r_i < regs; ++r_i) {
                const u64 c = static_cast<u64>(threadIdx.x) + static_cast<u64>(r_i) * 32u;
                if (c < W)
                    acc[r_i] = seismic_fma_rn(partials_in[p_off + c], w, acc[r_i]);
            }
        }
    }

    if (SK > 0) {
        const float sink = sinks[(row % SK) * SEISMIC_SINKS_STRIDE_0 + q_head * SEISMIC_SINKS_STRIDE_1];
        if (sink > overall_m) {
            const float rescale = overall_m > neg_inf ? expf(overall_m - sink) : 0.0f;
            total = total * rescale + 1.0f;
            for (int r_i = 0; r_i < regs; ++r_i)
                acc[r_i] *= rescale;
        } else if (overall_m > neg_inf) {
            total += expf(sink - overall_m);
        }
    }
    const float denom = fmaxf(total, 1e-30f);
    const auto result = SEISMIC_RESULT_0_OUT.offset(
        row * SEISMIC_RESULT_0_STRIDE_0 + q_head * SEISMIC_RESULT_0_STRIDE_1);
    const u64 g_base = row * SEISMIC_GATE_STRIDE_0 + q_head * SEISMIC_GATE_STRIDE_1;
    for (int r_i = 0; r_i < regs; ++r_i) {
        const u64 c = static_cast<u64>(threadIdx.x) + static_cast<u64>(r_i) * 32u;
        if (c < W) {
            float val = acc[r_i] / denom;
            if (U > 0) {
                const float gv = element::at<activation::Act>(gate, g_base + (c % U) * SEISMIC_GATE_STRIDE_2);
                val = rel_bias_attn::gated_column(val, gv, gate_function);
            }
            result.put(c * SEISMIC_RESULT_0_STRIDE_2, val);
        }
    }
}

extern "C" __global__ void attention_rel_bias_prefill(SEISMIC_KERNEL_PARAMS) {
    using rel_bias_attn::u64;
    using rel_bias_attn::u8;
    const u64 q_head = blockIdx.x;
    const u64 G = SEISMIC_DIM_G;
    const u64 kv_head = q_head / G;
    const u64 W = SEISMIC_DIM_W;
    const u64 R = SEISMIC_DIM_R;
    const u64 EXT = SEISMIC_DIM_EXT;
    const u64 U = SEISMIC_DIM_U;
    const u64 SK = SEISMIC_DIM_SK;
    const float scale = __uint_as_float((unsigned)SEISMIC_PARAM_SCALE);
    const int gate_function = static_cast<int>(SEISMIC_PARAM_GATE_FUNCTION);
    const int regs = static_cast<int>((W + 31u) / 32u);

    const u8 *query = SEISMIC_PTR(SEISMIC_BUFFER_QUERY);
    const u8 *gate = SEISMIC_PTR(SEISMIC_BUFFER_GATE);
    const u8 *key = SEISMIC_PTR(SEISMIC_BUFFER_KEY);
    const u8 *value = SEISMIC_PTR(SEISMIC_BUFFER_VALUE);
    const float *rel_bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_REL_BIAS));
    const float *sinks = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SINKS));
    const int *visible = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_VISIBLE));
    const int *fresh = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_FRESH));
    const int *destinations = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_DESTINATIONS));
    const auto history_key = SEISMIC_OUT_HISTORY_KEY;
    const auto history_value = SEISMIC_OUT_HISTORY_VALUE;

    for (u64 row = blockIdx.y; row < SEISMIC_DIM_M; row += gridDim.y) {
        if ((q_head % G) == 0) {
            rel_bias_attn::append_kv(key, value, destinations, history_key, history_value,
                row, kv_head, W, seismic_words_value);
        }

        const long long total_keys = rel_bias_attn::count_total_keys(visible, fresh, row, R, seismic_words_value);
        float q_reg[rel_bias_attn::MAX_REGS];
        float acc[rel_bias_attn::MAX_REGS];
        const u64 q_base = row * SEISMIC_QUERY_STRIDE_0 + q_head * SEISMIC_QUERY_STRIDE_1;
        const bool fast128 = (W == 128u && SEISMIC_HISTORY_KEY_STRIDE_2 == 1u && SEISMIC_HISTORY_VALUE_STRIDE_2 == 1u);
        const u64 c_base = static_cast<u64>(threadIdx.x) * 4u;
        if (fast128) {
            for (int r_i = 0; r_i < 4; ++r_i) {
                const u64 c = c_base + static_cast<u64>(r_i);
                q_reg[r_i] = element::at<activation::Act>(query, q_base + c * SEISMIC_QUERY_STRIDE_2) * scale;
                acc[r_i] = 0.0f;
            }
        } else {
            for (int r_i = 0; r_i < regs; ++r_i) {
                const u64 c = static_cast<u64>(threadIdx.x) + static_cast<u64>(r_i) * 32u;
                q_reg[r_i] = c < W ? element::at<activation::Act>(query, q_base + c * SEISMIC_QUERY_STRIDE_2) * scale : 0.0f;
                acc[r_i] = 0.0f;
            }
        }
        float maximum = -__int_as_float(0x7f800000);
        float denominator = 0.0f;

        if (total_keys > 0) {
            rel_bias_attn::scan_range(q_reg, acc, maximum, denominator,
                key, value, rel_bias, visible, fresh, history_key, history_value,
                row, q_head, kv_head, W, R, EXT, total_keys, 0LL, total_keys, seismic_words_value);
        }

        float total = denominator;
        if (SK > 0) {
            const float sink = sinks[(row % SK) * SEISMIC_SINKS_STRIDE_0 + q_head * SEISMIC_SINKS_STRIDE_1];
            if (sink > maximum) {
                const float rescale = maximum > -__int_as_float(0x7f800000) ? expf(maximum - sink) : 0.0f;
                total = total * rescale + 1.0f;
                for (int r_i = 0; r_i < regs; ++r_i)
                    acc[r_i] *= rescale;
            } else if (maximum > -__int_as_float(0x7f800000)) {
                total += expf(sink - maximum);
            }
        }
        const float denom = fmaxf(total, 1e-30f);
        const auto result = SEISMIC_RESULT_0_OUT.offset(
            row * SEISMIC_RESULT_0_STRIDE_0 + q_head * SEISMIC_RESULT_0_STRIDE_1);
        const u64 g_base = row * SEISMIC_GATE_STRIDE_0 + q_head * SEISMIC_GATE_STRIDE_1;
        if (fast128) {
            for (int r_i = 0; r_i < 4; ++r_i) {
                const u64 c = c_base + static_cast<u64>(r_i);
                float val = acc[r_i] / denom;
                if (U > 0) {
                    const float gv = element::at<activation::Act>(gate, g_base + (c % U) * SEISMIC_GATE_STRIDE_2);
                    val = rel_bias_attn::gated_column(val, gv, gate_function);
                }
                result.put(c * SEISMIC_RESULT_0_STRIDE_2, val);
            }
        } else {
            for (int r_i = 0; r_i < regs; ++r_i) {
                const u64 c = static_cast<u64>(threadIdx.x) + static_cast<u64>(r_i) * 32u;
                if (c < W) {
                    float val = acc[r_i] / denom;
                    if (U > 0) {
                        const float gv = element::at<activation::Act>(gate, g_base + (c % U) * SEISMIC_GATE_STRIDE_2);
                        val = rel_bias_attn::gated_column(val, gv, gate_function);
                    }
                    result.put(c * SEISMIC_RESULT_0_STRIDE_2, val);
                }
            }
        }
    }
}
