// attention_rel_bias (contract and portable body in attention.seismic):
// attention with in-kernel relative-position bias, LSE-fused static/dynamic
// per-token sinks, and optional output gate.
// Decode/verify (M <= VERIFY_ROWS): split-KV over REL_BIAS_PARTS partitions + merge.
// Prefill (M > VERIFY_ROWS): single-pass per (q_head, row), zero partial scratch.
// Uses 1 simdgroup (32 threads) per threadgroup for barrier-free `simd_sum`,
// hoisted slab pointer stepping (zero 64-bit divisions per token), pre-scaled Q,
// and vectorized 4-element loads for contiguous W=128 KV heads.

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"
#include <seismic/slab.h>

namespace rel_bias_attn {

constant constexpr uint MAX_REGS = (uint(SEISMIC_CONST_REL_BIAS_MAX_WIDTH) + 31u) / 32u;
static_assert(MAX_REGS * 32u >= uint(SEISMIC_CONST_REL_BIAS_MAX_WIDTH),
    "MAX_REGS must cover SEISMIC_CONST_REL_BIAS_MAX_WIDTH");

inline float gated_column(float attended, float g, int gate_function) {
    if (gate_function != 0)
        return attended * (metal::max(g, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(g))));
    return attended / (1.0f + metal::exp(-g));
}

inline long count_total_keys(
    device const int *visible, device const int *fresh,
    ulong row, ulong R, constant ulong *seismic_words) {
    long total = 0;
    for (ulong span = 0; span < R; ++span) {
        const long lo = visible[row * SEISMIC_VISIBLE_STRIDE_0 + span * SEISMIC_VISIBLE_STRIDE_1];
        const long hi = visible[row * SEISMIC_VISIBLE_STRIDE_0 + span * SEISMIC_VISIBLE_STRIDE_1
            + SEISMIC_VISIBLE_STRIDE_2];
        if (hi > lo && lo >= 0)
            total += hi - lo;
    }
    const long flo = fresh[row * SEISMIC_FRESH_STRIDE_0];
    const long fhi = fresh[row * SEISMIC_FRESH_STRIDE_0 + SEISMIC_FRESH_STRIDE_1];
    if (fhi > flo && flo >= 0)
        total += fhi - flo;
    return total;
}

inline void append_kv(
    device const uchar *key, device const uchar *value,
    device const int *destinations,
    SEISMIC_OUT_HISTORY_KEY_PARAM(history_key),
    SEISMIC_OUT_HISTORY_VALUE_PARAM(history_value),
    ulong row, ulong kv_head, ulong W, uint tid,
    constant ulong *seismic_words) {
    const int dest = destinations[row * SEISMIC_DESTINATIONS_STRIDE_0];
    if (dest < 0)
        return;
    const ulong slab_rows = ulong(SEISMIC_PARAM_SLAB_ROWS);
    const element::Out<activation::Act> hk = slab::out(
        slab::row_entry(history_key, ulong(dest), slab_rows, SEISMIC_HISTORY_KEY_STRIDE_0));
    const element::Out<activation::Act> hv = slab::out(
        slab::row_entry(history_value, ulong(dest), slab_rows, SEISMIC_HISTORY_VALUE_STRIDE_0));
    const ulong k_base = row * SEISMIC_KEY_STRIDE_0 + kv_head * SEISMIC_KEY_STRIDE_1;
    const ulong v_base = row * SEISMIC_VALUE_STRIDE_0 + kv_head * SEISMIC_VALUE_STRIDE_1;
    for (ulong c = tid; c < W; c += 32u) {
        hk.put(kv_head * SEISMIC_HISTORY_KEY_STRIDE_1 + c * SEISMIC_HISTORY_KEY_STRIDE_2,
            element::at<activation::Act>(key, k_base + c * SEISMIC_KEY_STRIDE_2));
        hv.put(kv_head * SEISMIC_HISTORY_VALUE_STRIDE_1 + c * SEISMIC_HISTORY_VALUE_STRIDE_2,
            element::at<activation::Act>(value, v_base + c * SEISMIC_VALUE_STRIDE_2));
    }
}

inline void scan_range(
    const thread float (&q_reg)[MAX_REGS],
    thread float (&acc)[MAX_REGS],
    thread float &maximum,
    thread float &denominator,
    device const uchar *key, device const uchar *value,
    device const float *rel_bias,
    device const int *visible, device const int *fresh,
    SEISMIC_OUT_HISTORY_KEY_PARAM(history_key),
    SEISMIC_OUT_HISTORY_VALUE_PARAM(history_value),
    ulong row, ulong q_head, ulong kv_head, ulong W, ulong R, ulong EXT,
    long total_keys, long lo_k, long hi_k,
    uint tid,
    constant ulong *seismic_words) {
    using Storage = typename activation::Act::storage;
    const ulong slab_rows = ulong(SEISMIC_PARAM_SLAB_ROWS);
    const ulong rb_base = row * SEISMIC_REL_BIAS_STRIDE_0 + q_head * SEISMIC_REL_BIAS_STRIDE_1;
    const ulong hk_s0 = SEISMIC_HISTORY_KEY_STRIDE_0;
    const ulong hv_s0 = SEISMIC_HISTORY_VALUE_STRIDE_0;
    const ulong hk_h_off = kv_head * SEISMIC_HISTORY_KEY_STRIDE_1;
    const ulong hv_h_off = kv_head * SEISMIC_HISTORY_VALUE_STRIDE_1;
    const bool fast_w128 = (W == 128u && SEISMIC_HISTORY_KEY_STRIDE_2 == 1u && SEISMIC_HISTORY_VALUE_STRIDE_2 == 1u);
    long ordinal = 0;

    for (ulong span = 0; span < R; ++span) {
        const long lo = visible[row * SEISMIC_VISIBLE_STRIDE_0 + span * SEISMIC_VISIBLE_STRIDE_1];
        const long hi = visible[row * SEISMIC_VISIBLE_STRIDE_0 + span * SEISMIC_VISIBLE_STRIDE_1
            + SEISMIC_VISIBLE_STRIDE_2];
        if (hi <= lo || lo < 0)
            continue;
        const long span_len = hi - lo;
        const long start = metal::max(0L, lo_k - ordinal);
        const long end = metal::min(span_len, hi_k - ordinal);
        long idx = start;
        while (idx < end) {
            const ulong tok = ulong(lo + idx);
            const ulong in_slab = tok % slab_rows;
            const long chunk_end = metal::min(end, idx + long(slab_rows - in_slab));
            device const Storage *hk = slab::view(
                slab::row_entry(history_key, tok, slab_rows, hk_s0)) + hk_h_off;
            device const Storage *hv = slab::view(
                slab::row_entry(history_value, tok, slab_rows, hv_s0)) + hv_h_off;

            if (fast_w128) {
                const float q0 = q_reg[0], q1 = q_reg[1], q2 = q_reg[2], q3 = q_reg[3];
                float a0 = acc[0], a1 = acc[1], a2 = acc[2], a3 = acc[3];
                const ulong c0 = ulong(tid) * 4u;
                for (; idx < chunk_end; ++idx, hk += hk_s0, hv += hv_s0) {
                    const float k0 = activation::Act::load(hk[c0]);
                    const float k1 = activation::Act::load(hk[c0 + 1u]);
                    const float k2 = activation::Act::load(hk[c0 + 2u]);
                    const float k3 = activation::Act::load(hk[c0 + 3u]);
                    float dot = metal::fma(q0, k0, metal::fma(q1, k1, metal::fma(q2, k2, q3 * k3)));
                    float score = metal::simd_sum(dot);
                    const long d = (total_keys - 1L) - (ordinal + idx);
                    if (EXT > 0 && d >= 0 && ulong(d) < EXT)
                        score += rel_bias[rb_base + ulong(d) * SEISMIC_REL_BIAS_STRIDE_2];
                    const float next = metal::max(maximum, score);
                    const float p = metal::exp(score - next);
                    const float carry = metal::exp(maximum - next);
                    denominator = metal::fma(denominator, carry, p);
                    const float v0 = activation::Act::load(hv[c0]);
                    const float v1 = activation::Act::load(hv[c0 + 1u]);
                    const float v2 = activation::Act::load(hv[c0 + 2u]);
                    const float v3 = activation::Act::load(hv[c0 + 3u]);
                    a0 = metal::fma(p, v0, a0 * carry);
                    a1 = metal::fma(p, v1, a1 * carry);
                    a2 = metal::fma(p, v2, a2 * carry);
                    a3 = metal::fma(p, v3, a3 * carry);
                    maximum = next;
                }
                acc[0] = a0; acc[1] = a1; acc[2] = a2; acc[3] = a3;
            } else {
                const uint regs = uint((W + 31u) / 32u);
                for (; idx < chunk_end; ++idx, hk += hk_s0, hv += hv_s0) {
                    float dot = 0.0f;
                    for (uint r_i = 0; r_i < regs; ++r_i) {
                        const ulong c = ulong(tid) + ulong(r_i) * 32u;
                        if (c < W) {
                            const float kv = activation::Act::load(hk[c * SEISMIC_HISTORY_KEY_STRIDE_2]);
                            dot = metal::fma(q_reg[r_i], kv, dot);
                        }
                    }
                    float score = metal::simd_sum(dot);
                    const long d = (total_keys - 1L) - (ordinal + idx);
                    if (EXT > 0 && d >= 0 && ulong(d) < EXT)
                        score += rel_bias[rb_base + ulong(d) * SEISMIC_REL_BIAS_STRIDE_2];
                    const float next = metal::max(maximum, score);
                    const float p = metal::exp(score - next);
                    const float carry = metal::exp(maximum - next);
                    denominator = metal::fma(denominator, carry, p);
                    for (uint r_i = 0; r_i < regs; ++r_i) {
                        const ulong c = ulong(tid) + ulong(r_i) * 32u;
                        if (c < W) {
                            const float vv = activation::Act::load(hv[c * SEISMIC_HISTORY_VALUE_STRIDE_2]);
                            acc[r_i] = metal::fma(p, vv, acc[r_i] * carry);
                        }
                    }
                    maximum = next;
                }
            }
        }
        ordinal += span_len;
    }

    const long flo = fresh[row * SEISMIC_FRESH_STRIDE_0];
    const long fhi = fresh[row * SEISMIC_FRESH_STRIDE_0 + SEISMIC_FRESH_STRIDE_1];
    if (fhi > flo && flo >= 0) {
        const long span_len = fhi - flo;
        const long start = metal::max(0L, lo_k - ordinal);
        const long end = metal::min(span_len, hi_k - ordinal);
        const uint regs = uint((W + 31u) / 32u);
        for (long idx = start; idx < end; ++idx) {
            const ulong tok = ulong(flo + idx);
            if (tok >= SEISMIC_DIM_M)
                continue;
            const ulong k_base = tok * SEISMIC_KEY_STRIDE_0 + kv_head * SEISMIC_KEY_STRIDE_1;
            const ulong v_base = tok * SEISMIC_VALUE_STRIDE_0 + kv_head * SEISMIC_VALUE_STRIDE_1;
            float dot = 0.0f;
            if (fast_w128) {
                const ulong c0 = ulong(tid) * 4u;
                for (uint j = 0; j < 4u; ++j) {
                    const float kv = element::at<activation::Act>(key, k_base + (c0 + j) * SEISMIC_KEY_STRIDE_2);
                    dot = metal::fma(q_reg[j], kv, dot);
                }
            } else {
                for (uint r_i = 0; r_i < regs; ++r_i) {
                    const ulong c = ulong(tid) + ulong(r_i) * 32u;
                    if (c < W) {
                        const float kv = element::at<activation::Act>(key, k_base + c * SEISMIC_KEY_STRIDE_2);
                        dot = metal::fma(q_reg[r_i], kv, dot);
                    }
                }
            }
            float score = metal::simd_sum(dot);
            const long d = (total_keys - 1L) - (ordinal + idx);
            if (EXT > 0 && d >= 0 && ulong(d) < EXT)
                score += rel_bias[rb_base + ulong(d) * SEISMIC_REL_BIAS_STRIDE_2];
            const float next = metal::max(maximum, score);
            const float p = metal::exp(score - next);
            const float carry = metal::exp(maximum - next);
            denominator = metal::fma(denominator, carry, p);
            if (fast_w128) {
                const ulong c0 = ulong(tid) * 4u;
                for (uint j = 0; j < 4u; ++j) {
                    const float vv = element::at<activation::Act>(value, v_base + (c0 + j) * SEISMIC_VALUE_STRIDE_2);
                    acc[j] = metal::fma(p, vv, acc[j] * carry);
                }
            } else {
                for (uint r_i = 0; r_i < regs; ++r_i) {
                    const ulong c = ulong(tid) + ulong(r_i) * 32u;
                    if (c < W) {
                        const float vv = element::at<activation::Act>(value, v_base + c * SEISMIC_VALUE_STRIDE_2);
                        acc[r_i] = metal::fma(p, vv, acc[r_i] * carry);
                    }
                }
            }
            maximum = next;
        }
    }
}

} // namespace rel_bias_attn

kernel void attention_rel_bias_partial(
    device const uchar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const uchar *key [[buffer(SEISMIC_BUFFER_KEY)]],
    device const uchar *value [[buffer(SEISMIC_BUFFER_VALUE)]],
    device const float *rel_bias [[buffer(SEISMIC_BUFFER_REL_BIAS)]],
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device const int *fresh [[buffer(SEISMIC_BUFFER_FRESH)]],
    device const int *destinations [[buffer(SEISMIC_BUFFER_DESTINATIONS)]],
    SEISMIC_OUT_HISTORY_KEY_PARAM(history_key),
    SEISMIC_OUT_HISTORY_VALUE_PARAM(history_value),
    device float *partials_out [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device float *stats_out [[buffer(SEISMIC_BUFFER_SCRATCH_STATISTICS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const ulong q_head = group.x;
    const ulong part = group.y;
    const ulong row = group.z;
    const ulong G = SEISMIC_DIM_G;
    const ulong kv_head = q_head / G;
    const ulong W = SEISMIC_DIM_W;
    const ulong R = SEISMIC_DIM_R;
    const ulong EXT = SEISMIC_DIM_EXT;
    const ulong parts = SEISMIC_CONST_REL_BIAS_PARTS;
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    const bool fast_w128 = (W == 128u && SEISMIC_HISTORY_KEY_STRIDE_2 == 1u && SEISMIC_HISTORY_VALUE_STRIDE_2 == 1u);

    if (part == 0 && (q_head % G) == 0) {
        rel_bias_attn::append_kv(key, value, destinations, history_key, history_value,
            row, kv_head, W, tid, seismic_words);
    }

    const long total_keys = rel_bias_attn::count_total_keys(visible, fresh, row, R, seismic_words);
    const long chunk = metal::max(32L, (total_keys + long(parts) - 1L) / long(parts));
    const long lo_k = long(part) * chunk;
    const long hi_k = metal::min(lo_k + chunk, total_keys);

    float q_reg[rel_bias_attn::MAX_REGS];
    float acc[rel_bias_attn::MAX_REGS];
    const ulong q_base = row * SEISMIC_QUERY_STRIDE_0 + q_head * SEISMIC_QUERY_STRIDE_1;
    if (fast_w128) {
        const ulong c0 = ulong(tid) * 4u;
        for (uint j = 0; j < 4u; ++j) {
            q_reg[j] = element::at<activation::Act>(query, q_base + (c0 + j) * SEISMIC_QUERY_STRIDE_2) * scale;
            acc[j] = 0.0f;
        }
    } else {
        for (uint r_i = 0; r_i < rel_bias_attn::MAX_REGS; ++r_i) {
            const ulong c = ulong(tid) + ulong(r_i) * 32u;
            q_reg[r_i] = c < W ? element::at<activation::Act>(query, q_base + c * SEISMIC_QUERY_STRIDE_2) * scale : 0.0f;
            acc[r_i] = 0.0f;
        }
    }
    float maximum = -INFINITY;
    float denominator = 0.0f;

    if (lo_k < hi_k) {
        rel_bias_attn::scan_range(q_reg, acc, maximum, denominator,
            key, value, rel_bias, visible, fresh, history_key, history_value,
            row, q_head, kv_head, W, R, EXT, total_keys, lo_k, hi_k,
            tid, seismic_words);
    }

    const ulong slot = (row * (SEISMIC_DIM_KV * G) + q_head) * parts + part;
    if (tid == 0) {
        stats_out[slot * 2] = maximum;
        stats_out[slot * 2 + 1] = denominator;
    }
    if (fast_w128) {
        const ulong c0 = ulong(tid) * 4u;
        for (uint j = 0; j < 4u; ++j)
            partials_out[slot * W + c0 + j] = acc[j];
    } else {
        for (uint r_i = 0; r_i < rel_bias_attn::MAX_REGS; ++r_i) {
            const ulong c = ulong(tid) + ulong(r_i) * 32u;
            if (c < W)
                partials_out[slot * W + c] = acc[r_i];
        }
    }
}

kernel void attention_rel_bias_merge(
    device const uchar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const float *sinks [[buffer(SEISMIC_BUFFER_SINKS)]],
    device const float *partials_in [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device const float *stats_in [[buffer(SEISMIC_BUFFER_SCRATCH_STATISTICS)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const ulong q_head = group.x;
    const ulong row = group.y;
    const ulong W = SEISMIC_DIM_W;
    const ulong U = SEISMIC_DIM_U;
    const ulong SK = SEISMIC_DIM_SK;
    const ulong parts = SEISMIC_CONST_REL_BIAS_PARTS;
    const int gate_function = int(SEISMIC_PARAM_GATE_FUNCTION);
    const ulong base_slot = (row * (SEISMIC_DIM_KV * SEISMIC_DIM_G) + q_head) * parts;

    float overall_m = -INFINITY;
    for (ulong p = 0; p < parts; ++p)
        overall_m = metal::max(overall_m, stats_in[(base_slot + p) * 2]);

    float total = 0.0f;
    float acc[rel_bias_attn::MAX_REGS];
    for (uint r_i = 0; r_i < rel_bias_attn::MAX_REGS; ++r_i)
        acc[r_i] = 0.0f;

    const uint regs = uint((W + 31u) / 32u);
    for (ulong p = 0; p < parts; ++p) {
        const float pm = stats_in[(base_slot + p) * 2];
        const float pl = stats_in[(base_slot + p) * 2 + 1];
        if (pl > 0.0f) {
            const float w = metal::exp(pm - overall_m);
            total = metal::fma(pl, w, total);
            const ulong p_off = (base_slot + p) * W;
            for (uint r_i = 0; r_i < regs; ++r_i) {
                const ulong c = ulong(tid) + ulong(r_i) * 32u;
                if (c < W)
                    acc[r_i] = metal::fma(partials_in[p_off + c], w, acc[r_i]);
            }
        }
    }

    if (SK > 0) {
        const float sink = sinks[(row % SK) * SEISMIC_SINKS_STRIDE_0 + q_head * SEISMIC_SINKS_STRIDE_1];
        if (sink > overall_m) {
            const float rescale = overall_m > -INFINITY ? metal::exp(overall_m - sink) : 0.0f;
            total = total * rescale + 1.0f;
            for (uint r_i = 0; r_i < regs; ++r_i)
                acc[r_i] *= rescale;
        } else if (overall_m > -INFINITY) {
            total += metal::exp(sink - overall_m);
        }
    }
    const float denom = metal::max(total, 1e-30f);
    const element::Out<activation::Act> result(result_cells);
    const ulong out_base = row * SEISMIC_RESULT_0_STRIDE_0 + q_head * SEISMIC_RESULT_0_STRIDE_1;
    const ulong g_base = row * SEISMIC_GATE_STRIDE_0 + q_head * SEISMIC_GATE_STRIDE_1;
    for (uint r_i = 0; r_i < regs; ++r_i) {
        const ulong c = ulong(tid) + ulong(r_i) * 32u;
        if (c < W) {
            float val = acc[r_i] / denom;
            if (U > 0) {
                const float gv = element::at<activation::Act>(gate, g_base + (c % U) * SEISMIC_GATE_STRIDE_2);
                val = rel_bias_attn::gated_column(val, gv, gate_function);
            }
            result.put(out_base + c * SEISMIC_RESULT_0_STRIDE_2, val);
        }
    }
}

kernel void attention_rel_bias_prefill(
    device const uchar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const uchar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const uchar *key [[buffer(SEISMIC_BUFFER_KEY)]],
    device const uchar *value [[buffer(SEISMIC_BUFFER_VALUE)]],
    device const float *rel_bias [[buffer(SEISMIC_BUFFER_REL_BIAS)]],
    device const float *sinks [[buffer(SEISMIC_BUFFER_SINKS)]],
    device const int *visible [[buffer(SEISMIC_BUFFER_VISIBLE)]],
    device const int *fresh [[buffer(SEISMIC_BUFFER_FRESH)]],
    device const int *destinations [[buffer(SEISMIC_BUFFER_DESTINATIONS)]],
    SEISMIC_OUT_HISTORY_KEY_PARAM(history_key),
    SEISMIC_OUT_HISTORY_VALUE_PARAM(history_value),
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const ulong q_head = group.x;
    const ulong row = group.y;
    const ulong G = SEISMIC_DIM_G;
    const ulong kv_head = q_head / G;
    const ulong W = SEISMIC_DIM_W;
    const ulong R = SEISMIC_DIM_R;
    const ulong EXT = SEISMIC_DIM_EXT;
    const ulong U = SEISMIC_DIM_U;
    const ulong SK = SEISMIC_DIM_SK;
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    const int gate_function = int(SEISMIC_PARAM_GATE_FUNCTION);
    const bool fast_w128 = (W == 128u && SEISMIC_HISTORY_KEY_STRIDE_2 == 1u && SEISMIC_HISTORY_VALUE_STRIDE_2 == 1u);

    if ((q_head % G) == 0) {
        rel_bias_attn::append_kv(key, value, destinations, history_key, history_value,
            row, kv_head, W, tid, seismic_words);
    }

    const long total_keys = rel_bias_attn::count_total_keys(visible, fresh, row, R, seismic_words);
    float q_reg[rel_bias_attn::MAX_REGS];
    float acc[rel_bias_attn::MAX_REGS];
    const ulong q_base = row * SEISMIC_QUERY_STRIDE_0 + q_head * SEISMIC_QUERY_STRIDE_1;
    if (fast_w128) {
        const ulong c0 = ulong(tid) * 4u;
        for (uint j = 0; j < 4u; ++j) {
            q_reg[j] = element::at<activation::Act>(query, q_base + (c0 + j) * SEISMIC_QUERY_STRIDE_2) * scale;
            acc[j] = 0.0f;
        }
    } else {
        for (uint r_i = 0; r_i < rel_bias_attn::MAX_REGS; ++r_i) {
            const ulong c = ulong(tid) + ulong(r_i) * 32u;
            q_reg[r_i] = c < W ? element::at<activation::Act>(query, q_base + c * SEISMIC_QUERY_STRIDE_2) * scale : 0.0f;
            acc[r_i] = 0.0f;
        }
    }
    float maximum = -INFINITY;
    float denominator = 0.0f;

    if (total_keys > 0) {
        rel_bias_attn::scan_range(q_reg, acc, maximum, denominator,
            key, value, rel_bias, visible, fresh, history_key, history_value,
            row, q_head, kv_head, W, R, EXT, total_keys, 0L, total_keys,
            tid, seismic_words);
    }

    float total = denominator;
    const uint regs = fast_w128 ? 4u : uint((W + 31u) / 32u);
    if (SK > 0) {
        const float sink = sinks[(row % SK) * SEISMIC_SINKS_STRIDE_0 + q_head * SEISMIC_SINKS_STRIDE_1];
        if (sink > maximum) {
            const float rescale = maximum > -INFINITY ? metal::exp(maximum - sink) : 0.0f;
            total = total * rescale + 1.0f;
            for (uint r_i = 0; r_i < regs; ++r_i)
                acc[r_i] *= rescale;
        } else if (maximum > -INFINITY) {
            total += metal::exp(sink - maximum);
        }
    }
    const float denom = metal::max(total, 1e-30f);
    const element::Out<activation::Act> result(result_cells);
    const ulong out_base = row * SEISMIC_RESULT_0_STRIDE_0 + q_head * SEISMIC_RESULT_0_STRIDE_1;
    const ulong g_base = row * SEISMIC_GATE_STRIDE_0 + q_head * SEISMIC_GATE_STRIDE_1;
    if (fast_w128) {
        const ulong c0 = ulong(tid) * 4u;
        for (uint j = 0; j < 4u; ++j) {
            const ulong c = c0 + j;
            float val = acc[j] / denom;
            if (U > 0) {
                const float gv = element::at<activation::Act>(gate, g_base + (c % U) * SEISMIC_GATE_STRIDE_2);
                val = rel_bias_attn::gated_column(val, gv, gate_function);
            }
            result.put(out_base + c * SEISMIC_RESULT_0_STRIDE_2, val);
        }
    } else {
        for (uint r_i = 0; r_i < regs; ++r_i) {
            const ulong c = ulong(tid) + ulong(r_i) * 32u;
            if (c < W) {
                float val = acc[r_i] / denom;
                if (U > 0) {
                    const float gv = element::at<activation::Act>(gate, g_base + (c % U) * SEISMIC_GATE_STRIDE_2);
                    val = rel_bias_attn::gated_column(val, gv, gate_function);
                }
                result.put(out_base + c * SEISMIC_RESULT_0_STRIDE_2, val);
            }
        }
    }
}
