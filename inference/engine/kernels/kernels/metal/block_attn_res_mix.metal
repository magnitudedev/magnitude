// block_attn_res_mix: depth-wise softmax attention over S F32 residual sources
// `[S, M, H]` using Q F32 query vectors `[Q, H]`. One threadgroup (256 threads = 8
// simdgroups) per (row, q), using register-cached queries, float4 loads/stores,
// and 2 barriers total across all S sources.
#include "lib/core/activation.h"
#include "lib/core/reduce.h"

kernel void block_attn_res_mix(
    device const float *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],
    device const float *queries [[buffer(SEISMIC_BUFFER_QUERIES)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong row = group.x;
    const ulong q = group.y;
    const ulong S = SEISMIC_DIM_S;
    const ulong H = SEISMIC_DIM_H;
    const float eps = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    const bool norm_v = SEISMIC_PARAM_NORMALIZE_VALUES != 0;

    // shared layout: [S * 16] simdgroup partials (8 sq + 8 dot per source) followed by [S] coeff.
    threadgroup float *sg_partials = shared;
    threadgroup float *coeff = shared + S * 16ul;

    device const float *q_row = queries + q * SEISMIC_QUERIES_STRIDE_0;
    const ulong src_s0 = SEISMIC_SOURCES_STRIDE_0;
    const ulong src_row_off = row * SEISMIC_SOURCES_STRIDE_1;
    const element::Out<element::F32> result(result_cells);
    const ulong out_base = q * SEISMIC_RESULT_0_STRIDE_0 + row * SEISMIC_RESULT_0_STRIDE_1;

    const bool fast_vec = (SEISMIC_SOURCES_STRIDE_2 == 1ul)
        && (SEISMIC_QUERIES_STRIDE_1 == 1ul)
        && (SEISMIC_RESULT_0_STRIDE_2 == 1ul)
        && ((H & 3ul) == 0ul)
        && (H <= 4096ul)
        && ((src_s0 & 3ul) == 0ul)
        && ((src_row_off & 3ul) == 0ul)
        && ((SEISMIC_QUERIES_STRIDE_0 & 3ul) == 0ul);

    if (fast_vec) {
        const ulong vecs = H >> 2;
        device const float4 *q_row4 = reinterpret_cast<device const float4 *>(q_row);
        float4 q_reg[4];
        int n_chunks = 0;
        for (int c = 0; c < 4; ++c) {
            const ulong v_idx = ulong(thread_index) + ulong(c) * 256ul;
            if (v_idx < vecs) {
                q_reg[c] = q_row4[v_idx];
                n_chunks = c + 1;
            }
        }
        for (ulong s = 0; s < S; ++s) {
            device const float4 *src4 = reinterpret_cast<device const float4 *>(sources + s * src_s0 + src_row_off);
            float sq = 0.0f;
            float dot = 0.0f;
            for (int c = 0; c < n_chunks; ++c) {
                const ulong v_idx = ulong(thread_index) + ulong(c) * 256ul;
                const float4 v = src4[v_idx];
                const float4 qr = q_reg[c];
                sq = metal::fma(v.x, v.x, sq);
                sq = metal::fma(v.y, v.y, sq);
                sq = metal::fma(v.z, v.z, sq);
                sq = metal::fma(v.w, v.w, sq);
                dot = metal::fma(qr.x, v.x, dot);
                dot = metal::fma(qr.y, v.y, dot);
                dot = metal::fma(qr.z, v.z, dot);
                dot = metal::fma(qr.w, v.w, dot);
            }
            sq = simd_sum(sq);
            dot = simd_sum(dot);
            if (lane == 0u) {
                sg_partials[s * 16ul + sg] = sq;
                sg_partials[s * 16ul + 8ul + sg] = dot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_index == 0u) {
            float inv_l[64];
            float logit_l[64];
            float maximum = -INFINITY;
            const float inv_h = 1.0f / float(H);
            for (ulong s = 0; s < S; ++s) {
                float total_sq = 0.0f;
                float total_dot = 0.0f;
                for (uint w = 0; w < 8u; ++w) {
                    total_sq += sg_partials[s * 16ul + w];
                    total_dot += sg_partials[s * 16ul + 8ul + w];
                }
                const float r = metal::rsqrt(total_sq * inv_h + eps);
                const float l = total_dot * r * scale;
                inv_l[s] = r;
                logit_l[s] = l;
                maximum = metal::max(maximum, l);
            }
            float total = 0.0f;
            for (ulong s = 0; s < S; ++s) {
                const float e = metal::exp(logit_l[s] - maximum);
                logit_l[s] = e;
                total += e;
            }
            const float inv_total = 1.0f / total;
            for (ulong s = 0; s < S; ++s) {
                const float w = logit_l[s] * inv_total;
                coeff[s] = norm_v ? w * inv_l[s] : w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = 0; c < n_chunks; ++c) {
            const ulong v_idx = ulong(thread_index) + ulong(c) * 256ul;
            float4 acc = float4(0.0f);
            for (ulong s = 0; s < S; ++s) {
                device const float4 *src4 = reinterpret_cast<device const float4 *>(sources + s * src_s0 + src_row_off);
                acc = metal::fma(float4(coeff[s]), src4[v_idx], acc);
            }
            const ulong base = out_base + (v_idx << 2);
            result.put(base + 0ul, acc.x);
            result.put(base + 1ul, acc.y);
            result.put(base + 2ul, acc.z);
            result.put(base + 3ul, acc.w);
        }
        return;
    }

    for (ulong s = 0; s < S; ++s) {
        device const float *src = sources + s * src_s0 + src_row_off;
        float sq = 0.0f;
        float dot = 0.0f;
        for (ulong i = thread_index; i < H; i += 256u) {
            const float v = src[i * SEISMIC_SOURCES_STRIDE_2];
            sq = metal::fma(v, v, sq);
            dot = metal::fma(q_row[i * SEISMIC_QUERIES_STRIDE_1], v, dot);
        }
        sq = simd_sum(sq);
        dot = simd_sum(dot);
        if (lane == 0u) {
            sg_partials[s * 16ul + sg] = sq;
            sg_partials[s * 16ul + 8ul + sg] = dot;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0u) {
        float inv_l[64];
        float logit_l[64];
        float maximum = -INFINITY;
        const float inv_h = 1.0f / float(H);
        for (ulong s = 0; s < S; ++s) {
            float total_sq = 0.0f;
            float total_dot = 0.0f;
            for (uint w = 0; w < 8u; ++w) {
                total_sq += sg_partials[s * 16ul + w];
                total_dot += sg_partials[s * 16ul + 8ul + w];
            }
            const float r = metal::rsqrt(total_sq * inv_h + eps);
            const float l = total_dot * r * scale;
            inv_l[s] = r;
            logit_l[s] = l;
            maximum = metal::max(maximum, l);
        }
        float total = 0.0f;
        for (ulong s = 0; s < S; ++s) {
            const float e = metal::exp(logit_l[s] - maximum);
            logit_l[s] = e;
            total += e;
        }
        const float inv_total = 1.0f / total;
        for (ulong s = 0; s < S; ++s) {
            const float w = logit_l[s] * inv_total;
            coeff[s] = norm_v ? w * inv_l[s] : w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ulong i = thread_index; i < H; i += 256u) {
        float acc = 0.0f;
        for (ulong s = 0; s < S; ++s) {
            const float v = sources[s * src_s0 + src_row_off + i * SEISMIC_SOURCES_STRIDE_2];
            acc = metal::fma(coeff[s], v, acc);
        }
        result.put(out_base + i * SEISMIC_RESULT_0_STRIDE_2, acc);
    }
}
