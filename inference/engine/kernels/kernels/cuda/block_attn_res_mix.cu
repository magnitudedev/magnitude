// block_attn_res_mix: depth-wise softmax attention over S F32 residual sources
// `[S, M, H]` using Q F32 query vectors `[Q, H]`. One block (256 threads = 8 warps)
// per (row, q), using register-cached queries, float4 vectorized loads/stores,
// and 2 barriers total across all S sources.
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

using element::u64;

extern "C" __global__ void block_attn_res_mix(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ float shared[];
    const u64 row = blockIdx.x;
    const u64 q = blockIdx.y;
    const u64 S = SEISMIC_DIM_S;
    const u64 H = SEISMIC_DIM_H;
    const float eps = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const float scale = __uint_as_float((unsigned)SEISMIC_PARAM_SCALE);
    const bool norm_v = SEISMIC_PARAM_NORMALIZE_VALUES != 0;

    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    // shared layout: [S * 16] warp partials (8 sq + 8 dot per source) followed by [S] coeff.
    float *warp_partials = shared;
    float *coeff = shared + S * 16u;

    const float *sources = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCES));
    const float *q_row = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_QUERIES))
        + q * SEISMIC_QUERIES_STRIDE_0;
    const u64 src_s0 = SEISMIC_SOURCES_STRIDE_0;
    const u64 src_row_off = row * SEISMIC_SOURCES_STRIDE_1;
    const auto result = SEISMIC_RESULT_0_OUT.offset(
        q * SEISMIC_RESULT_0_STRIDE_0 + row * SEISMIC_RESULT_0_STRIDE_1);

    const bool fast_vec = (SEISMIC_SOURCES_STRIDE_2 == 1u)
        && (SEISMIC_QUERIES_STRIDE_1 == 1u)
        && (SEISMIC_RESULT_0_STRIDE_2 == 1u)
        && ((H & 3u) == 0u)
        && (H <= 4096u);

    if (fast_vec) {
        const u64 vecs = H >> 2;
        float q_reg[4][4];
        int n_chunks = 0;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const u64 v_idx = static_cast<u64>(threadIdx.x) + static_cast<u64>(c) * 256u;
            if (v_idx < vecs) {
                activation::load4_f32_aligned(q_row + (v_idx << 2), q_reg[c]);
                n_chunks = c + 1;
            }
        }
        for (u64 s = 0; s < S; ++s) {
            const float *src = sources + s * src_s0 + src_row_off;
            float sq = 0.0f;
            float dot = 0.0f;
            for (int c = 0; c < n_chunks; ++c) {
                const u64 v_idx = static_cast<u64>(threadIdx.x) + static_cast<u64>(c) * 256u;
                float v[4];
                activation::load4_f32_aligned(src + (v_idx << 2), v);
                sq = seismic_fma_rn(v[0], v[0], sq);
                sq = seismic_fma_rn(v[1], v[1], sq);
                sq = seismic_fma_rn(v[2], v[2], sq);
                sq = seismic_fma_rn(v[3], v[3], sq);
                dot = seismic_fma_rn(q_reg[c][0], v[0], dot);
                dot = seismic_fma_rn(q_reg[c][1], v[1], dot);
                dot = seismic_fma_rn(q_reg[c][2], v[2], dot);
                dot = seismic_fma_rn(q_reg[c][3], v[3], dot);
            }
            sq = seismic_warp_sum_f32(sq);
            dot = seismic_warp_sum_f32(dot);
            if (lane == 0u) {
                warp_partials[s * 16u + warp] = sq;
                warp_partials[s * 16u + 8u + warp] = dot;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            float inv_l[64];
            float logit_l[64];
            float maximum = -__int_as_float(0x7f800000);
            const float inv_h = 1.0f / static_cast<float>(H);
            for (u64 s = 0; s < S; ++s) {
                float total_sq = 0.0f;
                float total_dot = 0.0f;
#pragma unroll
                for (unsigned w = 0; w < 8u; ++w) {
                    total_sq += warp_partials[s * 16u + w];
                    total_dot += warp_partials[s * 16u + 8u + w];
                }
                const float r = rsqrtf(total_sq * inv_h + eps);
                const float l = total_dot * r * scale;
                inv_l[s] = r;
                logit_l[s] = l;
                maximum = fmaxf(maximum, l);
            }
            float total = 0.0f;
            for (u64 s = 0; s < S; ++s) {
                const float e = expf(logit_l[s] - maximum);
                logit_l[s] = e;
                total += e;
            }
            const float inv_total = 1.0f / total;
            for (u64 s = 0; s < S; ++s) {
                const float w = logit_l[s] * inv_total;
                coeff[s] = norm_v ? w * inv_l[s] : w;
            }
        }
        __syncthreads();
        for (int c = 0; c < n_chunks; ++c) {
            const u64 off = (static_cast<u64>(threadIdx.x) + static_cast<u64>(c) * 256u) << 2;
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (u64 s = 0; s < S; ++s) {
                const float w = coeff[s];
                float v[4];
                activation::load4_f32_aligned(sources + s * src_s0 + src_row_off + off, v);
                acc[0] = seismic_fma_rn(w, v[0], acc[0]);
                acc[1] = seismic_fma_rn(w, v[1], acc[1]);
                acc[2] = seismic_fma_rn(w, v[2], acc[2]);
                acc[3] = seismic_fma_rn(w, v[3], acc[3]);
            }
            activation::put4_aligned(result, off, acc);
        }
        return;
    }

    for (u64 s = 0; s < S; ++s) {
        const float *src = sources + s * src_s0 + src_row_off;
        float sq = 0.0f;
        float dot = 0.0f;
        for (u64 i = threadIdx.x; i < H; i += blockDim.x) {
            const float v = src[i * SEISMIC_SOURCES_STRIDE_2];
            sq = seismic_fma_rn(v, v, sq);
            dot = seismic_fma_rn(q_row[i * SEISMIC_QUERIES_STRIDE_1], v, dot);
        }
        sq = seismic_warp_sum_f32(sq);
        dot = seismic_warp_sum_f32(dot);
        if (lane == 0u) {
            warp_partials[s * 16u + warp] = sq;
            warp_partials[s * 16u + 8u + warp] = dot;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        float inv_l[64];
        float logit_l[64];
        float maximum = -__int_as_float(0x7f800000);
        const float inv_h = 1.0f / static_cast<float>(H);
        for (u64 s = 0; s < S; ++s) {
            float total_sq = 0.0f;
            float total_dot = 0.0f;
#pragma unroll
            for (unsigned w = 0; w < 8u; ++w) {
                total_sq += warp_partials[s * 16u + w];
                total_dot += warp_partials[s * 16u + 8u + w];
            }
            const float r = rsqrtf(total_sq * inv_h + eps);
            const float l = total_dot * r * scale;
            inv_l[s] = r;
            logit_l[s] = l;
            maximum = fmaxf(maximum, l);
        }
        float total = 0.0f;
        for (u64 s = 0; s < S; ++s) {
            const float e = expf(logit_l[s] - maximum);
            logit_l[s] = e;
            total += e;
        }
        const float inv_total = 1.0f / total;
        for (u64 s = 0; s < S; ++s) {
            const float w = logit_l[s] * inv_total;
            coeff[s] = norm_v ? w * inv_l[s] : w;
        }
    }
    __syncthreads();

    for (u64 i = threadIdx.x; i < H; i += blockDim.x) {
        float acc = 0.0f;
        for (u64 s = 0; s < S; ++s) {
            const float v = sources[s * src_s0 + src_row_off + i * SEISMIC_SOURCES_STRIDE_2];
            acc = seismic_fma_rn(coeff[s], v, acc);
        }
        result.put(i * SEISMIC_RESULT_0_STRIDE_2, acc);
    }
}
