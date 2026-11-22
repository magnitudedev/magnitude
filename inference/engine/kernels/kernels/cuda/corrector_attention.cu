// corrector_attention: one block of 256 threads per (node, head). Each warp
// caches its strided slice of the biased query and key bias in registers
// (128-bit float4 when unit-strided and 16-byte aligned), scores a strided set
// of the block's keys (warp sum plus additive bias), and tracks its running
// maximum inline so `reduce::group_max` reduces the maximum and publishes
// `scores` in one barrier pair. The block then reduces the softmax denominator
// in parallel, keeping the weights in shared memory, and a thread per column
// accumulates the biased values in key order, multiplies by the inverse total,
// and rounds once to A. Dynamic shared memory: the N scores followed by 64
// bytes of reduction partials.
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"

extern "C" __global__ void corrector_attention(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    constexpr unsigned Q_REG_CAP = 8u;
    constexpr u64 VEC4_COUNT = static_cast<u64>(SEISMIC_DIM_W) / 4ull;
    constexpr u64 FULL_VEC_CHUNKS = (VEC4_COUNT + 31ull) / 32ull;
    constexpr unsigned VEC_CHUNKS =
        FULL_VEC_CHUNKS < static_cast<u64>(Q_REG_CAP) ? static_cast<unsigned>(FULL_VEC_CHUNKS) : Q_REG_CAP;
    constexpr unsigned VEC_ARRAY_LEN = VEC_CHUNKS > 0u ? VEC_CHUNKS : 1u;

    constexpr unsigned SCALAR_REG_CAP = Q_REG_CAP * 4u;
    constexpr u64 FULL_SCALAR_CHUNKS = (static_cast<u64>(SEISMIC_DIM_W) + 31ull) / 32ull;
    constexpr unsigned SCALAR_CHUNKS = FULL_SCALAR_CHUNKS < static_cast<u64>(SCALAR_REG_CAP)
                                           ? static_cast<unsigned>(FULL_SCALAR_CHUNKS)
                                           : SCALAR_REG_CAP;
    constexpr unsigned SCALAR_ARRAY_LEN = SCALAR_CHUNKS > 0u ? SCALAR_CHUNKS : 1u;

    const float *qkv = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_QKV));
    const float *input_bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INPUT_BIAS));
    const float *bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BIAS));
    const auto result = SEISMIC_RESULT_0_OUT;
    const u64 nodes = SEISMIC_DIM_N, width = SEISMIC_DIM_W;
    extern __shared__ float shared[];
    float *scores = shared;           // [N]
    float *partials = shared + nodes; // [16] (64 bytes, >= warps)
    const u64 block = blockIdx.x / nodes, query = blockIdx.x % nodes, head = blockIdx.y;
    const unsigned warp = threadIdx.x / 32u, lane = threadIdx.x % 32u, warps = blockDim.x / 32u;

    const float *block_rows = qkv + block * SEISMIC_QKV_STRIDE_0;
    const float *head_keys = block_rows + SEISMIC_QKV_STRIDE_2 + head * SEISMIC_QKV_STRIDE_3;
    const float *query_row = block_rows + query * SEISMIC_QKV_STRIDE_1 + head * SEISMIC_QKV_STRIDE_3;
    const float *query_bias = input_bias + head * SEISMIC_INPUT_BIAS_STRIDE_1;
    const float *key_bias = query_bias + SEISMIC_INPUT_BIAS_STRIDE_0;
    const float *bias_row = bias + head * SEISMIC_BIAS_STRIDE_0 + query * SEISMIC_BIAS_STRIDE_1;

    const float scale = 1.0f / sqrtf(float(width));
    float local_max = -__int_as_float(0x7f800000);

    if (static_cast<u64>(warp) < nodes) {
        auto record_score = [&](u64 key, float dot) {
            dot = seismic_warp_sum_f32(dot);
            if (lane == 0u) {
                const float score = dot * scale + bias_row[key * SEISMIC_BIAS_STRIDE_2];
                scores[key] = score;
                local_max = fmaxf(local_max, score);
            }
        };

        const bool vec4 = SEISMIC_QKV_STRIDE_4 == 1ull && SEISMIC_INPUT_BIAS_STRIDE_2 == 1ull &&
                          (nodes <= 1ull || (SEISMIC_QKV_STRIDE_1 & 3ull) == 0ull) &&
                          ((reinterpret_cast<u64>(query_row) | reinterpret_cast<u64>(head_keys) |
                            reinterpret_cast<u64>(query_bias) | reinterpret_cast<u64>(key_bias)) &
                           15ull) == 0ull;

        if (vec4) {
            [[maybe_unused]] float4 q_reg[VEC_ARRAY_LEN];
            [[maybe_unused]] float4 kb_reg[VEC_ARRAY_LEN];
#pragma unroll
            for (unsigned t = 0; t < VEC_CHUNKS; ++t) {
                const u64 vec = static_cast<u64>(t) * 32ull + lane;
                if (vec < VEC4_COUNT) {
                    const u64 col = vec * 4ull;
                    const float4 qv = *reinterpret_cast<const float4 *>(query_row + col);
                    const float4 qbv = *reinterpret_cast<const float4 *>(query_bias + col);
                    q_reg[t] = make_float4(qv.x + qbv.x, qv.y + qbv.y, qv.z + qbv.z, qv.w + qbv.w);
                    kb_reg[t] = *reinterpret_cast<const float4 *>(key_bias + col);
                }
            }
            [[maybe_unused]] float q_tail = 0.0f, kb_tail = 0.0f;
            if constexpr (SEISMIC_DIM_W % 4 != 0) {
                const u64 tail_col = VEC4_COUNT * 4ull + lane;
                if (tail_col < width) {
                    q_tail = query_row[tail_col] + query_bias[tail_col];
                    kb_tail = key_bias[tail_col];
                }
            }
            for (u64 key = warp; key < nodes; key += warps) {
                const float *key_row = head_keys + key * SEISMIC_QKV_STRIDE_1;
                float dot = 0.0f;
#pragma unroll
                for (unsigned t = 0; t < VEC_CHUNKS; ++t) {
                    const u64 vec = static_cast<u64>(t) * 32ull + lane;
                    if (vec < VEC4_COUNT) {
                        const float4 kv = *reinterpret_cast<const float4 *>(key_row + vec * 4ull);
                        const float4 q = q_reg[t];
                        const float4 kb = kb_reg[t];
                        dot = seismic_fma_rn(q.x, kv.x + kb.x, dot);
                        dot = seismic_fma_rn(q.y, kv.y + kb.y, dot);
                        dot = seismic_fma_rn(q.z, kv.z + kb.z, dot);
                        dot = seismic_fma_rn(q.w, kv.w + kb.w, dot);
                    }
                }
                if constexpr (FULL_VEC_CHUNKS > static_cast<u64>(Q_REG_CAP)) {
                    for (u64 vec = static_cast<u64>(VEC_CHUNKS) * 32ull + lane; vec < VEC4_COUNT; vec += 32ull) {
                        const u64 col = vec * 4ull;
                        const float4 qv = *reinterpret_cast<const float4 *>(query_row + col);
                        const float4 qbv = *reinterpret_cast<const float4 *>(query_bias + col);
                        const float4 kv = *reinterpret_cast<const float4 *>(key_row + col);
                        const float4 kbv = *reinterpret_cast<const float4 *>(key_bias + col);
                        dot = seismic_fma_rn(qv.x + qbv.x, kv.x + kbv.x, dot);
                        dot = seismic_fma_rn(qv.y + qbv.y, kv.y + kbv.y, dot);
                        dot = seismic_fma_rn(qv.z + qbv.z, kv.z + kbv.z, dot);
                        dot = seismic_fma_rn(qv.w + qbv.w, kv.w + kbv.w, dot);
                    }
                }
                if constexpr (SEISMIC_DIM_W % 4 != 0) {
                    const u64 tail_col = VEC4_COUNT * 4ull + lane;
                    if (tail_col < width)
                        dot = seismic_fma_rn(q_tail, key_row[tail_col] + kb_tail, dot);
                }
                record_score(key, dot);
            }
        } else {
            float q_reg[SCALAR_ARRAY_LEN];
            float kb_reg[SCALAR_ARRAY_LEN];
#pragma unroll
            for (unsigned t = 0; t < SCALAR_CHUNKS; ++t) {
                const u64 col = static_cast<u64>(t) * 32ull + lane;
                if (col < width) {
                    q_reg[t] = query_row[col * SEISMIC_QKV_STRIDE_4] + query_bias[col * SEISMIC_INPUT_BIAS_STRIDE_2];
                    kb_reg[t] = key_bias[col * SEISMIC_INPUT_BIAS_STRIDE_2];
                }
            }
            for (u64 key = warp; key < nodes; key += warps) {
                const float *key_row = head_keys + key * SEISMIC_QKV_STRIDE_1;
                float dot = 0.0f;
#pragma unroll
                for (unsigned t = 0; t < SCALAR_CHUNKS; ++t) {
                    const u64 col = static_cast<u64>(t) * 32ull + lane;
                    if (col < width)
                        dot = seismic_fma_rn(q_reg[t], key_row[col * SEISMIC_QKV_STRIDE_4] + kb_reg[t], dot);
                }
                if constexpr (FULL_SCALAR_CHUNKS > static_cast<u64>(SCALAR_REG_CAP)) {
                    for (u64 col = static_cast<u64>(SCALAR_CHUNKS) * 32ull + lane; col < width; col += 32ull) {
                        const float q =
                            query_row[col * SEISMIC_QKV_STRIDE_4] + query_bias[col * SEISMIC_INPUT_BIAS_STRIDE_2];
                        const float k =
                            key_row[col * SEISMIC_QKV_STRIDE_4] + key_bias[col * SEISMIC_INPUT_BIAS_STRIDE_2];
                        dot = seismic_fma_rn(q, k, dot);
                    }
                }
                record_score(key, dot);
            }
        }
    }

    // `group_max`'s barriers also publish every `scores[key]` to the block.
    const float maximum = reduce::group_max(local_max, partials);
    float local_sum = 0.0f;
    for (u64 key = threadIdx.x; key < nodes; key += blockDim.x) {
        const float weight = expf(scores[key] - maximum);
        scores[key] = weight;
        local_sum += weight;
    }
    // The sum's barriers also publish every weight to the block.
    const float total = reduce::group_sum(local_sum, partials);
    const float inv = total > 0.0f ? 1.0f / total : 0.0f;
    const float *values = block_rows + 2 * SEISMIC_QKV_STRIDE_2 + head * SEISMIC_QKV_STRIDE_3;
    const float *value_bias = input_bias + 2 * SEISMIC_INPUT_BIAS_STRIDE_0 + head * SEISMIC_INPUT_BIAS_STRIDE_1;
    const u64 out_base =
        block * SEISMIC_RESULT_0_STRIDE_0 + query * SEISMIC_RESULT_0_STRIDE_1 + head * SEISMIC_RESULT_0_STRIDE_2;
    for (u64 column = threadIdx.x; column < width; column += blockDim.x) {
        const float vb = value_bias[column * SEISMIC_INPUT_BIAS_STRIDE_2];
        const float *v_col = values + column * SEISMIC_QKV_STRIDE_4;
        float out = 0.0f;
        for (u64 key = 0; key < nodes; ++key)
            out = seismic_fma_rn(scores[key], v_col[key * SEISMIC_QKV_STRIDE_1] + vb, out);
        result.put(out_base + column * SEISMIC_RESULT_0_STRIDE_3, out * inv);
    }
}
