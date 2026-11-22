// corrector_attention: one threadgroup of 256 threads (8 simdgroups) per
// (node, head). Each active simdgroup caches its slice of the biased query
// and key bias in per-lane registers (using 128-bit `packed_float4` loads
// when unit-strided), then scores a strided set of the block's keys into
// shared memory while tracking its local maximum; the threadgroup reduces
// their maximum and softmax denominator, then each thread accumulates whole
// columns of the weighted biased values, published to A.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"
#include "lib/core/reduce.h"

typedef activation::Act activation_element;

#define CORRECTOR_THREADS 256u
#define CORRECTOR_SIMDGROUPS (CORRECTOR_THREADS / 32u)

kernel void corrector_attention(
    device const float *qkv [[buffer(SEISMIC_BUFFER_QKV)]],
    device const float *input_bias [[buffer(SEISMIC_BUFFER_INPUT_BIAS)]],
    device const float *bias [[buffer(SEISMIC_BUFFER_BIAS)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const element::Out<activation_element> result(result_cells);
    const ulong nodes = SEISMIC_DIM_N, width = SEISMIC_DIM_W;
    // The block's scores, then the reduction partials (N * 4 + 64 bytes).
    threadgroup float *scores = shared;
    threadgroup float *partials = shared + nodes;
    const ulong row = ulong(group.x), head = ulong(group.y);
    const ulong block = row / nodes, query_node = row % nodes;
    device const float *block_rows = qkv + block * SEISMIC_QKV_STRIDE_0;
    device const float *query = block_rows + query_node * SEISMIC_QKV_STRIDE_1 + head * SEISMIC_QKV_STRIDE_3;
    device const float *query_bias = input_bias + head * SEISMIC_INPUT_BIAS_STRIDE_1;
    device const float *key_bias = query_bias + SEISMIC_INPUT_BIAS_STRIDE_0;
    device const float *value_bias = key_bias + SEISMIC_INPUT_BIAS_STRIDE_0;
    device const float *bias_row = bias + head * SEISMIC_BIAS_STRIDE_0 + query_node * SEISMIC_BIAS_STRIDE_1;
    const float scale = 1.0f / metal::sqrt(float(width));
    float local = -INFINITY;
    if (ulong(sg) < nodes) {
        const ulong q_stride = SEISMIC_QKV_STRIDE_4;
        const ulong b_stride = SEISMIC_INPUT_BIAS_STRIDE_2;
        if (q_stride == 1ul && b_stride == 1ul) {
            constexpr uint VEC4_REG_CAP = 4u;
            constexpr ulong VEC4_COUNT = ulong(SEISMIC_DIM_W) / 4ul;
            constexpr ulong VEC4_FULL_CHUNKS = (VEC4_COUNT + 31ul) / 32ul;
            constexpr uint VEC4_CHUNKS =
                VEC4_FULL_CHUNKS < ulong(VEC4_REG_CAP) ? uint(VEC4_FULL_CHUNKS) : VEC4_REG_CAP;
            constexpr uint VEC4_ARRAY_LEN = VEC4_CHUNKS > 0u ? VEC4_CHUNKS : 1u;
            constexpr ulong TAIL_START = VEC4_COUNT * 4ul;
            constexpr ulong TAIL = ulong(SEISMIC_DIM_W) - TAIL_START;

            float4 q_reg[VEC4_ARRAY_LEN];
            float4 kb_reg[VEC4_ARRAY_LEN];
            _Pragma("clang loop unroll(full)")
            for (uint t = 0u; t < VEC4_CHUNKS; ++t) {
                const ulong v = ulong(t) * 32ul + ulong(lane);
                if (v < VEC4_COUNT) {
                    const ulong c = v * 4ul;
                    q_reg[t] = float4(*reinterpret_cast<device const packed_float4 *>(query + c)) +
                               float4(*reinterpret_cast<device const packed_float4 *>(query_bias + c));
                    kb_reg[t] = float4(*reinterpret_cast<device const packed_float4 *>(key_bias + c));
                }
            }
            float q_tail = 0.0f;
            float kb_tail = 0.0f;
            if (TAIL > 0ul && ulong(lane) < TAIL) {
                const ulong c = TAIL_START + ulong(lane);
                q_tail = query[c] + query_bias[c];
                kb_tail = key_bias[c];
            }

            for (ulong key = sg; key < nodes; key += CORRECTOR_SIMDGROUPS) {
                device const float *k =
                    block_rows + key * SEISMIC_QKV_STRIDE_1 + SEISMIC_QKV_STRIDE_2 + head * SEISMIC_QKV_STRIDE_3;
                float4 dot4 = float4(0.0f);
                _Pragma("clang loop unroll(full)")
                for (uint t = 0u; t < VEC4_CHUNKS; ++t) {
                    const ulong v = ulong(t) * 32ul + ulong(lane);
                    if (v < VEC4_COUNT) {
                        const float4 kv =
                            float4(*reinterpret_cast<device const packed_float4 *>(k + v * 4ul)) + kb_reg[t];
                        dot4 = metal::fma(q_reg[t], kv, dot4);
                    }
                }
                for (ulong v = ulong(VEC4_CHUNKS) * 32ul + ulong(lane); v < VEC4_COUNT; v += 32ul) {
                    const ulong c = v * 4ul;
                    const float4 qv = float4(*reinterpret_cast<device const packed_float4 *>(query + c)) +
                                      float4(*reinterpret_cast<device const packed_float4 *>(query_bias + c));
                    const float4 kv = float4(*reinterpret_cast<device const packed_float4 *>(k + c)) +
                                      float4(*reinterpret_cast<device const packed_float4 *>(key_bias + c));
                    dot4 = metal::fma(qv, kv, dot4);
                }
                float dot = (dot4.x + dot4.y) + (dot4.z + dot4.w);
                if (TAIL > 0ul && ulong(lane) < TAIL)
                    dot = metal::fma(q_tail, k[TAIL_START + ulong(lane)] + kb_tail, dot);
                dot = simd_sum(dot);
                if (lane == 0u) {
                    const float score = dot * scale + bias_row[key * SEISMIC_BIAS_STRIDE_2];
                    scores[key] = score;
                    local = metal::max(local, score);
                }
            }
        } else {
            constexpr uint SCALAR_REG_CAP = 16u;
            constexpr ulong SCALAR_FULL_CHUNKS = (ulong(SEISMIC_DIM_W) + 31ul) / 32ul;
            constexpr uint SCALAR_CHUNKS =
                SCALAR_FULL_CHUNKS < ulong(SCALAR_REG_CAP) ? uint(SCALAR_FULL_CHUNKS) : SCALAR_REG_CAP;
            constexpr uint SCALAR_ARRAY_LEN = SCALAR_CHUNKS > 0u ? SCALAR_CHUNKS : 1u;

            float q_reg[SCALAR_ARRAY_LEN];
            float kb_reg[SCALAR_ARRAY_LEN];
            _Pragma("clang loop unroll(full)")
            for (uint t = 0u; t < SCALAR_CHUNKS; ++t) {
                const ulong c = ulong(t) * 32ul + ulong(lane);
                if (c < width) {
                    q_reg[t] = query[c * q_stride] + query_bias[c * b_stride];
                    kb_reg[t] = key_bias[c * b_stride];
                }
            }

            for (ulong key = sg; key < nodes; key += CORRECTOR_SIMDGROUPS) {
                device const float *k =
                    block_rows + key * SEISMIC_QKV_STRIDE_1 + SEISMIC_QKV_STRIDE_2 + head * SEISMIC_QKV_STRIDE_3;
                float dot = 0.0f;
                _Pragma("clang loop unroll(full)")
                for (uint t = 0u; t < SCALAR_CHUNKS; ++t) {
                    const ulong c = ulong(t) * 32ul + ulong(lane);
                    if (c < width)
                        dot = metal::fma(q_reg[t], k[c * q_stride] + kb_reg[t], dot);
                }
                for (ulong c = ulong(SCALAR_CHUNKS) * 32ul + ulong(lane); c < width; c += 32ul) {
                    dot = metal::fma(
                        query[c * q_stride] + query_bias[c * b_stride], k[c * q_stride] + key_bias[c * b_stride], dot);
                }
                dot = simd_sum(dot);
                if (lane == 0u) {
                    const float score = dot * scale + bias_row[key * SEISMIC_BIAS_STRIDE_2];
                    scores[key] = score;
                    local = metal::max(local, score);
                }
            }
        }
    }
    // `group_max`'s barriers also publish every `scores[key]` to the threadgroup.
    const float maximum = reduce::group_max<CORRECTOR_SIMDGROUPS>(local, partials, sg, lane);
    float partial = 0.0f;
    for (ulong key = thread_index; key < nodes; key += CORRECTOR_THREADS) {
        const float weight = metal::exp(scores[key] - maximum);
        scores[key] = weight;
        partial += weight;
    }
    // The sum's barriers also publish every weight to the threadgroup.
    const float total = reduce::group_sum<CORRECTOR_SIMDGROUPS>(partial, partials, sg, lane);
    const float inv = total > 0.0f ? 1.0f / total : 0.0f;
    const ulong out_base = block * SEISMIC_RESULT_0_STRIDE_0 + query_node * SEISMIC_RESULT_0_STRIDE_1 +
        head * SEISMIC_RESULT_0_STRIDE_2;
    device const float *values = block_rows + 2u * SEISMIC_QKV_STRIDE_2 + head * SEISMIC_QKV_STRIDE_3;
    for (ulong column = thread_index; column < width; column += CORRECTOR_THREADS) {
        const float vb = value_bias[column * SEISMIC_INPUT_BIAS_STRIDE_2];
        device const float *v_col = values + column * SEISMIC_QKV_STRIDE_4;
        float acc = 0.0f;
        for (ulong key = 0; key < nodes; ++key)
            acc = metal::fma(scores[key], v_col[key * SEISMIC_QKV_STRIDE_1] + vb, acc);
        result.put(out_base + column * SEISMIC_RESULT_0_STRIDE_3, acc * inv);
    }
}
