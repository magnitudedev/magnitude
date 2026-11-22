// corrector_rows: one threadgroup of 256 threads per row (4 elements per
// thread per iteration): values plus bias, then by form the identity (0),
// silu (1) or unit length (2, its squared length reduced over the
// threadgroup), plus the residual row when R = 1, published to Y.
#include <seismic/element.h>
#include "lib/core/reduce.h"

typedef ELEMENT_OF(SEISMIC_ELEMENT_Y) output_element;

#define ROWS_THREADS 256u

kernel void corrector_rows(
    device const float *values [[buffer(SEISMIC_BUFFER_VALUES)]],
    device const float *bias [[buffer(SEISMIC_BUFFER_BIAS)]],
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr ulong CACHE_MAX_N = 4096ul;
    threadgroup float partials[ROWS_THREADS / 32u];
    const element::Out<output_element> result(result_cells);
    const ulong width = SEISMIC_DIM_N;
    const ulong width4 = width & ~3ul;
    const int form = int(SEISMIC_PARAM_FORM);
    device const float *line = values + ulong(row) * SEISMIC_VALUES_STRIDE_0;
    device const float *res_line = residual + ulong(row) * SEISMIC_RESIDUAL_STRIDE_1;
    const ulong result_row = ulong(row) * SEISMIC_RESULT_0_STRIDE_0;
    const bool bias_vec = SEISMIC_VALUES_STRIDE_1 == 1ul && SEISMIC_BIAS_STRIDE_0 == 1ul;
    const bool out_vec =
        bias_vec && SEISMIC_RESULT_0_STRIDE_1 == 1ul && (SEISMIC_DIM_R == 0 || SEISMIC_RESIDUAL_STRIDE_2 == 1ul);

    // Form 2 single-read register-cached path when the entire row fits in <= 4
    // float4s per thread (N <= 4096) on the unit-strided layout.
    if constexpr (SEISMIC_DIM_N <= CACHE_MAX_N) {
        if (form == 2 && out_vec) {
            constexpr ulong STEP4 = ulong(ROWS_THREADS) * 4ul;
            constexpr ulong WIDTH4 = ulong(SEISMIC_DIM_N) & ~3ul;
            constexpr uint CACHE_VECS = WIDTH4 > 0ul ? uint((WIDTH4 + STEP4 - 1ul) / STEP4) : 1u;
            float4 cached[CACHE_VECS];
            float tail_val = 0.0f;
            float partial = 0.0f;
            _Pragma("clang loop unroll(full)")
            for (uint slot = 0; slot < CACHE_VECS; ++slot) {
                const ulong column = ulong(thread_index) * 4ul + ulong(slot) * STEP4;
                if (column < width4) {
                    const float4 biased = float4(*reinterpret_cast<device const packed_float4 *>(line + column)) +
                                          float4(*reinterpret_cast<device const packed_float4 *>(bias + column));
                    cached[slot] = biased;
                    partial = metal::fma(biased.x, biased.x, partial);
                    partial = metal::fma(biased.y, biased.y, partial);
                    partial = metal::fma(biased.z, biased.z, partial);
                    partial = metal::fma(biased.w, biased.w, partial);
                }
            }
            const ulong tail_col = width4 + thread_index;
            if (tail_col < width) {
                tail_val = line[tail_col] + bias[tail_col];
                partial = metal::fma(tail_val, tail_val, partial);
            }
            const float length = metal::sqrt(reduce::group_sum<ROWS_THREADS / 32u>(partial, partials, sg, lane));
            const float scale = 1.0f / metal::max(length, as_type<float>(uint(SEISMIC_PARAM_EPSILON)));
            _Pragma("clang loop unroll(full)")
            for (uint slot = 0; slot < CACHE_VECS; ++slot) {
                const ulong column = ulong(thread_index) * 4ul + ulong(slot) * STEP4;
                if (column < width4) {
                    float4 out = cached[slot] * scale;
                    if (SEISMIC_DIM_R == 1)
                        out += float4(*reinterpret_cast<device const packed_float4 *>(res_line + column));
                    result.put4(result_row + column, out);
                }
            }
            if (tail_col < width) {
                float out = tail_val * scale;
                if (SEISMIC_DIM_R == 1)
                    out += res_line[tail_col];
                result.put(result_row + tail_col, out);
            }
            return;
        }
    }

    float scale = 1.0f;
    if (form == 2) {
        float partial = 0.0f;
        if (bias_vec) {
            for (ulong column = ulong(thread_index) * 4ul; column < width4; column += ROWS_THREADS * 4ul) {
                const float4 biased = float4(*reinterpret_cast<device const packed_float4 *>(line + column)) +
                                      float4(*reinterpret_cast<device const packed_float4 *>(bias + column));
                partial = metal::fma(biased.x, biased.x, partial);
                partial = metal::fma(biased.y, biased.y, partial);
                partial = metal::fma(biased.z, biased.z, partial);
                partial = metal::fma(biased.w, biased.w, partial);
            }
            for (ulong column = width4 + thread_index; column < width; column += ROWS_THREADS) {
                const float biased = line[column] + bias[column];
                partial = metal::fma(biased, biased, partial);
            }
        } else {
            for (ulong column = thread_index; column < width; column += ROWS_THREADS) {
                const float biased = line[column * SEISMIC_VALUES_STRIDE_1] + bias[column * SEISMIC_BIAS_STRIDE_0];
                partial = metal::fma(biased, biased, partial);
            }
        }
        const float length = metal::sqrt(reduce::group_sum<ROWS_THREADS / 32u>(partial, partials, sg, lane));
        scale = 1.0f / metal::max(length, as_type<float>(uint(SEISMIC_PARAM_EPSILON)));
    }
    if (out_vec) {
        for (ulong column = ulong(thread_index) * 4ul; column < width4; column += ROWS_THREADS * 4ul) {
            const float4 biased = float4(*reinterpret_cast<device const packed_float4 *>(line + column)) +
                                  float4(*reinterpret_cast<device const packed_float4 *>(bias + column));
            float4 out = biased;
            if (form == 1)
                out = biased / (1.0f + metal::exp(-biased));
            else if (form == 2)
                out = biased * scale;
            if (SEISMIC_DIM_R == 1)
                out += float4(*reinterpret_cast<device const packed_float4 *>(res_line + column));
            result.put4(result_row + column, out);
        }
        for (ulong column = width4 + thread_index; column < width; column += ROWS_THREADS) {
            const float biased = line[column] + bias[column];
            float out = biased;
            if (form == 1)
                out = biased / (1.0f + metal::exp(-biased));
            else if (form == 2)
                out = biased * scale;
            if (SEISMIC_DIM_R == 1)
                out += res_line[column];
            result.put(result_row + column, out);
        }
    } else {
        for (ulong column = thread_index; column < width; column += ROWS_THREADS) {
            const float biased = line[column * SEISMIC_VALUES_STRIDE_1] + bias[column * SEISMIC_BIAS_STRIDE_0];
            float out = biased;
            if (form == 1)
                out = biased / (1.0f + metal::exp(-biased));
            else if (form == 2)
                out = biased * scale;
            if (SEISMIC_DIM_R == 1)
                out += res_line[column * SEISMIC_RESIDUAL_STRIDE_2];
            result.put(result_row + column * SEISMIC_RESULT_0_STRIDE_1, out);
        }
    }
}
