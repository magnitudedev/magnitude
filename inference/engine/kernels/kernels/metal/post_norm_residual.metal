// post_norm_residual: a sandwich-norm tail over the `out_rows` rows,
// (residual + rms(projected) * norm) * scale, all F32. One threadgroup per
// output row: the square sum of the projected row, then the update. For
// unit-stride rows up to 8 float4 slots/thread (width <= 8192 at 256 threads),
// Pass 1 caches `projected` in registers so Pass 2 avoids re-reading it.
#include "lib/core/activation.h"
#include "lib/core/reduce.h"

typedef ELEMENT_OF(SEISMIC_NORM) Norm;
typedef ELEMENT_OF(SEISMIC_RESULT_0) ResultElem;

constant constexpr int CACHE_SLOTS = 8;
constant constexpr ulong STEP4 = 1024ul;

static inline void store4(
    const float4 pv, device const float *r, device const Norm::storage *norm_base,
    const element::Out<ResultElem> result, const ulong result_row,
    const ulong i, const float inverse, const float scale) {
    const float4 rv = float4(*reinterpret_cast<device const packed_float4 *>(r + i));
    const float4 nv = Norm::load4(norm_base + i);
    const float4 normalized = (pv * inverse) * nv;
    result.put4(result_row + i, (rv + normalized) * scale);
}

kernel void post_norm_residual(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const float *projected [[buffer(SEISMIC_BUFFER_PROJECTED)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[8];
    const element::Out<ResultElem> result(result_cells);
    const ulong width = SEISMIC_DIM_D;
    const ulong width4 = width & ~3ul;
    const ulong base4 = ulong(thread_index) * 4ul;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    device const float *p = projected + ulong(row) * SEISMIC_PROJECTED_STRIDE_0;
    const bool p_vec = SEISMIC_PROJECTED_STRIDE_1 == 1ul;
    const bool p_cache = p_vec && width <= STEP4 * ulong(CACHE_SLOTS);
    float4 p_buf[CACHE_SLOTS];
    float p_tail = 0.0f;
    float squares = 0.0f;
    if (p_cache) {
        #pragma unroll
        for (int k = 0; k < CACHE_SLOTS; ++k) {
            const ulong i = base4 + ulong(k) * STEP4;
            if (i < width4) {
                const float4 v = float4(*reinterpret_cast<device const packed_float4 *>(p + i));
                p_buf[k] = v;
                squares = metal::fma(v.x, v.x, squares);
                squares = metal::fma(v.y, v.y, squares);
                squares = metal::fma(v.z, v.z, squares);
                squares = metal::fma(v.w, v.w, squares);
            }
        }
        const ulong tail_i = width4 + thread_index;
        if (tail_i < width) {
            p_tail = p[tail_i];
            squares = metal::fma(p_tail, p_tail, squares);
        }
    } else if (p_vec) {
        for (ulong i = base4; i < width4; i += STEP4) {
            const float4 v = float4(*reinterpret_cast<device const packed_float4 *>(p + i));
            squares = metal::fma(v.x, v.x, squares);
            squares = metal::fma(v.y, v.y, squares);
            squares = metal::fma(v.z, v.z, squares);
            squares = metal::fma(v.w, v.w, squares);
        }
        for (ulong i = width4 + thread_index; i < width; i += 256u) {
            const float v = p[i];
            squares = metal::fma(v, v, squares);
        }
    } else {
        for (ulong i = thread_index; i < width; i += 256u) {
            const float v = p[i * SEISMIC_PROJECTED_STRIDE_1];
            squares = metal::fma(v, v, squares);
        }
    }
    const float total = reduce::group_sum<8>(squares, partials, sg, lane);
    const float inverse = metal::rsqrt(total / float(width) + epsilon);
    device const float *r = residual + ulong(out_rows[ulong(row) * SEISMIC_OUT_ROWS_STRIDE_0]) * SEISMIC_RESIDUAL_STRIDE_0;
    const ulong result_row = ulong(row) * SEISMIC_RESULT_0_STRIDE_0;
    if (p_vec && SEISMIC_RESIDUAL_STRIDE_1 == 1ul
        && SEISMIC_NORM_STRIDE_0 == 1ul && SEISMIC_RESULT_0_STRIDE_1 == 1ul) {
        device const Norm::storage *norm_base = reinterpret_cast<device const Norm::storage *>(norm);
        if (p_cache) {
            #pragma unroll
            for (int k = 0; k < CACHE_SLOTS; ++k) {
                const ulong i = base4 + ulong(k) * STEP4;
                if (i < width4) {
                    store4(p_buf[k], r, norm_base, result, result_row, i, inverse, scale);
                }
            }
            const ulong tail_i = width4 + thread_index;
            if (tail_i < width) {
                const float normalized = p_tail * inverse * element::at<Norm>(norm, tail_i);
                result.put(result_row + tail_i, (r[tail_i] + normalized) * scale);
            }
        } else {
            for (ulong i = base4; i < width4; i += STEP4) {
                const float4 pv = float4(*reinterpret_cast<device const packed_float4 *>(p + i));
                store4(pv, r, norm_base, result, result_row, i, inverse, scale);
            }
            for (ulong i = width4 + thread_index; i < width; i += 256u) {
                const float normalized = p[i] * inverse * element::at<Norm>(norm, i);
                result.put(result_row + i, (r[i] + normalized) * scale);
            }
        }
    } else {
        for (ulong i = thread_index; i < width; i += 256u) {
            const float normalized = p[i * SEISMIC_PROJECTED_STRIDE_1] * inverse
                * element::at<Norm>(norm, i * SEISMIC_NORM_STRIDE_0);
            result.put(result_row + i * SEISMIC_RESULT_0_STRIDE_1,
                (r[i * SEISMIC_RESIDUAL_STRIDE_1] + normalized) * scale);
        }
    }
}
