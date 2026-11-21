// post_norm_residual: a sandwich-norm tail over the `out_rows` rows,
// (residual + rms(projected) * norm) * scale, all F32. One threadgroup per
// output row: the square sum of the projected row, then the update.
#include "lib/core/activation.h"
#include "lib/core/reduce.h"

typedef ELEMENT_OF(SEISMIC_NORM) Norm;

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
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> result(result_cells);
    const ulong width = SEISMIC_DIM_D;
    const ulong width4 = width & ~3ul;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    device const float *p = projected + ulong(row) * SEISMIC_PROJECTED_STRIDE_0;
    float squares = 0.0f;
    if (SEISMIC_PROJECTED_STRIDE_1 == 1ul) {
        for (ulong i = ulong(thread_index) * 4ul; i < width4; i += 1024ul) {
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
    if (SEISMIC_PROJECTED_STRIDE_1 == 1ul && SEISMIC_RESIDUAL_STRIDE_1 == 1ul
        && SEISMIC_NORM_STRIDE_0 == 1ul && SEISMIC_RESULT_0_STRIDE_1 == 1ul) {
        device const Norm::storage *norm_base = reinterpret_cast<device const Norm::storage *>(norm);
        for (ulong i = ulong(thread_index) * 4ul; i < width4; i += 1024ul) {
            const float4 pv = float4(*reinterpret_cast<device const packed_float4 *>(p + i));
            const float4 rv = float4(*reinterpret_cast<device const packed_float4 *>(r + i));
            const float4 nv = Norm::load4(norm_base + i);
            const float4 normalized = (pv * inverse) * nv;
            result.put4(result_row + i, (rv + normalized) * scale);
        }
        for (ulong i = width4 + thread_index; i < width; i += 256u) {
            const float normalized = p[i] * inverse * element::at<Norm>(norm, i);
            result.put(result_row + i, (r[i] + normalized) * scale);
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
