// vision_position: four contiguous columns per thread on a 2-D (column, row)
// grid: the source plus the four table corners scaled by their coefficients,
// summed in order, with row-constant indices and coefficients hoisted once per
// 4-column vector.
#include <seismic/element.cuh>

using element::u64;
using element::u8;
typedef ELEMENT_OF(SEISMIC_TABLE) Table;

extern "C" __global__ void vision_position(SEISMIC_KERNEL_PARAMS) {
    const u64 columns = SEISMIC_DIM_H;
    const u64 col = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 4ull;
    if (col >= columns)
        return;
    const int *indices = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_INDICES));
    const float *coefficients = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENTS));
    const float *source = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    const u8 *table = SEISMIC_PTR(SEISMIC_BUFFER_TABLE);
    const auto result = SEISMIC_RESULT_0_OUT;
    const bool full_vec = col + 4ull <= columns &&
                          SEISMIC_SOURCE_STRIDE_1 == 1ull &&
                          SEISMIC_TABLE_STRIDE_1 == 1ull &&
                          SEISMIC_RESULT_0_STRIDE_1 == 1ull;
    constexpr u64 table_align_mask = 4ull * Table::bytes - 1ull;
    const u64 limit = col + 4ull < columns ? col + 4ull : columns;
    for (u64 row = blockIdx.y; row < SEISMIC_DIM_M; row += gridDim.y) {
        const u64 idx_row = row * SEISMIC_INDICES_STRIDE_0;
        const u64 coeff_row = row * SEISMIC_COEFFICIENTS_STRIDE_0;
        u64 table_rows[4];
        float coeffs[4];
        u64 table_bits = 0ull;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            table_rows[i] = (u64)indices[idx_row + (u64)i * SEISMIC_INDICES_STRIDE_1] * SEISMIC_TABLE_STRIDE_0;
            coeffs[i] = coefficients[coeff_row + (u64)i * SEISMIC_COEFFICIENTS_STRIDE_1];
            table_bits |= reinterpret_cast<u64>(table + table_rows[i] * Table::bytes);
        }
        const u64 src_row = row * SEISMIC_SOURCE_STRIDE_0;
        const u64 out_row = row * SEISMIC_RESULT_0_STRIDE_0;
        const bool aligned = ((reinterpret_cast<u64>(source + src_row) |
                               reinterpret_cast<u64>(result.stored(out_row))) & 15ull) == 0ull &&
                             (table_bits & table_align_mask) == 0ull;
        if (full_vec && aligned) {
            float blended[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float tv[4];
                element::span<Table, 4, false>(table, table_rows[i] + col, tv);
                const float coeff = coeffs[i];
                blended[0] += tv[0] * coeff;
                blended[1] += tv[1] * coeff;
                blended[2] += tv[2] * coeff;
                blended[3] += tv[3] * coeff;
            }
            const float4 sv = *reinterpret_cast<const float4 *>(source + src_row + col);
            float out[4];
            out[0] = sv.x + blended[0];
            out[1] = sv.y + blended[1];
            out[2] = sv.z + blended[2];
            out[3] = sv.w + blended[3];
            result.put_span<4>(out_row + col, out);
        } else {
            for (u64 c = col; c < limit; ++c) {
                float blended = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    blended += element::at<Table>(table, table_rows[i] + c * SEISMIC_TABLE_STRIDE_1) * coeffs[i];
                result.put(out_row + c * SEISMIC_RESULT_0_STRIDE_1,
                           source[src_row + c * SEISMIC_SOURCE_STRIDE_1] + blended);
            }
        }
    }
}
