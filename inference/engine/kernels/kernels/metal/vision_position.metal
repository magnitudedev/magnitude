// vision_position: four contiguous columns per thread on a 2-D (column, row)
// grid: the source plus the four table corners scaled by their coefficients,
// summed in order, with row-constant indices and coefficients hoisted once per
// 4-column vector.
#include <seismic/element.h>

typedef ELEMENT_OF(SEISMIC_TABLE) table_element;

kernel void vision_position(device const float *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device const uchar *table [[buffer(SEISMIC_BUFFER_TABLE)]],
    device const int *indices [[buffer(SEISMIC_BUFFER_INDICES)]],
    device const float *coefficients [[buffer(SEISMIC_BUFFER_COEFFICIENTS)]],
    SEISMIC_RESULT_0_PARAM(result),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong columns = SEISMIC_DIM_H;
    const ulong col = ulong(position.x) * 4ul;
    const ulong row = ulong(position.y);
    if (col >= columns || row >= SEISMIC_DIM_M)
        return;
    const ulong idx_row = row * SEISMIC_INDICES_STRIDE_0;
    const ulong coeff_row = row * SEISMIC_COEFFICIENTS_STRIDE_0;
    ulong table_rows[4];
    float coeffs[4];
    for (uint i = 0; i < 4; ++i) {
        table_rows[i] = ulong(indices[idx_row + ulong(i) * SEISMIC_INDICES_STRIDE_1]) * SEISMIC_TABLE_STRIDE_0;
        coeffs[i] = coefficients[coeff_row + ulong(i) * SEISMIC_COEFFICIENTS_STRIDE_1];
    }
    const ulong src_row = row * SEISMIC_SOURCE_STRIDE_0;
    const ulong out_row = row * SEISMIC_RESULT_0_STRIDE_0;
    const element::Out<element::F32> out(result);
    if (col + 4ul <= columns &&
        SEISMIC_SOURCE_STRIDE_1 == 1ul &&
        SEISMIC_TABLE_STRIDE_1 == 1ul &&
        SEISMIC_RESULT_0_STRIDE_1 == 1ul) {
        device const table_element::storage *table_base =
            reinterpret_cast<device const table_element::storage *>(table);
        float4 blended = float4(0.0f);
        for (uint i = 0; i < 4; ++i)
            blended = blended + table_element::load4(table_base + table_rows[i] + col) * coeffs[i];
        const float4 sv = float4(*reinterpret_cast<device const packed_float4 *>(source + src_row + col));
        out.put4(out_row + col, sv + blended);
    } else {
        const ulong limit = metal::min(col + 4ul, columns);
        for (ulong c = col; c < limit; ++c) {
            float blended = 0.0f;
            for (uint i = 0; i < 4; ++i)
                blended = blended + element::at<table_element>(table, table_rows[i] + c * SEISMIC_TABLE_STRIDE_1)
                    * coeffs[i];
            out.put(out_row + c * SEISMIC_RESULT_0_STRIDE_1,
                source[src_row + c * SEISMIC_SOURCE_STRIDE_1] + blended);
        }
    }
}
