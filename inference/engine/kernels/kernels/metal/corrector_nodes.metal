// corrector_nodes: 4 elements per thread (slot s, mask row j, candidate k,
// column) on a (column, node, slot) grid: the sum of its token, pass, feature,
// slot and rank rows with their biases, in the portable body's order.
#include <seismic/element.h>

kernel void corrector_nodes(
    device const float *token [[buffer(SEISMIC_BUFFER_TOKEN)]],
    device const float *token_bias [[buffer(SEISMIC_BUFFER_TOKEN_BIAS)]],
    device const float *pass [[buffer(SEISMIC_BUFFER_PASS)]],
    device const float *pass_bias [[buffer(SEISMIC_BUFFER_PASS_BIAS)]],
    device const float *feature [[buffer(SEISMIC_BUFFER_FEATURE)]],
    device const float *feature_bias [[buffer(SEISMIC_BUFFER_FEATURE_BIAS)]],
    device const float *slot_rows [[buffer(SEISMIC_BUFFER_SLOT)]],
    device const float *rank_rows [[buffer(SEISMIC_BUFFER_RANK)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 position [[thread_position_in_grid]]) {
    const ulong rows = SEISMIC_DIM_J, count = SEISMIC_DIM_K, width = SEISMIC_DIM_H;
    const ulong col = ulong(position.x) * 4ul;
    const ulong node = ulong(position.y);
    const ulong slot = ulong(position.z);
    if (col >= width || node >= rows * count || slot >= ulong(SEISMIC_DIM_S))
        return;
    const ulong mask = node / count;
    const ulong candidate = node % count;
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> result(result_cells);
    device const float *t_row =
        token + slot * SEISMIC_TOKEN_STRIDE_0 + mask * SEISMIC_TOKEN_STRIDE_1 + candidate * SEISMIC_TOKEN_STRIDE_2;
    device const float *p_row = pass + slot * SEISMIC_PASS_STRIDE_0 + mask * SEISMIC_PASS_STRIDE_1;
    device const float *f_row = feature + slot * SEISMIC_FEATURE_STRIDE_0 + mask * SEISMIC_FEATURE_STRIDE_1 +
                                candidate * SEISMIC_FEATURE_STRIDE_2;
    device const float *s_row = slot_rows + mask * SEISMIC_SLOT_STRIDE_0;
    device const float *r_row = rank_rows + candidate * SEISMIC_RANK_STRIDE_0;
    const ulong out_base =
        slot * SEISMIC_RESULT_0_STRIDE_0 + mask * SEISMIC_RESULT_0_STRIDE_1 + candidate * SEISMIC_RESULT_0_STRIDE_2;

    const bool unit_strides = SEISMIC_TOKEN_STRIDE_3 == 1ul && SEISMIC_PASS_STRIDE_2 == 1ul &&
                              SEISMIC_PASS_BIAS_STRIDE_0 == 1ul && SEISMIC_FEATURE_STRIDE_3 == 1ul &&
                              SEISMIC_FEATURE_BIAS_STRIDE_0 == 1ul && SEISMIC_SLOT_STRIDE_1 == 1ul &&
                              SEISMIC_RANK_STRIDE_1 == 1ul && SEISMIC_RESULT_0_STRIDE_3 == 1ul &&
                              (SEISMIC_DIM_TB == 0 || SEISMIC_TOKEN_BIAS_STRIDE_1 == 1ul);

    if (col + 4ul <= width && unit_strides) {
        float4 value = float4(*reinterpret_cast<device const packed_float4 *>(t_row + col));
        value += float4(*reinterpret_cast<device const packed_float4 *>(p_row + col));
        value += float4(*reinterpret_cast<device const packed_float4 *>(pass_bias + col));
        value += float4(*reinterpret_cast<device const packed_float4 *>(f_row + col));
        value += float4(*reinterpret_cast<device const packed_float4 *>(feature_bias + col));
        value += float4(*reinterpret_cast<device const packed_float4 *>(s_row + col));
        value += float4(*reinterpret_cast<device const packed_float4 *>(r_row + col));
        if (SEISMIC_DIM_TB == 1)
            value += float4(*reinterpret_cast<device const packed_float4 *>(token_bias + col));
        result.put4(out_base + col, value);
    } else {
        const ulong end = col + 4ul < width ? col + 4ul : width;
        for (ulong c = col; c < end; ++c) {
            float value = t_row[c * SEISMIC_TOKEN_STRIDE_3];
            value += p_row[c * SEISMIC_PASS_STRIDE_2];
            value += pass_bias[c * SEISMIC_PASS_BIAS_STRIDE_0];
            value += f_row[c * SEISMIC_FEATURE_STRIDE_3];
            value += feature_bias[c * SEISMIC_FEATURE_BIAS_STRIDE_0];
            value += s_row[c * SEISMIC_SLOT_STRIDE_1];
            value += r_row[c * SEISMIC_RANK_STRIDE_1];
            if (SEISMIC_DIM_TB == 1)
                value += token_bias[c * SEISMIC_TOKEN_BIAS_STRIDE_1];
            result.put(out_base + c * SEISMIC_RESULT_0_STRIDE_3, value);
        }
    }
}
