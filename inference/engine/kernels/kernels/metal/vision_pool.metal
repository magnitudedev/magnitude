// vision_pool: four contiguous columns per thread on a 2-D (column, row) grid:
// the G member rows times `weight`, summed in order, times `scale`, then
// standardized with the NS rows.
#include <seismic/element.h>

typedef ELEMENT_OF(SEISMIC_STANDARD_BIAS) standard_bias_element;
typedef ELEMENT_OF(SEISMIC_STANDARD_SCALE) standard_scale_element;

kernel void vision_pool(device const float *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device const uchar *standard_bias [[buffer(SEISMIC_BUFFER_STANDARD_BIAS)]],
    device const uchar *standard_scale [[buffer(SEISMIC_BUFFER_STANDARD_SCALE)]],
    SEISMIC_RESULT_0_PARAM(result),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong columns = SEISMIC_DIM_H;
    const ulong col = ulong(position.x) * 4ul;
    const ulong row = ulong(position.y);
    if (col >= columns || row >= SEISMIC_DIM_M)
        return;
    const float weight = as_type<float>(uint(SEISMIC_PARAM_WEIGHT));
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    const ulong src_row = row * SEISMIC_SOURCE_STRIDE_0;
    const ulong out_row = row * SEISMIC_RESULT_0_STRIDE_0;
    const element::Out<element::F32> out(result);
    if (col + 4ul <= columns &&
        SEISMIC_SOURCE_STRIDE_2 == 1ul &&
        SEISMIC_RESULT_0_STRIDE_1 == 1ul) {
        float4 sum = float4(0.0f);
        for (uint member = 0; member < SEISMIC_DIM_G; ++member) {
            const float4 sv = float4(*reinterpret_cast<device const packed_float4 *>(
                source + src_row + ulong(member) * SEISMIC_SOURCE_STRIDE_1 + col));
            sum = sum + sv * weight;
        }
        float4 value = sum * scale;
        if (SEISMIC_DIM_NS == 1) {
            const float4 sb = SEISMIC_STANDARD_BIAS_STRIDE_1 == 1ul
                ? standard_bias_element::load4(
                    reinterpret_cast<device const standard_bias_element::storage *>(standard_bias) + col)
                : float4(
                    element::at<standard_bias_element>(standard_bias, (col + 0ul) * SEISMIC_STANDARD_BIAS_STRIDE_1),
                    element::at<standard_bias_element>(standard_bias, (col + 1ul) * SEISMIC_STANDARD_BIAS_STRIDE_1),
                    element::at<standard_bias_element>(standard_bias, (col + 2ul) * SEISMIC_STANDARD_BIAS_STRIDE_1),
                    element::at<standard_bias_element>(standard_bias, (col + 3ul) * SEISMIC_STANDARD_BIAS_STRIDE_1));
            const float4 ss = SEISMIC_STANDARD_SCALE_STRIDE_1 == 1ul
                ? standard_scale_element::load4(
                    reinterpret_cast<device const standard_scale_element::storage *>(standard_scale) + col)
                : float4(
                    element::at<standard_scale_element>(standard_scale, (col + 0ul) * SEISMIC_STANDARD_SCALE_STRIDE_1),
                    element::at<standard_scale_element>(standard_scale, (col + 1ul) * SEISMIC_STANDARD_SCALE_STRIDE_1),
                    element::at<standard_scale_element>(standard_scale, (col + 2ul) * SEISMIC_STANDARD_SCALE_STRIDE_1),
                    element::at<standard_scale_element>(standard_scale, (col + 3ul) * SEISMIC_STANDARD_SCALE_STRIDE_1));
            value = (value - sb) * ss;
        }
        out.put4(out_row + col, value);
    } else {
        const ulong limit = metal::min(col + 4ul, columns);
        const ulong count = limit - col;
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint member = 0; member < SEISMIC_DIM_G; ++member) {
            const ulong member_row = src_row + ulong(member) * SEISMIC_SOURCE_STRIDE_1;
            for (ulong k = 0; k < count; ++k)
                sum[k] = sum[k] + source[member_row + (col + k) * SEISMIC_SOURCE_STRIDE_2] * weight;
        }
        for (ulong k = 0; k < count; ++k) {
            const ulong c = col + k;
            float value = sum[k] * scale;
            if (SEISMIC_DIM_NS == 1)
                value = (value - element::at<standard_bias_element>(standard_bias, c * SEISMIC_STANDARD_BIAS_STRIDE_1))
                    * element::at<standard_scale_element>(standard_scale, c * SEISMIC_STANDARD_SCALE_STRIDE_1);
            out.put(out_row + c * SEISMIC_RESULT_0_STRIDE_1, value);
        }
    }
}
