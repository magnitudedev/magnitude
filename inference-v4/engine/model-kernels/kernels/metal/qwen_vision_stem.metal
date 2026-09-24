// The vision patch stem: one thread per (row, output) sums both temporal
// patch projections, the bias and the bilinear position-table blend. Every
// weight is dense (packed projector weights are unsupported bindings).
#include "common/element.h"

typedef ELEMENT_OF(SEISMIC_TEMPORAL_WEIGHT_0) TemporalWeight0;
typedef ELEMENT_OF(SEISMIC_TEMPORAL_WEIGHT_1) TemporalWeight1;
typedef ELEMENT_OF(SEISMIC_BIAS) Bias;
typedef ELEMENT_OF(SEISMIC_TABLE) Table;

kernel void qwen_vision_stem(
    device const float *pixels [[buffer(SEISMIC_BUFFER_PIXELS)]],
    device const uchar *temporal_weight_0 [[buffer(SEISMIC_BUFFER_TEMPORAL_WEIGHT_0)]],
    device const uchar *temporal_weight_1 [[buffer(SEISMIC_BUFFER_TEMPORAL_WEIGHT_1)]],
    device const uchar *bias [[buffer(SEISMIC_BUFFER_BIAS)]],
    device const uchar *table [[buffer(SEISMIC_BUFFER_TABLE)]],
    device const int *indices [[buffer(SEISMIC_BUFFER_INDICES)]],
    device const float *coefficients [[buffer(SEISMIC_BUFFER_COEFFICIENTS)]],
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_M * SEISMIC_DIM_H) return;
    ulong row = index / SEISMIC_DIM_H;
    ulong output = index % SEISMIC_DIM_H;
    float projected = element::at<Bias>(bias, output * SEISMIC_BIAS_STRIDE_0);
    for (ulong y = 0; y < SEISMIC_DIM_P; ++y)
        for (ulong x = 0; x < SEISMIC_DIM_P; ++x)
            for (ulong channel = 0; channel < SEISMIC_DIM_C; ++channel) {
                ulong pixel_base = row * SEISMIC_PIXELS_STRIDE_0
                    + channel * SEISMIC_PIXELS_STRIDE_1
                    + y * SEISMIC_PIXELS_STRIDE_3 + x * SEISMIC_PIXELS_STRIDE_4;
                ulong weight_0_index = y * SEISMIC_TEMPORAL_WEIGHT_0_STRIDE_0
                    + x * SEISMIC_TEMPORAL_WEIGHT_0_STRIDE_1
                    + channel * SEISMIC_TEMPORAL_WEIGHT_0_STRIDE_2
                    + output * SEISMIC_TEMPORAL_WEIGHT_0_STRIDE_3;
                ulong weight_1_index = y * SEISMIC_TEMPORAL_WEIGHT_1_STRIDE_0
                    + x * SEISMIC_TEMPORAL_WEIGHT_1_STRIDE_1
                    + channel * SEISMIC_TEMPORAL_WEIGHT_1_STRIDE_2
                    + output * SEISMIC_TEMPORAL_WEIGHT_1_STRIDE_3;
                projected = metal::fma(
                    pixels[pixel_base],
                    element::at<TemporalWeight0>(temporal_weight_0, weight_0_index), projected);
                projected = metal::fma(
                    pixels[pixel_base + SEISMIC_PIXELS_STRIDE_2],
                    element::at<TemporalWeight1>(temporal_weight_1, weight_1_index), projected);
            }
    for (ulong corner = 0; corner < 4; ++corner) {
        int table_row = indices[row * SEISMIC_INDICES_STRIDE_0 + corner * SEISMIC_INDICES_STRIDE_1];
        float coefficient = coefficients[row * SEISMIC_COEFFICIENTS_STRIDE_0
            + corner * SEISMIC_COEFFICIENTS_STRIDE_1];
        ulong table_index = output * SEISMIC_TABLE_STRIDE_0 + ulong(table_row) * SEISMIC_TABLE_STRIDE_1;
        projected = metal::fma(element::at<Table>(table, table_index), coefficient, projected);
    }
    element::put<element::Act>(result, row * SEISMIC_RESULT_0_STRIDE_0 + output * SEISMIC_RESULT_0_STRIDE_1,
        projected);
}
