// vision_position: one thread per value, the source plus the four table
// corners scaled by their coefficients, summed in order.
#include <seismic/element.h>

typedef ELEMENT_OF(SEISMIC_TABLE) table_element;

kernel void vision_position(device const float *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device const uchar *table [[buffer(SEISMIC_BUFFER_TABLE)]],
    device const int *indices [[buffer(SEISMIC_BUFFER_INDICES)]],
    device const float *coefficients [[buffer(SEISMIC_BUFFER_COEFFICIENTS)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    const ulong columns = SEISMIC_DIM_H;
    if (index >= SEISMIC_DIM_M * columns)
        return;
    const ulong row = index / columns, column = index % columns;
    float blended = 0.0f;
    for (uint i = 0; i < 4; ++i)
        blended = blended + element::at<table_element>(table, ulong(indices[row * 4 + i]) * SEISMIC_TABLE_STRIDE_0
            + column) * coefficients[row * 4 + i];
    result[row * SEISMIC_RESULT_0_STRIDE_0 + column] = source[row * SEISMIC_SOURCE_STRIDE_0 + column] + blended;
}
