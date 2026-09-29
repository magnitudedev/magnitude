// draft_gated_rows: result = A(silu(gate) * up); one thread per element.
#include "lib/core/activation.h"

typedef element::Act activation;

kernel void draft_gated_rows(
    device const uchar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const uchar *up [[buffer(SEISMIC_BUFFER_UP)]],
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    const ulong index = ulong(raw_index);
    const ulong width = SEISMIC_DIM_F;
    if (index >= SEISMIC_DIM_M * width) return;
    const ulong row = index / width, column = index % width;
    const float a = element::at<activation>(gate, row * SEISMIC_GATE_STRIDE_0 + column * SEISMIC_GATE_STRIDE_1);
    const float b = element::at<activation>(up, row * SEISMIC_UP_STRIDE_0 + column * SEISMIC_UP_STRIDE_1);
    element::put<activation>(result, row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1,
        a / (1.0f + metal::exp(-a)) * b);
}
