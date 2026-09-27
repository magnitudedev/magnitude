#include "lib/core/activation.h"

typedef ELEMENT_OF(SEISMIC_ELEMENT_E) Source;
typedef ELEMENT_OF(SEISMIC_ELEMENT_U) Destination;

// One thread per element of the [B, N, K] view, in row-major order.
kernel void import_dense(
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device uchar *destination [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_B * SEISMIC_DIM_N * SEISMIC_DIM_K) return;
    ulong k = index % SEISMIC_DIM_K;
    ulong n = index / SEISMIC_DIM_K % SEISMIC_DIM_N;
    ulong b = index / SEISMIC_DIM_K / SEISMIC_DIM_N;
    element::put<Destination>(destination,
        b * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1 + k * SEISMIC_RESULT_0_STRIDE_2,
        element::at<Source>(source,
            b * SEISMIC_SOURCE_STRIDE_0 + n * SEISMIC_SOURCE_STRIDE_1 + k * SEISMIC_SOURCE_STRIDE_2));
}
