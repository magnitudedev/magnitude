// Prefill output of the general routed feed-forward: the grouped expert
// outputs of each row unpermuted and weighted in slot order, published as
//     base + selected.
// A thread owns 8 consecutive columns of one row: it reads each choice's
// position and weight once and the choice's 8 expert outputs as one 16-byte
// load (the `routed_combine` gather without the shared expert).

#include "lib/routed/routed.h"

// Threads of a threadgroup (8 columns each).
constant constexpr uint scatter_threads = 256;

kernel void routed_scatter(
    device const float *base [[buffer(SEISMIC_BUFFER_BASE)]],
    device const uchar *expert_output [[buffer(SEISMIC_BUFFER_EXPERT_OUTPUT)]],
    device const int *inverse [[buffer(SEISMIC_BUFFER_INVERSE)]],
    device const float *weights [[buffer(SEISMIC_BUFFER_WEIGHTS)]],
    device uchar *value [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    typedef routed::Act A;
    typedef typename A::storage S;
    const uint m = group.y;
    const uint n = 8u * (group.x * scatter_threads + thread_index);
    const uint columns = uint(SEISMIC_DIM_H);
    if (n >= columns)
        return;
    const bool vector = SEISMIC_EXPERT_OUTPUT_STRIDE_2 == 1 && (SEISMIC_EXPERT_OUTPUT_STRIDE_1 & 7u) == 0
        && (SEISMIC_EXPERT_OUTPUT_STRIDE_0 & 7u) == 0;
    float4 selected_even = float4(0.0f), selected_odd = float4(0.0f);
    for (ulong k = 0; k < SEISMIC_DIM_K; ++k) {
        const ulong position = ulong(inverse[m * SEISMIC_INVERSE_STRIDE_0 + k * SEISMIC_INVERSE_STRIDE_1]);
        const float weight = weights[m * SEISMIC_WEIGHTS_STRIDE_0 + k * SEISMIC_WEIGHTS_STRIDE_1];
        device const S *row = reinterpret_cast<device const S *>(expert_output)
            + (position / SEISMIC_DIM_T) * SEISMIC_EXPERT_OUTPUT_STRIDE_0
            + (position % SEISMIC_DIM_T) * SEISMIC_EXPERT_OUTPUT_STRIDE_1;
        float4 even, odd;
        projection::load8_storage<A>(row, SEISMIC_EXPERT_OUTPUT_STRIDE_2, n, columns, vector, even, odd);
        selected_even = metal::fma(float4(weight), even, selected_even);
        selected_odd = metal::fma(float4(weight), odd, selected_odd);
    }
    const float selected[8] = {selected_even.x, selected_odd.x, selected_even.y, selected_odd.y,
        selected_even.z, selected_odd.z, selected_even.w, selected_odd.w};
    typedef ELEMENT_OF(SEISMIC_ELEMENT_R) Published;
    device typename Published::storage *published = reinterpret_cast<device typename Published::storage *>(value);
    for (uint i = 0; i < 8u && n + i < columns; ++i) {
        const ulong column = n + i;
        published[m * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1] =
            Published::store(base[m * SEISMIC_BASE_STRIDE_0 + column * SEISMIC_BASE_STRIDE_1] + selected[i]);
    }
}
