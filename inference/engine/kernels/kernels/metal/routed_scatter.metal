// Prefill output of the general routed feed-forward: the grouped expert
// outputs of each row unpermuted and weighted in slot order, published as
//     base + selected.
// A thread owns 8 consecutive columns of one row: it reads each choice's
// position and weight once and the choice's 8 expert outputs as one 16-byte
// load (the `routed_combine` gather without the shared expert).

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/routed/routed.h"

// Threads of a threadgroup (8 columns each).
constant constexpr uint scatter_threads = 256;

kernel void routed_scatter(
    device const float *base [[buffer(SEISMIC_BUFFER_BASE)]],
    device const uchar *expert_output [[buffer(SEISMIC_BUFFER_EXPERT_OUTPUT)]],
    device const int *inverse [[buffer(SEISMIC_BUFFER_INVERSE)]],
    device const float *weights [[buffer(SEISMIC_BUFFER_WEIGHTS)]],
    SEISMIC_RESULT_0_PARAM(value_cells),
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
    const uint t_dim = uint(SEISMIC_DIM_T);
    const uint k_dim = uint(SEISMIC_DIM_K);
    device const int *inv_row = inverse + ulong(m) * SEISMIC_INVERSE_STRIDE_0;
    device const float *w_row = weights + ulong(m) * SEISMIC_WEIGHTS_STRIDE_0;
    float4 selected_even = float4(0.0f), selected_odd = float4(0.0f);
    for (uint k = 0; k < k_dim; ++k) {
        const uint position = uint(inv_row[ulong(k) * SEISMIC_INVERSE_STRIDE_1]);
        const float weight = w_row[ulong(k) * SEISMIC_WEIGHTS_STRIDE_1];
        const uint bucket = position / t_dim;
        const uint slot = position - bucket * t_dim;
        device const S *row = reinterpret_cast<device const S *>(expert_output)
            + ulong(bucket) * SEISMIC_EXPERT_OUTPUT_STRIDE_0
            + ulong(slot) * SEISMIC_EXPERT_OUTPUT_STRIDE_1;
        float4 even, odd;
        projection::load8_storage<A>(row, SEISMIC_EXPERT_OUTPUT_STRIDE_2, n, columns, vector, even, odd);
        selected_even = metal::fma(float4(weight), even, selected_even);
        selected_odd = metal::fma(float4(weight), odd, selected_odd);
    }
    const float4 selected_low = float4(selected_even.x, selected_odd.x, selected_even.y, selected_odd.y);
    const float4 selected_high = float4(selected_even.z, selected_odd.z, selected_even.w, selected_odd.w);
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> out_row =
        element::Out<ELEMENT_OF(SEISMIC_RESULT_0)>(value_cells).offset(ulong(m) * SEISMIC_RESULT_0_STRIDE_0);
    device const float *base_row = base + ulong(m) * SEISMIC_BASE_STRIDE_0;
    if (SEISMIC_BASE_STRIDE_1 == 1 && SEISMIC_RESULT_0_STRIDE_1 == 1 && n + 8u <= columns) {
        device const packed_float4 *from = reinterpret_cast<device const packed_float4 *>(base_row + n);
        out_row.put4(n, float4(from[0]) + selected_low);
        out_row.put4(n + 4u, float4(from[1]) + selected_high);
    } else {
        const float selected[8] = {selected_low.x, selected_low.y, selected_low.z, selected_low.w,
            selected_high.x, selected_high.y, selected_high.z, selected_high.w};
        for (uint i = 0; i < 8u && n + i < columns; ++i) {
            const uint column = n + i;
            out_row.put(ulong(column) * SEISMIC_RESULT_0_STRIDE_1,
                base_row[ulong(column) * SEISMIC_BASE_STRIDE_1] + selected[i]);
        }
    }
}
