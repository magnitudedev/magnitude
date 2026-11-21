// draft_convolve_input: DFlash2's half-0 grouped dynamic causal convolution of
// a sublayer's normed input rows within each draft block; 4 columns per thread
// on a (column, row) grid, summing taps in offset order.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"

typedef activation::Act activation_element;

kernel void draft_convolve_input(
    device const uchar *input [[buffer(SEISMIC_BUFFER_INPUT)]],
    device const float *dynamic [[buffer(SEISMIC_BUFFER_DYNAMIC)]],
    device const float *base [[buffer(SEISMIC_BUFFER_BASE)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 gid [[thread_position_in_grid]]) {
    const ulong channels = SEISMIC_DIM_C;
    const ulong width = SEISMIC_DIM_G * channels;
    const ulong col = ulong(gid.x) * 4ul;
    const ulong row = ulong(gid.y);
    if (col >= width || row >= SEISMIC_DIM_M) return;

    const element::Out<activation_element> result(result_cells);
    const uint channels32 = uint(channels);
    const uint col32 = uint(col);
    const uint group = col32 / channels32;
    const uint lane = col32 - group * channels32;
    const uint position = gid.y % uint(SEISMIC_PARAM_BLOCK);
    const uint k_dim = uint(SEISMIC_DIM_K);

    if (col + 4ul <= width &&
        lane + 4u <= channels32 &&
        SEISMIC_INPUT_STRIDE_1 == 1ul &&
        SEISMIC_BASE_STRIDE_2 == 1ul &&
        SEISMIC_RESULT_0_STRIDE_1 == 1ul) {
        device const activation_element::storage *input_elems =
            reinterpret_cast<device const activation_element::storage *>(input);
        device const float *base_col = base + col;
        device const float *dyn_row =
            dynamic + row * SEISMIC_DYNAMIC_STRIDE_0 + ulong(group) * SEISMIC_DYNAMIC_STRIDE_3;
        float4 sum = float4(0.0f);
        for (uint offset = 0u; offset < k_dim && offset <= position; ++offset) {
            const float dyn = dyn_row[ulong(offset) * SEISMIC_DYNAMIC_STRIDE_2];
            const float4 b = float4(*reinterpret_cast<device const packed_float4 *>(
                base_col + ulong(offset) * SEISMIC_BASE_STRIDE_1));
            const float4 inp = activation_element::load4(
                input_elems + (row - ulong(offset)) * SEISMIC_INPUT_STRIDE_0 + col);
            sum = metal::fma(b + dyn, inp, sum);
        }
        result.put4(row * SEISMIC_RESULT_0_STRIDE_0 + col, sum);
        return;
    }

    const uint end = (col + 4ul < width) ? (col32 + 4u) : uint(width);
    const ulong dyn_row_off = row * SEISMIC_DYNAMIC_STRIDE_0;
    const ulong res_row_off = row * SEISMIC_RESULT_0_STRIDE_0;
    uint g = group;
    uint l = lane;
    for (uint c = col32; c < end; ++c) {
        const ulong column = ulong(c);
        const ulong dyn_rg_off = dyn_row_off + ulong(g) * SEISMIC_DYNAMIC_STRIDE_3;
        const ulong base_c_off = column * SEISMIC_BASE_STRIDE_2;
        const ulong inp_c_off = column * SEISMIC_INPUT_STRIDE_1;
        float sum = 0.0f;
        for (uint offset = 0u; offset < k_dim && offset <= position; ++offset) {
            const float coefficient =
                base[ulong(offset) * SEISMIC_BASE_STRIDE_1 + base_c_off] +
                dynamic[dyn_rg_off + ulong(offset) * SEISMIC_DYNAMIC_STRIDE_2];
            sum = metal::fma(
                coefficient,
                element::at<activation_element>(input, (row - ulong(offset)) * SEISMIC_INPUT_STRIDE_0 + inp_c_off),
                sum);
        }
        result.put(res_row_off + column * SEISMIC_RESULT_0_STRIDE_1, sum);
        if (++l == channels32) {
            l = 0u;
            ++g;
        }
    }
}
