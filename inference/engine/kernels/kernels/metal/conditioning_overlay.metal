// conditioning_overlay: out[row, column] = input[row, column], a copy of the
// op's input through the owner; 4 elements per thread on a (column, row) grid.
#include <seismic/element.h>

kernel void conditioning_overlay(
    device const float *input [[buffer(SEISMIC_BUFFER_INPUT)]],
    SEISMIC_OUT_OUT_PARAM(out),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong width = SEISMIC_DIM_D;
    const ulong col = ulong(position.x) * 4ul;
    const ulong row = ulong(position.y);
    if (col >= width || row >= SEISMIC_DIM_M)
        return;
    const element::Out<ELEMENT_OF(SEISMIC_OUT)> dst(out);
    device const float *in_row = input + row * SEISMIC_INPUT_STRIDE_0;
    const ulong out_base = row * SEISMIC_OUT_STRIDE_0;

    if (col + 4ul <= width && SEISMIC_INPUT_STRIDE_1 == 1ul && SEISMIC_OUT_STRIDE_1 == 1ul) {
        dst.put4(out_base + col, float4(*reinterpret_cast<device const packed_float4 *>(in_row + col)));
    } else {
        const ulong end = col + 4ul < width ? col + 4ul : width;
        for (ulong c = col; c < end; ++c) {
            dst.put(out_base + c * SEISMIC_OUT_STRIDE_1, in_row[c * SEISMIC_INPUT_STRIDE_1]);
        }
    }
}
