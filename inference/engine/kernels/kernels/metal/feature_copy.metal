// feature_copy: out[row, column] = input[row, column], a copy of the op's
// input through the owner of A.
#include <seismic/element.h>

kernel void feature_copy(
    device const uchar *input [[buffer(SEISMIC_BUFFER_INPUT)]],
    SEISMIC_OUT_OUT_PARAM(out),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_M * SEISMIC_DIM_D) return;
    static_assert(ELEMENT_KIND(SEISMIC_INPUT) == ELEMENT_KIND(SEISMIC_OUT), "feature_copy copies one element type");
    typedef ELEMENT_OF(SEISMIC_OUT) element_type;
    ulong row = index / SEISMIC_DIM_D;
    ulong column = index % SEISMIC_DIM_D;
    element::Out<element_type>(out).copy_stored(row * SEISMIC_OUT_STRIDE_0 + column * SEISMIC_OUT_STRIDE_1,
        reinterpret_cast<device const typename element_type::storage *>(input),
        row * SEISMIC_INPUT_STRIDE_0 + column * SEISMIC_INPUT_STRIDE_1, 1);
}
