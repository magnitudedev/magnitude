// feature_copy: out[row, column] = input[row, column], a copy of the op's
// input through the owner of A. Each thread copies one 16-byte chunk (`VEC` elements) along D.
#include <seismic/element.h>

kernel void feature_copy(
    device const uchar *input [[buffer(SEISMIC_BUFFER_INPUT)]],
    SEISMIC_OUT_OUT_PARAM(out),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    static_assert(ELEMENT_KIND(SEISMIC_INPUT) == ELEMENT_KIND(SEISMIC_OUT), "feature_copy copies one element type");
    typedef ELEMENT_OF(SEISMIC_OUT) element_type;
    constexpr uint VEC = element_type::bytes >= 16u ? 1u : 16u / element_type::bytes;
    const ulong chunks_per_row = (SEISMIC_DIM_D + ulong(VEC) - 1ul) / ulong(VEC);
    const ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_M * chunks_per_row) return;
    const ulong row = index / chunks_per_row;
    const uint column = uint(index % chunks_per_row) * VEC;
    const uint count = min(VEC, uint(SEISMIC_DIM_D - ulong(column)));
    const element::Out<element_type> destination(out);
    device const typename element_type::storage *source
        = reinterpret_cast<device const typename element_type::storage *>(input);
    if (SEISMIC_OUT_STRIDE_1 == 1ul && SEISMIC_INPUT_STRIDE_1 == 1ul) {
        destination.copy_stored(row * SEISMIC_OUT_STRIDE_0 + ulong(column), source,
            row * SEISMIC_INPUT_STRIDE_0 + ulong(column), ulong(count));
    } else {
        for (uint i = 0; i < count; ++i) {
            const ulong c = ulong(column + i);
            destination.copy_stored(row * SEISMIC_OUT_STRIDE_0 + c * SEISMIC_OUT_STRIDE_1, source,
                row * SEISMIC_INPUT_STRIDE_0 + c * SEISMIC_INPUT_STRIDE_1, 1);
        }
    }
}
