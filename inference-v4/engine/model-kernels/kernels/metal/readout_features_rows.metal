// readout_features_rows: the final RMS normalization of the `out_rows` hidden
// rows, published in the activation type. One threadgroup per output row.
#include "lib/projection/projection.h"
#include "lib/core/reduce.h"

typedef element::Act activation;
static_assert(activation::bytes == 2, "readout_features_rows requires a bf16 or f16 activation");
typedef ELEMENT_OF(SEISMIC_NORM) norm_element;

kernel void readout_features_rows(
    device const float *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device const int *out_rows [[buffer(SEISMIC_BUFFER_OUT_ROWS)]],
    device uchar *features [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[8];
    const uint width = uint(SEISMIC_DIM_D);
    projection::Rms<activation, norm_element, projection::SelectedRows> in{hidden, SEISMIC_HIDDEN_STRIDE_0,
        SEISMIC_HIDDEN_STRIDE_1, norm, SEISMIC_NORM_STRIDE_0,
        as_type<float>(uint(SEISMIC_PARAM_EPSILON)), width, {out_rows}};
    float squares = 0.0f;
    for (uint i = thread_index; i < width; i += 256u) {
        float v = in.norm_input(row, 0, i);
        squares = metal::fma(v, v, squares);
    }
    const float total = reduce::group_sum<8>(squares, partials, sg, lane);
    const float inverse = metal::rsqrt(total / float(width) + in.epsilon());
    device typename activation::storage *out = reinterpret_cast<device typename activation::storage *>(features);
    for (uint i = thread_index; i < width; i += 256u)
        out[ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + ulong(i) * SEISMIC_RESULT_0_STRIDE_1] =
            activation::store(in.value(row, i, inverse));
}
