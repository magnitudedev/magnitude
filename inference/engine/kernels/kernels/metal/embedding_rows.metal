// embedding_rows: gather one table row per token and decode it with the
// packet decoders, times `scale`; with `normalize` the row's weightless RMS
// inverse is reduced first (a pass over the decoded row), then the row is
// decoded again and normalized. Published in the activation type and as F32
// of that value.
#define KERNEL_W0 SEISMIC_TABLE
#include "lib/projection/projection.h"
#include "lib/core/reduce.h"

typedef element::Act activation;
static_assert(activation::bytes == 2, "embedding_rows requires a bf16 or f16 activation");

constant constexpr uint embedding_threads = 256;

kernel void embedding_rows(
    device const uchar *table [[buffer(SEISMIC_BUFFER_TABLE)]],
    device const int *tokens [[buffer(SEISMIC_BUFFER_TOKENS)]],
    device uchar *embedded [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *widened [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[embedding_threads / 32];
    const uint width = uint(SEISMIC_DIM_D);
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    projection::Weights<packets::W0> rows{table, KERNEL_W0_LAYOUT(width), width, nullptr};
    // (token, status) rows; a failed selection (-1) embeds token 0.
    uint token = uint(max(tokens[ulong(row) * SEISMIC_TOKENS_STRIDE_0], 0));
    float inverse = 1.0f;
    if (SEISMIC_PARAM_NORMALIZE != 0) {
        float squares = 0.0f;
        for (uint p = thread_index; p * 32u < width; p += embedding_threads) {
            typename packets::W0::packet packet = rows.packet(token, p);
            for (uint i = 0; i < 32u && 32u * p + i < width; ++i) {
                const float value = packets::value_at<packets::W0>(packet, i) * scale;
                squares = metal::fma(value, value, squares);
            }
        }
        const float total = reduce::group_sum<embedding_threads / 32>(squares, partials, sg, lane);
        inverse = metal::rsqrt(total / float(width) + as_type<float>(uint(SEISMIC_PARAM_EPSILON)));
    }
    device typename activation::storage *out =
        reinterpret_cast<device typename activation::storage *>(embedded);
    for (uint p = thread_index; p * 32u < width; p += embedding_threads) {
        typename packets::W0::packet packet = rows.packet(token, p);
        for (uint step = 0; step < 4; ++step) {
            float4 even, odd;
            packets::W0::codes(packet, step, even, odd);
            for (uint i = 0; i < 8; ++i) {
                uint column = 32u * p + 8u * step + i;
                if (column < width) {
                    float code = (i & 1u) ? odd[i >> 1] : even[i >> 1];
                    float decoded = packets::W0::value(packet, step, code) * scale;
                    float value = activation::round(SEISMIC_PARAM_NORMALIZE != 0 ? decoded * inverse : decoded);
                    out[ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + ulong(column) * SEISMIC_RESULT_0_STRIDE_1] =
                        activation::store(value);
                    widened[ulong(row) * SEISMIC_RESULT_1_STRIDE_0 + ulong(column) * SEISMIC_RESULT_1_STRIDE_1] = value;
                }
            }
        }
    }
}
