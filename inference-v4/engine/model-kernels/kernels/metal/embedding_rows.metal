// embedding_rows: gather one table row per token and decode it with the
// packet decoders, published in the activation type and as F32 of that value.
#define KERNEL_W0 SEISMIC_TABLE
#include "lib/projection/projection.h"

typedef element::Act activation;
static_assert(activation::bytes == 2, "embedding_rows requires a bf16 or f16 activation");

kernel void embedding_rows(
    device const uchar *table [[buffer(SEISMIC_BUFFER_TABLE)]],
    device const int *tokens [[buffer(SEISMIC_BUFFER_TOKENS)]],
    device uchar *embedded [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *widened [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint width = uint(SEISMIC_DIM_D);
    projection::Weights<packets::W0> rows{table, KERNEL_W0_LAYOUT(width), width, nullptr};
    // (token, status) rows; a failed selection (-1) embeds token 0.
    uint token = uint(max(tokens[ulong(row) * SEISMIC_TOKENS_STRIDE_0], 0));
    device typename activation::storage *out =
        reinterpret_cast<device typename activation::storage *>(embedded);
    for (uint p = thread_index; p * 32u < width; p += 256u) {
        typename packets::W0::packet packet = rows.packet(token, p);
        for (uint step = 0; step < 4; ++step) {
            float4 even, odd;
            packets::W0::codes(packet, step, even, odd);
            for (uint i = 0; i < 8; ++i) {
                uint column = 32u * p + 8u * step + i;
                if (column < width) {
                    float code = (i & 1u) ? odd[i >> 1] : even[i >> 1];
                    float value = activation::round(packets::W0::value(packet, step, code));
                    out[ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + ulong(column) * SEISMIC_RESULT_0_STRIDE_1] =
                        activation::store(value);
                    widened[ulong(row) * SEISMIC_RESULT_1_STRIDE_0 + ulong(column) * SEISMIC_RESULT_1_STRIDE_1] = value;
                }
            }
        }
    }
}
