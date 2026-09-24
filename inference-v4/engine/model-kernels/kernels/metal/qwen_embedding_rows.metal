// qwen_embedding_rows: gather one table row per token and decode it with the
// packet decoders, published in the activation type and as F32 of that value.
#include "common/projection.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
typedef packets::bf16 activation;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
typedef packets::f16 activation;
#else
#error "qwen_embedding_rows requires a bf16 or f16 activation"
#endif
#if defined(SEISMIC_TABLE_REPRESENTATION_Q4K)
typedef packets::q4k table_packet;
#define TABLE_LAYOUT {SEISMIC_TABLE_ROW_STRIDE_BYTES, SEISMIC_TABLE_PLANE_CODES_LO_ROW_OFFSET, 0, SEISMIC_TABLE_PLANE_SCALES_ROW_OFFSET, SEISMIC_TABLE_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_TABLE_REPRESENTATION_Q5K)
typedef packets::q5k table_packet;
#define TABLE_LAYOUT {SEISMIC_TABLE_ROW_STRIDE_BYTES, SEISMIC_TABLE_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_TABLE_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_TABLE_PLANE_SCALES_ROW_OFFSET, SEISMIC_TABLE_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_TABLE_REPRESENTATION_Q6K)
typedef packets::q6k table_packet;
#define TABLE_LAYOUT {SEISMIC_TABLE_ROW_STRIDE_BYTES, SEISMIC_TABLE_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_TABLE_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_TABLE_PLANE_SCALES_ROW_OFFSET, SEISMIC_TABLE_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_TABLE_REPRESENTATION_Q8G32S)
typedef packets::q8 table_packet;
#define TABLE_LAYOUT {SEISMIC_TABLE_ROW_STRIDE_BYTES, SEISMIC_TABLE_PLANE_CODES_ROW_OFFSET, 0, 0, SEISMIC_TABLE_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_TABLE_REPRESENTATION_BF16)
typedef packets::dense<packets::bf16> table_packet;
#define TABLE_LAYOUT {SEISMIC_TABLE_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_TABLE_REPRESENTATION_F16)
typedef packets::dense<packets::f16> table_packet;
#define TABLE_LAYOUT {SEISMIC_TABLE_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_TABLE_REPRESENTATION_F32)
typedef packets::dense<packets::f32> table_packet;
#define TABLE_LAYOUT {SEISMIC_TABLE_STRIDE_0 * 4, 0, 0, 0, 0}
#else
#error "unsupported qwen_embedding_rows table representation"
#endif
#if defined(SEISMIC_TABLE_KIND_PACKED) && !defined(SEISMIC_TABLE_LAYOUT_ROWS16)
#error "qwen_embedding_rows requires the rows16 layout for table"
#endif

kernel void qwen_embedding_rows(
    device const uchar *table [[buffer(SEISMIC_BUFFER_TABLE)]],
    device const int *tokens [[buffer(SEISMIC_BUFFER_TOKENS)]],
    device uchar *embedded [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device float *widened [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint width = uint(SEISMIC_DIM_D);
    projection::weight_rows<table_packet> rows{table, TABLE_LAYOUT, width, nullptr};
    uint token = uint(tokens[ulong(row) * SEISMIC_TOKENS_STRIDE_0]);
    device typename activation::storage *out =
        reinterpret_cast<device typename activation::storage *>(embedded);
    for (uint p = thread_index; p * 32u < width; p += 256u) {
        typename table_packet::packet packet = rows.packet(token, p);
        for (uint step = 0; step < 4; ++step) {
            float4 even, odd;
            table_packet::codes(packet, step, even, odd);
            for (uint i = 0; i < 8; ++i) {
                uint column = 32u * p + 8u * step + i;
                if (column < width) {
                    float code = (i & 1u) ? odd[i >> 1] : even[i >> 1];
                    float value = activation::round(table_packet::value(packet, step, code));
                    out[ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + ulong(column) * SEISMIC_RESULT_0_STRIDE_1] =
                        activation::store(value);
                    widened[ulong(row) * SEISMIC_RESULT_1_STRIDE_0 + ulong(column) * SEISMIC_RESULT_1_STRIDE_1] = value;
                }
            }
        }
    }
}
