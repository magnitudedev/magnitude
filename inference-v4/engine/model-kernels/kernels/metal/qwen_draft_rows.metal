// qwen_draft_rows: the draft head's input row. One threadgroup per (32
// outputs, row): it decodes the successor token's embedding row and reads the
// conditioning row, RMS-normalizes both into the joined [2D] input (rounded
// to A, in threadgroup memory), then each simdgroup reduces four combine rows
// over the joined input. Weights use the `rows16` packet library.
#include "common/projection.h"

#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
typedef packets::bf16 activation;
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
typedef packets::f16 activation;
#else
#error "qwen_draft_rows requires a bf16 or f16 activation"
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
#error "unsupported qwen_draft_rows table representation"
#endif
#if defined(SEISMIC_TABLE_KIND_PACKED) && !defined(SEISMIC_TABLE_LAYOUT_ROWS16)
#error "qwen_draft_rows requires the rows16 layout for table"
#endif

#if defined(SEISMIC_COMBINE_REPRESENTATION_Q4K)
typedef packets::q4k combine_packet;
#define COMBINE_LAYOUT {SEISMIC_COMBINE_ROW_STRIDE_BYTES, SEISMIC_COMBINE_PLANE_CODES_LO_ROW_OFFSET, 0, SEISMIC_COMBINE_PLANE_SCALES_ROW_OFFSET, SEISMIC_COMBINE_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_COMBINE_REPRESENTATION_Q5K)
typedef packets::q5k combine_packet;
#define COMBINE_LAYOUT {SEISMIC_COMBINE_ROW_STRIDE_BYTES, SEISMIC_COMBINE_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_COMBINE_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_COMBINE_PLANE_SCALES_ROW_OFFSET, SEISMIC_COMBINE_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_COMBINE_REPRESENTATION_Q6K)
typedef packets::q6k combine_packet;
#define COMBINE_LAYOUT {SEISMIC_COMBINE_ROW_STRIDE_BYTES, SEISMIC_COMBINE_PLANE_CODES_LO_ROW_OFFSET, SEISMIC_COMBINE_PLANE_CODES_HI_ROW_OFFSET, SEISMIC_COMBINE_PLANE_SCALES_ROW_OFFSET, SEISMIC_COMBINE_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_COMBINE_REPRESENTATION_Q8G32S)
typedef packets::q8 combine_packet;
#define COMBINE_LAYOUT {SEISMIC_COMBINE_ROW_STRIDE_BYTES, SEISMIC_COMBINE_PLANE_CODES_ROW_OFFSET, 0, 0, SEISMIC_COMBINE_PLANE_SUPERS_ROW_OFFSET}
#elif defined(SEISMIC_COMBINE_REPRESENTATION_BF16)
typedef packets::dense<packets::bf16> combine_packet;
#define COMBINE_LAYOUT {SEISMIC_COMBINE_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_COMBINE_REPRESENTATION_F16)
typedef packets::dense<packets::f16> combine_packet;
#define COMBINE_LAYOUT {SEISMIC_COMBINE_STRIDE_0 * 2, 0, 0, 0, 0}
#elif defined(SEISMIC_COMBINE_REPRESENTATION_F32)
typedef packets::dense<packets::f32> combine_packet;
#define COMBINE_LAYOUT {SEISMIC_COMBINE_STRIDE_0 * 4, 0, 0, 0, 0}
#else
#error "unsupported qwen_draft_rows combine representation"
#endif
#if defined(SEISMIC_COMBINE_KIND_PACKED) && !defined(SEISMIC_COMBINE_LAYOUT_ROWS16)
#error "qwen_draft_rows requires the rows16 layout for combine"
#endif

#if defined(SEISMIC_EMBEDDING_NORM_REPRESENTATION_F32)
typedef packets::f32 embedding_norm_element;
#elif defined(SEISMIC_EMBEDDING_NORM_REPRESENTATION_F16)
typedef packets::f16 embedding_norm_element;
#elif defined(SEISMIC_EMBEDDING_NORM_REPRESENTATION_BF16)
typedef packets::bf16 embedding_norm_element;
#else
#error "qwen_draft_rows requires a dense embedding norm"
#endif
#if defined(SEISMIC_HIDDEN_NORM_REPRESENTATION_F32)
typedef packets::f32 hidden_norm_element;
#elif defined(SEISMIC_HIDDEN_NORM_REPRESENTATION_F16)
typedef packets::f16 hidden_norm_element;
#elif defined(SEISMIC_HIDDEN_NORM_REPRESENTATION_BF16)
typedef packets::bf16 hidden_norm_element;
#else
#error "qwen_draft_rows requires a dense hidden norm"
#endif

constant constexpr uint draft_threads = 256;
constant constexpr uint draft_simdgroups = draft_threads / 32;
constant constexpr uint draft_rows_per_simdgroup = 4;
constant constexpr uint draft_outputs = draft_simdgroups * draft_rows_per_simdgroup;

// Sum of one value per thread in a fixed order: simdgroup sums, then the
// simdgroup partials in index order (every thread reads the same total).
inline float draft_sum(float value, threadgroup float *partials, uint sg, uint lane) {
    value = simd_sum(value);
    if (lane == 0)
        partials[sg] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint j = 0; j < draft_simdgroups; ++j)
        total += partials[j];
    return total;
}

kernel void qwen_draft_rows(
    device const int *tokens [[buffer(SEISMIC_BUFFER_TOKENS)]],
    device const uchar *table [[buffer(SEISMIC_BUFFER_TABLE)]],
    device const uchar *conditioning [[buffer(SEISMIC_BUFFER_CONDITIONING)]],
    device const uchar *embedding_norm [[buffer(SEISMIC_BUFFER_EMBEDDING_NORM)]],
    device const uchar *hidden_norm [[buffer(SEISMIC_BUFFER_HIDDEN_NORM)]],
    device const uchar *combine [[buffer(SEISMIC_BUFFER_COMBINE)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    typedef typename activation::storage storage;
    const uint width = uint(SEISMIC_DIM_D);
    const ulong row = ulong(group.y);
    threadgroup storage *joined = reinterpret_cast<threadgroup storage *>(shared);
    threadgroup float *partials = reinterpret_cast<threadgroup float *>(shared + 4ul * width);
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));

    // The embedding row in A, packet by packet, and its sum of squares.
    projection::weight_rows<table_packet> rows{table, TABLE_LAYOUT, width, nullptr};
    const uint token = uint(tokens[row * SEISMIC_TOKENS_STRIDE_0]);
    float embedding_squares = 0.0f;
    for (uint p = thread_index; p * 32u < width; p += draft_threads) {
        typename table_packet::packet packet = rows.packet(token, p);
        for (uint step = 0; step < 4; ++step) {
            float4 even, odd;
            table_packet::codes(packet, step, even, odd);
            for (uint i = 0; i < 8; ++i) {
                const uint column = 32u * p + 8u * step + i;
                if (column < width) {
                    const float code = (i & 1u) ? odd[i >> 1] : even[i >> 1];
                    const float value = activation::round(table_packet::value(packet, step, code));
                    joined[column] = activation::store(value);
                    embedding_squares = metal::fma(value, value, embedding_squares);
                }
            }
        }
    }
    // The conditioning row (already A) and its sum of squares.
    device const storage *condition = reinterpret_cast<device const storage *>(conditioning);
    float hidden_squares = 0.0f;
    for (uint column = thread_index; column < width; column += draft_threads) {
        const storage stored = condition[row * SEISMIC_CONDITIONING_STRIDE_0
            + ulong(column) * SEISMIC_CONDITIONING_STRIDE_1];
        const float value = activation::load(stored);
        joined[width + column] = stored;
        hidden_squares = metal::fma(value, value, hidden_squares);
    }
    const float embedding_inverse =
        metal::rsqrt(draft_sum(embedding_squares, partials, sg, lane) / float(width) + epsilon);
    const float hidden_inverse =
        metal::rsqrt(draft_sum(hidden_squares, partials + draft_simdgroups, sg, lane) / float(width) + epsilon);
    // Both sums' barriers ordered every joined write before these reads.
    for (uint column = thread_index; column < width; column += draft_threads) {
        const float embedded = activation::load(joined[column]) * embedding_inverse
            * packets::vector_at<embedding_norm_element>(embedding_norm, ulong(column) * SEISMIC_EMBEDDING_NORM_STRIDE_0);
        const float hidden = activation::load(joined[width + column]) * hidden_inverse
            * packets::vector_at<hidden_norm_element>(hidden_norm, ulong(column) * SEISMIC_HIDDEN_NORM_STRIDE_0);
        joined[column] = activation::store(embedded);
        joined[width + column] = activation::store(hidden);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Four combine rows per simdgroup; lanes own packets of the 2D inputs.
    const uint inputs = 2u * width;
    projection::weight_rows<combine_packet> weights{combine, COMBINE_LAYOUT, inputs, nullptr};
    for (uint r = 0; r < draft_rows_per_simdgroup; ++r) {
        const uint output = group.x * draft_outputs + sg * draft_rows_per_simdgroup + r;
        if (output >= width)
            break;
        float sum = 0.0f;
        for (uint p = lane; p * 32u < inputs; p += 32u) {
            typename combine_packet::packet packet = weights.packet(output, p);
            for (uint step = 0; step < 4; ++step) {
                float4 even, odd;
                combine_packet::codes(packet, step, even, odd);
                for (uint i = 0; i < 8; ++i) {
                    const uint column = 32u * p + 8u * step + i;
                    if (column < inputs) {
                        const float code = (i & 1u) ? odd[i >> 1] : even[i >> 1];
                        sum = metal::fma(combine_packet::value(packet, step, code),
                            activation::load(joined[column]), sum);
                    }
                }
            }
        }
        sum = simd_sum(sum);
        if (lane == 0)
            result[row * SEISMIC_RESULT_0_STRIDE_0 + ulong(output) * SEISMIC_RESULT_0_STRIDE_1] = sum;
    }
}
