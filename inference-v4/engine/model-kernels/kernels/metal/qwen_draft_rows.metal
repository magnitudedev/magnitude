// qwen_draft_rows: the draft head's input row. One threadgroup per (32
// outputs, row): it decodes the successor token's embedding row and reads the
// conditioning row, RMS-normalizes both into the joined [2D] input (rounded
// to A, in threadgroup memory), then each simdgroup reduces four combine rows
// over the joined input. Weights use the `rows16` packet library.
#define KERNEL_W0 SEISMIC_TABLE
#define KERNEL_W1 SEISMIC_COMBINE
#include "common/projection.h"
#include "common/reduce.h"

typedef element::Act activation;
static_assert(activation::bytes == 2, "qwen_draft_rows requires a bf16 or f16 activation");
typedef ELEMENT_OF(SEISMIC_EMBEDDING_NORM) embedding_norm_element;
typedef ELEMENT_OF(SEISMIC_HIDDEN_NORM) hidden_norm_element;

constant constexpr uint draft_threads = 256;
constant constexpr uint draft_simdgroups = draft_threads / 32;
constant constexpr uint draft_rows_per_simdgroup = 4;
constant constexpr uint draft_outputs = draft_simdgroups * draft_rows_per_simdgroup;

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
    projection::Weights<packets::W0> rows{table, KERNEL_W0_LAYOUT(width), width, nullptr};
    // Column 0 of a (token, status) selection row; a failed selection is -1.
    const uint token = uint(max(tokens[row * SEISMIC_TOKENS_STRIDE_0], 0));
    float embedding_squares = 0.0f;
    for (uint p = thread_index; p * 32u < width; p += draft_threads) {
        typename packets::W0::packet packet = rows.packet(token, p);
        for (uint step = 0; step < 4; ++step) {
            float4 even, odd;
            packets::W0::codes(packet, step, even, odd);
            for (uint i = 0; i < 8; ++i) {
                const uint column = 32u * p + 8u * step + i;
                if (column < width) {
                    const float code = (i & 1u) ? odd[i >> 1] : even[i >> 1];
                    const float value = activation::round(packets::W0::value(packet, step, code));
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
    // Fixed-order threadgroup sums (simdgroup sums, then the simdgroup
    // partials in index order); their barriers order every joined write
    // before these reads.
    const float embedding_inverse = metal::rsqrt(
        reduce::group_sum<draft_simdgroups>(embedding_squares, partials, sg, lane) / float(width) + epsilon);
    const float hidden_inverse = metal::rsqrt(
        reduce::group_sum<draft_simdgroups>(hidden_squares, partials + draft_simdgroups, sg, lane) / float(width)
        + epsilon);
    for (uint column = thread_index; column < width; column += draft_threads) {
        const float embedded = activation::load(joined[column]) * embedding_inverse
            * element::at<embedding_norm_element>(embedding_norm, ulong(column) * SEISMIC_EMBEDDING_NORM_STRIDE_0);
        const float hidden = activation::load(joined[width + column]) * hidden_inverse
            * element::at<hidden_norm_element>(hidden_norm, ulong(column) * SEISMIC_HIDDEN_NORM_STRIDE_0);
        joined[column] = activation::store(embedded);
        joined[width + column] = activation::store(hidden);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Four combine rows per simdgroup; lanes own packets of the 2D inputs.
    const uint inputs = 2u * width;
    projection::Weights<packets::W1> weights{combine, KERNEL_W1_LAYOUT(inputs), inputs, nullptr};
    for (uint r = 0; r < draft_rows_per_simdgroup; ++r) {
        const uint output = group.x * draft_outputs + sg * draft_rows_per_simdgroup + r;
        if (output >= width)
            break;
        float sum = 0.0f;
        for (uint p = lane; p * 32u < inputs; p += 32u) {
            typename packets::W1::packet packet = weights.packet(output, p);
            for (uint step = 0; step < 4; ++step) {
                float4 even, odd;
                packets::W1::codes(packet, step, even, odd);
                for (uint i = 0; i < 8; ++i) {
                    const uint column = 32u * p + 8u * step + i;
                    if (column < inputs) {
                        const float code = (i & 1u) ? odd[i >> 1] : even[i >> 1];
                        sum = metal::fma(packets::W1::value(packet, step, code),
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
