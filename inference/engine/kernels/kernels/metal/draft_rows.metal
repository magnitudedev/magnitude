// draft_rows: normalize the successor embedding and conditioning in A, then
// project their joined [2D] row. Decode fuses these into one threadgroup per
// (32 outputs, row). Prefill joins each row once to scratch, then projects
// through the shared matrix library so a weight tile serves many rows.
#define KERNEL_W0 SEISMIC_TABLE
#define KERNEL_W1 SEISMIC_COMBINE
#include "lib/projection/projection.h"
#include "lib/core/reduce.h"

typedef element::Act activation;
static_assert(activation::bytes == 2, "draft_rows requires a bf16 or f16 activation");
typedef ELEMENT_OF(SEISMIC_EMBEDDING_NORM) embedding_norm_element;
typedef ELEMENT_OF(SEISMIC_HIDDEN_NORM) hidden_norm_element;

constant constexpr uint draft_threads = 256;
constant constexpr uint draft_simdgroups = draft_threads / 32;
constant constexpr uint draft_rows_per_simdgroup = 4;
constant constexpr uint draft_outputs = draft_simdgroups * draft_rows_per_simdgroup;

#define DRAFT_ROWS_ARGUMENTS \
    device const int *tokens [[buffer(SEISMIC_BUFFER_TOKENS)]], \
    device const uchar *table [[buffer(SEISMIC_BUFFER_TABLE)]], \
    device const uchar *conditioning [[buffer(SEISMIC_BUFFER_CONDITIONING)]], \
    device const uchar *embedding_norm [[buffer(SEISMIC_BUFFER_EMBEDDING_NORM)]], \
    device const uchar *hidden_norm [[buffer(SEISMIC_BUFFER_HIDDEN_NORM)]], \
    device const uchar *combine [[buffer(SEISMIC_BUFFER_COMBINE)]], \
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]], \
    device uchar *joined_rows [[buffer(SEISMIC_BUFFER_SCRATCH_JOINED)]], \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

inline void draft_join(
    device const int *tokens, device const uchar *table, device const uchar *conditioning,
    device const uchar *embedding_norm, device const uchar *hidden_norm,
    constant ulong *seismic_words, threadgroup uchar *shared, ulong row,
    uint thread_index, uint sg, uint lane) {
    typedef typename activation::storage storage;
    const uint width = uint(SEISMIC_DIM_D);
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

}

kernel void draft_rows(DRAFT_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    typedef typename activation::storage storage;
    const uint width = uint(SEISMIC_DIM_D);
    const ulong row = ulong(group.y);
    draft_join(tokens, table, conditioning, embedding_norm, hidden_norm, seismic_words,
        shared, row, thread_index, sg, lane);
    threadgroup storage *joined = reinterpret_cast<threadgroup storage *>(shared);
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

kernel void draft_rows_join(DRAFT_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    typedef typename activation::storage storage;
    const uint width = uint(SEISMIC_DIM_D);
    const ulong row = ulong(group.y);
    draft_join(tokens, table, conditioning, embedding_norm, hidden_norm, seismic_words,
        shared, row, thread_index, sg, lane);
    threadgroup storage *joined = reinterpret_cast<threadgroup storage *>(shared);
    device storage *out = reinterpret_cast<device storage *>(joined_rows);
    for (uint column = thread_index; column < 2u * width; column += draft_threads)
        out[row * 2ul * width + column] = joined[column];
}

kernel void draft_rows_gemm(DRAFT_ROWS_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, 64, 64);
    const uint width = uint(SEISMIC_DIM_D);
    const uint inputs = 2u * width;
    projection::Plain<activation, projection::AllRows> in{joined_rows, inputs, 1, inputs, {}};
    projection::Store<element::F32> out{reinterpret_cast<device uchar *>(result),
        SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1, 0};
    projection::Weights<packets::W1> weights{combine, KERNEL_W1_LAYOUT(inputs), inputs, nullptr};
    projection::gemm<packets::W1, 64, 64>(in, out, weights, uint(SEISMIC_DIM_M),
        width, inputs, tile.y, tile.x, shared, sg, lane);
}
