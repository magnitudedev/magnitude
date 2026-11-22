// embedding_rows: gather one table row per token through the
// simdgroup-cooperative gather decoders (<seismic/gather.h>), times `scale`;
// with `normalize` the row's weightless RMS inverse is reduced and normalized
// in a single pass from per-thread register caches for widths up to
// `embedding_cache_cap * embedding_threads`, falling back to a second pass for
// wider rows. Published in the activation type and as F32 of that value.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_T SEISMIC_TABLE
#include "lib/core/activation.h"
#include "lib/core/gathered.h"
#include "lib/core/reduce.h"
#include <seismic/gather.h>

typedef activation::Act activation_element;
static_assert(activation_element::bytes == 2, "embedding_rows requires a bf16 or f16 activation");

constant constexpr uint embedding_threads = 256;
constant constexpr uint embedding_simdgroups = embedding_threads / 32;
constant constexpr uint embedding_cache_cap = 32;

kernel void embedding_rows(
    device const uchar *table [[buffer(SEISMIC_BUFFER_TABLE)]],
    device const int *tokens [[buffer(SEISMIC_BUFFER_TOKENS)]],
    device const int *directory [[buffer(SEISMIC_BUFFER_DIRECTORY)]],
    SEISMIC_RESULT_0_PARAM(embedded_cells),
    SEISMIC_RESULT_1_PARAM(widened_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[embedding_simdgroups];
    threadgroup long shared_held;
    if (thread_index == 0u) {
        shared_held = gathered::row(directory, SEISMIC_DIRECTORY_STRIDE_0, SEISMIC_DIM_R,
            SEISMIC_DIM_F, tokens[ulong(row) * SEISMIC_TOKENS_STRIDE_0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const long held = shared_held;
    const uint width = uint(SEISMIC_DIM_D);
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    const auto rows = KERNEL_T_AT(table, width);
    const element::Out<activation_element> out(embedded_cells);
    const element::Out<element::F32> widened(widened_cells);
    const ulong out0 = ulong(row) * SEISMIC_RESULT_0_STRIDE_0;
    const ulong out1 = ulong(row) * SEISMIC_RESULT_1_STRIDE_0;
    // A row that names no table row reads zeros (the whole threadgroup's).
    if (held < 0) {
        for (uint column = thread_index; column < width; column += embedding_threads) {
            out.put(out0 + ulong(column) * SEISMIC_RESULT_0_STRIDE_1, 0.0f);
            widened.put(out1 + ulong(column) * SEISMIC_RESULT_1_STRIDE_1, 0.0f);
        }
        return;
    }
    const ulong token = ulong(held);
    const bool normalize = SEISMIC_PARAM_NORMALIZE != 0;
    float inverse = 1.0f;
    if (normalize) {
        if constexpr (uint(SEISMIC_DIM_D) <= embedding_cache_cap * embedding_threads) {
            constexpr uint slots = (uint(SEISMIC_DIM_D) + embedding_threads - 1u) / embedding_threads;
            constexpr uint array_len = slots > 0u ? slots : 1u;
            float cached_val[array_len];
            uint cached_col[array_len];
            uint count = 0u;
            float squares = 0.0f;
            packets::gather_row(rows, token, 0u, width, sg, embedding_simdgroups, lane, [&](uint column, float decoded) {
                const float value = decoded * scale;
                squares = metal::fma(value, value, squares);
                cached_val[count] = value;
                cached_col[count] = column;
                ++count;
            });
            const float total = reduce::group_sum<embedding_simdgroups>(squares, partials, sg, lane);
            inverse = metal::rsqrt(total / float(width) + as_type<float>(uint(SEISMIC_PARAM_EPSILON)));
            for (uint i = 0u; i < count; ++i) {
                const float value = activation_element::round(cached_val[i] * inverse);
                const uint column = cached_col[i];
                out.put(out0 + ulong(column) * SEISMIC_RESULT_0_STRIDE_1, value);
                widened.put(out1 + ulong(column) * SEISMIC_RESULT_1_STRIDE_1, value);
            }
            return;
        } else {
            float squares = 0.0f;
            packets::gather_row(rows, token, 0u, width, sg, embedding_simdgroups, lane, [&](uint, float decoded) {
                const float value = decoded * scale;
                squares = metal::fma(value, value, squares);
            });
            const float total = reduce::group_sum<embedding_simdgroups>(squares, partials, sg, lane);
            inverse = metal::rsqrt(total / float(width) + as_type<float>(uint(SEISMIC_PARAM_EPSILON)));
        }
    }
    packets::gather_row(rows, token, 0u, width, sg, embedding_simdgroups, lane, [&](uint column, float decoded) {
        const float scaled = decoded * scale;
        const float value = activation_element::round(normalize ? scaled * inverse : scaled);
        out.put(out0 + ulong(column) * SEISMIC_RESULT_0_STRIDE_1, value);
        widened.put(out1 + ulong(column) * SEISMIC_RESULT_1_STRIDE_1, value);
    });
}
