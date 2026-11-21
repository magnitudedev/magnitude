// Read directory-selected slots, rows16 rows of the table's resident
// representation, through the simdgroup-cooperative gather decoders
// (<seismic/gather.h>), and round once to the activation element. Slot s is
// row s % P of record s / P, read where its storage holds the record
// (`records (slots)`).
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_T SEISMIC_SLOTS
#include "lib/core/activation.h"
#include "lib/core/gathered.h"
#include <seismic/gather.h>

kernel void stored_rows(
    device const ulong *slots [[buffer(SEISMIC_BUFFER_SLOTS)]],
    device const int *tokens [[buffer(SEISMIC_BUFFER_TOKENS)]],
    device const int *directory [[buffer(SEISMIC_BUFFER_DIRECTORY)]],
    SEISMIC_RESULT_0_PARAM(rows_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const element::Out<activation::Act> rows(rows_cells);
    const ulong row = group.x;
    const uint first = group.y * 256u;
    const uint width = uint(SEISMIC_DIM_W);
    if (first >= width)
        return;
    const uint count = metal::min(256u, width - first);
    const ulong out_row = row * SEISMIC_RESULT_0_STRIDE_0;
    const long held = gathered::row(directory, SEISMIC_DIRECTORY_STRIDE_0, SEISMIC_DIM_R, SEISMIC_DIM_F,
        tokens[row * SEISMIC_TOKENS_STRIDE_0]);
    // A row without a slot reads zeros (uniform across the threadgroup).
    if (held < 0) {
        if (thread_index < count)
            rows.put(out_row + ulong(first + thread_index) * SEISMIC_RESULT_0_STRIDE_1, 0.0f);
        return;
    }
    const ulong per = ulong(SEISMIC_DIM_P);
    const auto region = slab::region<const uchar, packets::space>(slots, ulong(held) / per);
    const auto table = KERNEL_T_AT(region, width);
    packets::gather_row(table, ulong(held) % per, first, count, sg, 8u, lane, [&](uint offset, float value) {
        rows.put(out_row + ulong(first + offset) * SEISMIC_RESULT_0_STRIDE_1, value);
    });
}
