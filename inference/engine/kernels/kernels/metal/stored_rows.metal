// Read directory-selected slots, rows16 rows of the table's resident
// representation, through the packet decoders, a thread per value, and round
// once to the activation element.
#define KERNEL_W0 SEISMIC_SLOTS
#include "lib/projection/projection.h"
#include "lib/core/gathered.h"

kernel void stored_rows(
    device const uchar *slots [[buffer(SEISMIC_BUFFER_SLOTS)]],
    device const int *tokens [[buffer(SEISMIC_BUFFER_TOKENS)]],
    device const int *directory [[buffer(SEISMIC_BUFFER_DIRECTORY)]],
    device uchar *rows [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const ulong row = group.x;
    const uint column = group.y * 256u + thread_index;
    const uint width = uint(SEISMIC_DIM_W);
    if (column >= width)
        return;
    const uint slot = uint(gathered::row(directory, SEISMIC_DIRECTORY_STRIDE_0, SEISMIC_DIM_R, SEISMIC_DIM_F,
        tokens[row * SEISMIC_TOKENS_STRIDE_0]));
    projection::Weights<packets::W0> table{slots, KERNEL_W0_LAYOUT(width), width, nullptr};
    typename packets::W0::packet packet = table.packet(slot, column / 32u);
    element::put<element::Act>(rows, row * SEISMIC_RESULT_0_STRIDE_0 + ulong(column) * SEISMIC_RESULT_0_STRIDE_1,
        packets::value_at<packets::W0>(packet, column % 32u));
}
