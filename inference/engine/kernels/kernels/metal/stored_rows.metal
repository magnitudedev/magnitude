// stored_rows: rows of a streamed table whose slots hold them as stored
// (gguf_iq4_nl_block), found through the directory and decoded into A. A
// threadgroup per (row, 256 columns), a thread per value: the slot's block
// of the value (an f16 scale, then sixteen bytes whose low nibbles are values
// 0..16 and high nibbles 16..32) indexes the iq4 table, times the scale.
#include "lib/core/activation.h"
#include "lib/core/gathered.h"

#if !defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_IQ4_NL_BLOCK)
#error "stored_rows decodes gguf_iq4_nl_block slots"
#endif

constant float stored_iq4[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
    1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f,
};

kernel void stored_rows(
    device const uchar *slots [[buffer(SEISMIC_BUFFER_SLOTS)]],
    device const int *tokens [[buffer(SEISMIC_BUFFER_TOKENS)]],
    device const int *directory [[buffer(SEISMIC_BUFFER_DIRECTORY)]],
    device uchar *rows [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const ulong row = group.x;
    const ulong column = ulong(group.y) * 256ul + thread_index;
    if (column >= SEISMIC_DIM_W)
        return;
    const ulong slot = gathered::row(directory, SEISMIC_DIRECTORY_STRIDE_0, SEISMIC_DIM_R, SEISMIC_DIM_F,
        tokens[row * SEISMIC_TOKENS_STRIDE_0]);
    // Strides of external storage count packets.
    device const uchar *block = slots + slot * SEISMIC_SLOTS_STRIDE_0 * SEISMIC_ELEMENT_E_PACKET_SIZE +
                                (column / 32ul) * 18ul;
    const float scale = float(as_type<half>(ushort(block[0] | (ushort(block[1]) << 8))));
    const uint within = uint(column % 32ul);
    const uint code = (block[2 + within % 16u] >> (4u * (within / 16u))) & 15u;
    element::put<element::Act>(rows, row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1,
        stored_iq4[code] * scale);
}
