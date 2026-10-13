// stored_rows: rows of a streamed table whose slots hold them as stored
// (gguf_iq4_nl_block), found through the directory and decoded into A. A block
// per (row, 256 columns), a thread per value: the slot's block of the value
// (an f16 scale, then sixteen bytes whose low nibbles are values 0..16 and
// high nibbles 16..32) indexes the iq4 table, times the scale. Rows past the
// 65535-block grid are visited by a grid-stride loop.
#include "lib/core/activation.cuh"
#include "lib/core/gathered.cuh"

#if !defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_IQ4_NL_BLOCK)
#error "stored_rows decodes gguf_iq4_nl_block slots"
#endif

using element::Act;
using element::u16;
using element::u32;
using element::u64;
using element::u8;

__constant__ float stored_iq4[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
    1.0f,    13.0f,   25.0f,  38.0f,  53.0f,  69.0f,  89.0f,  113.0f,
};

extern "C" __global__ void stored_rows(SEISMIC_KERNEL_PARAMS) {
    const u8 *slots = SEISMIC_PTR(SEISMIC_BUFFER_SLOTS);
    const int *tokens = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKENS));
    const int *directory = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_DIRECTORY));
    u8 *rows = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    const u64 column = (u64)blockIdx.y * 256ull + threadIdx.x;
    if (column >= SEISMIC_DIM_W)
        return;
    for (u64 row = blockIdx.x; row < SEISMIC_DIM_M; row += gridDim.x) {
        const u64 slot = gathered::row(directory, SEISMIC_DIRECTORY_STRIDE_0, SEISMIC_DIM_R, SEISMIC_DIM_F,
                                       tokens[row * SEISMIC_TOKENS_STRIDE_0]);
        // Strides of external storage count packets.
        const u8 *block =
            slots + slot * SEISMIC_SLOTS_STRIDE_0 * SEISMIC_ELEMENT_E_PACKET_SIZE + (column / 32ull) * 18ull;
        const float scale = element::F16::load((u16)(block[0] | ((u16)block[1] << 8)));
        const u32 within = (u32)(column % 32ull);
        const u32 code = (block[2 + within % 16u] >> (4u * (within / 16u))) & 15u;
        element::put<Act>(rows, row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1,
                          stored_iq4[code] * scale);
    }
}
