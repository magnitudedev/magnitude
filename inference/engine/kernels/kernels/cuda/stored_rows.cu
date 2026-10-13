// Read directory-selected slots in their stored IQ4_NL or Q8_0 encoding.
// Both have an F16 scale per 32 values; codes are nonlinear nibbles or
// signed bytes. Decode directly and round once to the activation element.
#include "lib/core/activation.cuh"
#include "lib/core/gathered.cuh"

#if defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q8_0)
#define STORED_BLOCK_BYTES 34
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_IQ4_NL_BLOCK)
#define STORED_BLOCK_BYTES 18
#else
#error "stored_rows requires IQ4_NL or Q8_0 blocks"
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
            slots + slot * SEISMIC_SLOTS_STRIDE_0 * SEISMIC_ELEMENT_E_PACKET_SIZE + (column / 32ull) * STORED_BLOCK_BYTES;
        const float scale = element::F16::load((u16)(block[0] | ((u16)block[1] << 8)));
        const u32 within = (u32)(column % 32ull);
#if defined(SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q8_0)
        const u32 raw = block[2 + within];
        const float value = float(int(raw) - (raw >= 128u ? 256 : 0));
#else
        const u32 code = (block[2 + within % 16u] >> (4u * (within / 16u))) & 15u;
        const float value = stored_iq4[code];
#endif
        element::put<Act>(rows, row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1,
                          value * scale);
    }
}
