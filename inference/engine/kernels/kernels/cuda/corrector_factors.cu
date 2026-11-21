// corrector_factors: 4 elements per thread of a node's row x and its block's
// anchor a on a (column, node, block) grid, writing the joined row
// [x ; a ; x · a], each rounded once to A.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"

extern "C" __global__ void corrector_factors(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const element::u8 *nodes = SEISMIC_PTR(SEISMIC_BUFFER_NODES);
    const element::u8 *anchor = SEISMIC_PTR(SEISMIC_BUFFER_ANCHOR);
    const auto result = SEISMIC_RESULT_0_OUT;
    const u64 blocks = SEISMIC_DIM_S, count = SEISMIC_DIM_N, width = SEISMIC_DIM_H;
    const u64 col = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 4ull;
    if (col >= width)
        return;
    const u64 out_stride2 = SEISMIC_RESULT_0_STRIDE_2;
    const bool unit_strides = SEISMIC_NODES_STRIDE_2 == 1ull && SEISMIC_ANCHOR_STRIDE_1 == 1ull &&
                              SEISMIC_RESULT_0_STRIDE_3 == 1ull;
    constexpr u64 align_mask = 4ull * activation::Act::bytes - 1ull;
    const u64 end = (col + 4ull < width) ? (col + 4ull) : width;

    for (u64 block = blockIdx.z; block < blocks; block += gridDim.z) {
        const u64 anchor_base = block * SEISMIC_ANCHOR_STRIDE_0;
        for (u64 node = blockIdx.y; node < count; node += gridDim.y) {
            const u64 nodes_base = block * SEISMIC_NODES_STRIDE_0 + node * SEISMIC_NODES_STRIDE_1;
            const u64 out_base = block * SEISMIC_RESULT_0_STRIDE_0 + node * SEISMIC_RESULT_0_STRIDE_1;
            const bool aligned =
                ((reinterpret_cast<u64>(nodes + (nodes_base + col) * activation::Act::bytes) |
                  reinterpret_cast<u64>(anchor + (anchor_base + col) * activation::Act::bytes) |
                  reinterpret_cast<u64>(result.stored(out_base + col)) |
                  reinterpret_cast<u64>(result.stored(out_base + out_stride2 + col)) |
                  reinterpret_cast<u64>(result.stored(out_base + 2ull * out_stride2 + col))) &
                 align_mask) == 0ull;
            if (col + 4ull <= width && unit_strides && aligned) {
                float x[4], a[4], xa[4];
                element::span<activation::Act, 4, false>(nodes, nodes_base + col, x);
                element::span<activation::Act, 4, false>(anchor, anchor_base + col, a);
                xa[0] = x[0] * a[0];
                xa[1] = x[1] * a[1];
                xa[2] = x[2] * a[2];
                xa[3] = x[3] * a[3];
                result.put_span<4>(out_base + col, x);
                result.put_span<4>(out_base + out_stride2 + col, a);
                result.put_span<4>(out_base + 2ull * out_stride2 + col, xa);
            } else {
                for (u64 c = col; c < end; ++c) {
                    const float x = element::at<activation::Act>(nodes, nodes_base + c * SEISMIC_NODES_STRIDE_2);
                    const float a = element::at<activation::Act>(anchor, anchor_base + c * SEISMIC_ANCHOR_STRIDE_1);
                    const u64 at = out_base + c * SEISMIC_RESULT_0_STRIDE_3;
                    result.put(at, x);
                    result.put(at + out_stride2, a);
                    result.put(at + 2ull * out_stride2, x * a);
                }
            }
        }
    }
}
