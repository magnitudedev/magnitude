// One warp computes the 16x8 tile, loading each operand fragment element by
// element in the layout the prelude documents for the MMA helpers.

extern "C" __global__ void ptx_mma_m16n8k32_s8(SEISMIC_KERNEL_PARAMS) {
    const int *a = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_A));
    const int *b = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_B));
    int *result = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const unsigned g = threadIdx.x / 4;
    const unsigned t = threadIdx.x % 4;
    // The four bytes k..k+3 of `row`, lowest k in the lowest byte.
    auto a_quad = [&](unsigned long long row, unsigned long long k) {
        unsigned quad = 0;
        for (unsigned j = 0; j < 4; ++j) {
            quad |= ((unsigned)a[row * SEISMIC_A_STRIDE_0 + (k + j) * SEISMIC_A_STRIDE_1] & 0xffu)
                    << (8 * j);
        }
        return quad;
    };
    auto b_quad = [&](unsigned long long column, unsigned long long k) {
        unsigned quad = 0;
        for (unsigned j = 0; j < 4; ++j) {
            quad |= ((unsigned)b[column * SEISMIC_B_STRIDE_0 + (k + j) * SEISMIC_B_STRIDE_1] & 0xffu)
                    << (8 * j);
        }
        return quad;
    };
    int acc[4] = {0, 0, 0, 0};
    for (unsigned long long k0 = 0; k0 < SEISMIC_DIM_K; k0 += 32) {
        const unsigned fa[4] = {a_quad(g, k0 + 4 * t), a_quad(g + 8, k0 + 4 * t),
                                a_quad(g, k0 + 4 * t + 16), a_quad(g + 8, k0 + 4 * t + 16)};
        const unsigned fb[2] = {b_quad(g, k0 + 4 * t), b_quad(g, k0 + 4 * t + 16)};
        seismic_mma_m16n8k32_s8(acc, fa, fb);
    }
    const unsigned long long r0 = SEISMIC_RESULT_0_STRIDE_0, r1 = SEISMIC_RESULT_0_STRIDE_1;
    result[g * r0 + (2 * t) * r1] = acc[0];
    result[g * r0 + (2 * t + 1) * r1] = acc[1];
    result[(g + 8) * r0 + (2 * t) * r1] = acc[2];
    result[(g + 8) * r0 + (2 * t + 1) * r1] = acc[3];
}
