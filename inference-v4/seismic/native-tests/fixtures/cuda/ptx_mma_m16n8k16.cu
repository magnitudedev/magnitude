// One warp computes the 16x8 tile, loading each operand fragment element by
// element in the layout the prelude documents for the MMA helpers.

extern "C" __global__ void ptx_mma_m16n8k16(SEISMIC_KERNEL_PARAMS) {
    const unsigned short *a = reinterpret_cast<const unsigned short *>(SEISMIC_PTR(SEISMIC_BUFFER_A));
    const unsigned short *b = reinterpret_cast<const unsigned short *>(SEISMIC_PTR(SEISMIC_BUFFER_B));
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const unsigned g = threadIdx.x / 4;
    const unsigned t = threadIdx.x % 4;
    // The pair (k, k + 1) of `row`, lower k in the low half.
    auto a_pair = [&](unsigned long long row, unsigned long long k) {
        return (unsigned)a[row * SEISMIC_A_STRIDE_0 + k * SEISMIC_A_STRIDE_1] |
               ((unsigned)a[row * SEISMIC_A_STRIDE_0 + (k + 1) * SEISMIC_A_STRIDE_1] << 16);
    };
    auto b_pair = [&](unsigned long long column, unsigned long long k) {
        return (unsigned)b[column * SEISMIC_B_STRIDE_0 + k * SEISMIC_B_STRIDE_1] |
               ((unsigned)b[column * SEISMIC_B_STRIDE_0 + (k + 1) * SEISMIC_B_STRIDE_1] << 16);
    };
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned long long k0 = 0; k0 < SEISMIC_DIM_K; k0 += 16) {
        const unsigned fa[4] = {a_pair(g, k0 + 2 * t), a_pair(g + 8, k0 + 2 * t),
                                a_pair(g, k0 + 2 * t + 8), a_pair(g + 8, k0 + 2 * t + 8)};
        const unsigned fb[2] = {b_pair(g, k0 + 2 * t), b_pair(g, k0 + 2 * t + 8)};
#if defined(SEISMIC_ELEMENT_E_REPRESENTATION_BF16)
        seismic_mma_m16n8k16_bf16(acc, fa, fb);
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_F16)
        seismic_mma_m16n8k16_f16(acc, fa, fb);
#else
#error "ptx_mma_m16n8k16 binds E to f16 or bf16"
#endif
    }
    const unsigned long long r0 = SEISMIC_RESULT_0_STRIDE_0, r1 = SEISMIC_RESULT_0_STRIDE_1;
    result[g * r0 + (2 * t) * r1] = acc[0];
    result[g * r0 + (2 * t + 1) * r1] = acc[1];
    result[(g + 8) * r0 + (2 * t) * r1] = acc[2];
    result[(g + 8) * r0 + (2 * t + 1) * r1] = acc[3];
}
