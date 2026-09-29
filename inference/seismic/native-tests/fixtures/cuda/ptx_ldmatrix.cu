// Stages four 8x8 matrices in shared memory, reads them with COUNT matrices
// per `ldmatrix`, and writes lane l's register for matrix m to row l/4,
// columns 2*(l%4) and 2*(l%4)+1 of result matrix m. For the plain form that
// reproduces the input; for `.trans` it produces each matrix's transpose.

template <bool TRANS>
__device__ __forceinline__ void ptx_ldmatrix_body(const unsigned short *x, unsigned short *result,
                                                  unsigned long long x0, unsigned long long x1,
                                                  unsigned long long x2, unsigned long long r0,
                                                  unsigned long long r1, unsigned long long r2) {
    __shared__ __align__(16) unsigned short tile[4 * 8 * 8];
    const unsigned lane = threadIdx.x;
    // Lane l stages row l%8 of matrix l/8.
    for (unsigned c = 0; c < 8; ++c) {
        tile[lane * 8 + c] = x[(lane / 8) * x0 + (lane % 8) * x1 + c * x2];
    }
    __syncthreads();
    for (unsigned base = 0; base < 4; base += SEISMIC_TUNE_COUNT) {
        // Lane 8*i + r addresses row r of matrix base + i.
        const unsigned short *row = tile + ((base + (lane / 8) % SEISMIC_TUNE_COUNT) * 8 + lane % 8) * 8;
        unsigned fragment[SEISMIC_TUNE_COUNT];
#if SEISMIC_TUNE_COUNT == 1
        if (TRANS) { seismic_ldmatrix_x1_trans(fragment, row); } else { seismic_ldmatrix_x1(fragment, row); }
#elif SEISMIC_TUNE_COUNT == 2
        if (TRANS) { seismic_ldmatrix_x2_trans(fragment, row); } else { seismic_ldmatrix_x2(fragment, row); }
#elif SEISMIC_TUNE_COUNT == 4
        if (TRANS) { seismic_ldmatrix_x4_trans(fragment, row); } else { seismic_ldmatrix_x4(fragment, row); }
#else
#error "COUNT is 1, 2 or 4"
#endif
        for (unsigned i = 0; i < SEISMIC_TUNE_COUNT; ++i) {
            const unsigned long long place = (base + i) * r0 + (lane / 4) * r1 + (2 * (lane % 4)) * r2;
            result[place] = (unsigned short)(fragment[i] & 0xffffu);
            result[place + r2] = (unsigned short)(fragment[i] >> 16);
        }
    }
}

extern "C" __global__ void ptx_ldmatrix(SEISMIC_KERNEL_PARAMS) {
    ptx_ldmatrix_body<false>(
        reinterpret_cast<const unsigned short *>(SEISMIC_PTR(SEISMIC_BUFFER_X)),
        reinterpret_cast<unsigned short *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)), SEISMIC_X_STRIDE_0,
        SEISMIC_X_STRIDE_1, SEISMIC_X_STRIDE_2, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1,
        SEISMIC_RESULT_0_STRIDE_2);
}

extern "C" __global__ void ptx_ldmatrix_trans(SEISMIC_KERNEL_PARAMS) {
    ptx_ldmatrix_body<true>(
        reinterpret_cast<const unsigned short *>(SEISMIC_PTR(SEISMIC_BUFFER_X)),
        reinterpret_cast<unsigned short *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)), SEISMIC_X_STRIDE_0,
        SEISMIC_X_STRIDE_1, SEISMIC_X_STRIDE_2, SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1,
        SEISMIC_RESULT_0_STRIDE_2);
}
