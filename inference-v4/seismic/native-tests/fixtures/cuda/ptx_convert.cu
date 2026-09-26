// One thread per element pair; the 16-bit tensors are contiguous, so a pair
// is one 32-bit word with the lower index in the low half.

extern "C" __global__ void ptx_pack(SEISMIC_KERNEL_PARAMS) {
    const float *x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    unsigned *pairs = reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const unsigned long long pair = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * pair >= SEISMIC_DIM_N) {
        return;
    }
    const float lo = x[2 * pair * SEISMIC_X_STRIDE_0];
    const float hi = x[(2 * pair + 1) * SEISMIC_X_STRIDE_0];
#if defined(SEISMIC_ELEMENT_E_REPRESENTATION_BF16)
    pairs[pair] = seismic_pack_bf16x2(lo, hi);
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_F16)
    pairs[pair] = seismic_pack_f16x2(lo, hi);
#else
#error "ptx_pack binds E to f16 or bf16"
#endif
}

extern "C" __global__ void ptx_unpack(SEISMIC_KERNEL_PARAMS) {
    const unsigned *pairs = reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const unsigned long long pair = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * pair >= SEISMIC_DIM_N) {
        return;
    }
#if defined(SEISMIC_ELEMENT_E_REPRESENTATION_BF16)
    const float2 values = seismic_unpack_bf16x2(seismic_ld_nc_u32(pairs + pair));
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_F16)
    const float2 values = seismic_unpack_f16x2(seismic_ld_nc_u32(pairs + pair));
#else
#error "ptx_unpack binds E to f16 or bf16"
#endif
    result[2 * pair * SEISMIC_RESULT_0_STRIDE_0] = values.x;
    result[(2 * pair + 1) * SEISMIC_RESULT_0_STRIDE_0] = values.y;
}
