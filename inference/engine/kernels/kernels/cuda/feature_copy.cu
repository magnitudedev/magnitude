// feature_copy: out[row, column] = input[row, column] (raw storage);
// one thread per element.

extern "C" __global__ void feature_copy(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const unsigned char *input = reinterpret_cast<const unsigned char *>(SEISMIC_PTR(SEISMIC_BUFFER_INPUT));
    unsigned char *out = reinterpret_cast<unsigned char *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT));
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (u64)SEISMIC_DIM_M * SEISMIC_DIM_D)
        return;
    const u64 row = index / SEISMIC_DIM_D;
    const u64 column = index % SEISMIC_DIM_D;
    for (u64 byte = 0; byte < SEISMIC_INPUT_PACKET_SIZE; ++byte)
        out[(row * SEISMIC_OUT_STRIDE_0 + column * SEISMIC_OUT_STRIDE_1) * SEISMIC_OUT_PACKET_SIZE + byte] =
            input[(row * SEISMIC_INPUT_STRIDE_0 + column * SEISMIC_INPUT_STRIDE_1) * SEISMIC_INPUT_PACKET_SIZE + byte];
}
