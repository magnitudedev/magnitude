// message_gate_mix: `(1 - tanh(forget)) * dest + tanh(gain) * msg`, F32.
#include "lib/core/activation.cuh"

using element::u64;

extern "C" __global__ void message_gate_mix(SEISMIC_KERNEL_PARAMS) {
    const u64 index = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const u64 total = SEISMIC_DIM_M * SEISMIC_DIM_H;
    if (index >= total)
        return;
    const u64 row = index / SEISMIC_DIM_H;
    const u64 col = index % SEISMIC_DIM_H;
    const float *dest = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_DEST));
    const float *msg = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_MSG));
    const float *gain = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_GAIN));
    const float *forget = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_FORGET));
    const float d = dest[row * SEISMIC_DEST_STRIDE_0 + col * SEISMIC_DEST_STRIDE_1];
    const float m = msg[row * SEISMIC_MSG_STRIDE_0 + col * SEISMIC_MSG_STRIDE_1];
    const float g = gain[row * SEISMIC_GAIN_STRIDE_0 + col * SEISMIC_GAIN_STRIDE_1];
    const float f = forget[row * SEISMIC_FORGET_STRIDE_0 + col * SEISMIC_FORGET_STRIDE_1];
    const float tf = 2.0f / (1.0f + expf(-2.0f * f)) - 1.0f;
    const float tg = 2.0f / (1.0f + expf(-2.0f * g)) - 1.0f;
    SEISMIC_RESULT_0_OUT.put(row * SEISMIC_RESULT_0_STRIDE_0 + col * SEISMIC_RESULT_0_STRIDE_1,
        (1.0f - tf) * d + tg * m);
}
