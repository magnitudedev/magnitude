// kimi_delta_chunk (prefill row classes): the Kimi delta rule in the chunked
// WY form over pieces of 16 rows (`chunk::prepare` and `chunk::scan` over
// `kimi::ChannelGates`: one log decay per key channel, inside the products),
// then the gate launch.

#include "lib/recurrent/kimi.cuh"
#include "lib/recurrent/chunk.cuh"

#ifdef SEISMIC_FORMING_KIMI_DELTA_CHUNK_PREPARE
__global__ void kimi_delta_chunk_prepare(SEISMIC_KERNEL_PARAMS) {
    const kimi::Inputs in = KIMI_INPUTS();
    chunk::prepare<kimi::ChannelGates>(in, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PIECES));
}
#endif

#ifdef SEISMIC_FORMING_KIMI_DELTA_CHUNK_SCAN
template <unsigned ROWS>
__global__ void kimi_delta_chunk_scan(SEISMIC_KERNEL_PARAMS) {
    const kimi::Inputs in = KIMI_INPUTS();
    chunk::scan<kimi::ChannelGates, ROWS>(in, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PIECES),
                                          SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_MIXED));
}
#endif

#ifdef SEISMIC_FORMING_KIMI_DELTA_CHUNK_GATE
extern "C" __global__ void kimi_delta_chunk_gate(SEISMIC_KERNEL_PARAMS) {
    const kimi::Inputs in = KIMI_INPUTS();
    recurrent::gate_row<ELEMENT_OF(SEISMIC_RECURRENT_NORM)>(
        in.base, kimi::ChannelGates{in}.gate_input(), SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_MIXED),
        SEISMIC_PTR(SEISMIC_BUFFER_RECURRENT_NORM), element::word_f32(SEISMIC_PARAM_EPSILON),
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), blockIdx.x);
}
#endif
