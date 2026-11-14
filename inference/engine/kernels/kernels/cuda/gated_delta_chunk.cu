// gated_delta_chunk (prefill row classes): the gated delta rule in the
// chunked WY form over pieces of 16 rows (`chunk::prepare` and `chunk::scan`
// over `recurrent::HeadGates`: one decay per head, factored out of the
// products), then the gate launch.

#include "lib/recurrent/chunk.cuh"

#ifdef SEISMIC_FORMING_GATED_DELTA_CHUNK_PREPARE
__global__ void gated_delta_chunk_prepare(SEISMIC_KERNEL_PARAMS) {
    const recurrent::Inputs in = RECURRENT_INPUTS();
    chunk::prepare<recurrent::HeadGates>(in, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PIECES));
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_CHUNK_SCAN
template <unsigned ROWS>
__global__ void gated_delta_chunk_scan(SEISMIC_KERNEL_PARAMS) {
    const recurrent::Inputs in = RECURRENT_INPUTS();
    chunk::scan<recurrent::HeadGates, ROWS>(in, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PIECES),
                                            SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_MIXED));
}
#endif

#ifdef SEISMIC_FORMING_GATED_DELTA_CHUNK_GATE
extern "C" __global__ void gated_delta_chunk_gate(SEISMIC_KERNEL_PARAMS) {
    const recurrent::Inputs in = RECURRENT_INPUTS();
    recurrent::gate_row<ELEMENT_OF(SEISMIC_RECURRENT_NORM)>(
        in, recurrent::HeadGates{in}.gate_input(), SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_MIXED),
        SEISMIC_PTR(SEISMIC_BUFFER_RECURRENT_NORM), element::word_f32(SEISMIC_PARAM_EPSILON),
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), blockIdx.x);
}
#endif
