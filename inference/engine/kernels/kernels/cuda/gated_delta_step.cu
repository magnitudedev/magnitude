// gated_delta_step (decode, MTP verify and small row classes): the
// row-sequential gated delta rule with its preparation fused
// (`recurrent::step` over `recurrent::HeadGates`).

#include "lib/recurrent/recurrent.cuh"

#ifdef SEISMIC_FORMING_GATED_DELTA_STEP
template <unsigned ROWS, unsigned WARPS>
__global__ void gated_delta_step(SEISMIC_KERNEL_PARAMS) {
    const recurrent::Inputs in = RECURRENT_INPUTS();
    recurrent::step<ROWS, WARPS, ELEMENT_OF(SEISMIC_RECURRENT_NORM), recurrent::HeadGates>(
        in, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_MIXED),
        reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS)),
        SEISMIC_PTR(SEISMIC_BUFFER_RECURRENT_NORM), element::word_f32(SEISMIC_PARAM_EPSILON),
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
}
#endif
