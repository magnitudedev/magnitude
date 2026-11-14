// kimi_delta_step (decode, MTP verify and small row classes): the
// row-sequential Kimi delta rule with its preparation fused
// (`recurrent::step` over `kimi::ChannelGates`).

#include "lib/recurrent/kimi.cuh"

#ifdef SEISMIC_FORMING_KIMI_DELTA_STEP
template <unsigned ROWS, unsigned WARPS>
__global__ void kimi_delta_step(SEISMIC_KERNEL_PARAMS) {
    const kimi::Inputs in = KIMI_INPUTS();
    recurrent::step<ROWS, WARPS, ELEMENT_OF(SEISMIC_RECURRENT_NORM), kimi::ChannelGates>(
        in, SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_MIXED),
        reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS)),
        SEISMIC_PTR(SEISMIC_BUFFER_RECURRENT_NORM), element::word_f32(SEISMIC_PARAM_EPSILON),
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
}
#endif
