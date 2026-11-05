// Expert streaming, the step's sequence (`routed_step` in routed.seismic);
// the CUDA form of `metal/routed_step.metal`: one thread copies the step
// pair (STEP_LOW, STEP_HIGH) the host wrote before submitting the step.
// Every later entry of the step reads it in order on the same stream.

extern "C" __global__ void routed_step(SEISMIC_KERNEL_PARAMS) {
    if (threadIdx.x != 0)
        return;
    int *step = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP));
    const int *sequence = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SEQUENCE));
    step[SEISMIC_CONST_STEP_LOW * SEISMIC_STEP_STRIDE_0] = sequence[SEISMIC_CONST_STEP_LOW * SEISMIC_SEQUENCE_STRIDE_0];
    step[SEISMIC_CONST_STEP_HIGH * SEISMIC_STEP_STRIDE_0] =
        sequence[SEISMIC_CONST_STEP_HIGH * SEISMIC_SEQUENCE_STRIDE_0];
}
