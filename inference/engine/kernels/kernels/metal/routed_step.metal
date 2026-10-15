// Expert streaming, the step sequence (`routed_step` in routed.seismic): one
// thread sets it to the step's own sequence, which the host wrote before
// submitting the step. Every later entry of the step reads it in order on
// the same queue.

kernel void routed_step(
    device int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    device const int *sequence [[buffer(SEISMIC_BUFFER_SEQUENCE)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0)
        step[0] = sequence[0];
}
