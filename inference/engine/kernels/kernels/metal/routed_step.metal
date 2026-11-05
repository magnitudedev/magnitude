// Expert streaming, the step's sequence (`routed_step` in routed.seismic):
// one thread copies the pair the host wrote before submitting the step.
// Every later entry of the step reads it in order on the same queue.

kernel void routed_step(
    device int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    device const int *sequence [[buffer(SEISMIC_BUFFER_SEQUENCE)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0)
        for (ulong word = 0; word < SEISMIC_CONST_STEP_WORDS; ++word)
            step[word * SEISMIC_STEP_STRIDE_0] = sequence[word * SEISMIC_SEQUENCE_STRIDE_0];
}
