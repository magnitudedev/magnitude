// Expert streaming, a scratch-served layer's request
// (`routed_scratch_request` in routed.seismic), over the backend's shim
// (streaming/contract.hpp): each expert's flag, the number of choices routed
// to it (0 when none), then the request in the SCRATCH_ASK lane of the
// step's slot of the layer's scratch table. One group counts in its group
// words, a word an expert (SEISMIC_DIM_E of them). It does not wait. The
// includer includes streaming/round.hpp first.

// Every thread of the one group calls it: `routes` the step's [M, K]
// choices, `step` the step pair, `flags` the chain place's flags, `scratch`
// the layer's scratch table.
STREAMING_FN void streaming_scratch_request(STREAMING_DEVICE_READ routes, STREAMING_DEVICE_READ step,
    STREAMING_SYSTEM_WORDS flags, STREAMING_SYSTEM_WORDS scratch, STREAMING_GROUP_PARAM, u32 tid,
    u32 threads STREAMING_WORDS_PARAM) {
    const u32 experts = u32(SEISMIC_DIM_E), K = u32(SEISMIC_DIM_K), choices = u32(SEISMIC_DIM_M) * K;
    for (u32 e = tid; e < experts; e += threads)
        STREAMING_GROUP(e) = 0;
    streaming_barrier();
    for (u32 c = tid; c < choices; c += threads)
        STREAMING_GROUP_ADD(streaming_device_load(STREAMING_AT(routes,
                                u64(c / K) * SEISMIC_ROUTES_STRIDE_0 + u64(c % K) * SEISMIC_ROUTES_STRIDE_1)),
            1);
    streaming_barrier();
    for (u32 e = tid; e < experts; e += threads)
        streaming_store(STREAMING_AT(flags, u64(e) * SEISMIC_FLAGS_STRIDE_0), STREAMING_GROUP(e));
    // Every thread's flags, published before the request.
    streaming_publish_barrier();
    if (tid != 0u)
        return;
    const u64 q = streaming_step(step, SEISMIC_STEP_STRIDE_0);
    streaming_request_slot(
        STREAMING_AT(scratch, streaming_scratch_lane(q, u64(SEISMIC_CONST_SCRATCH_ASK) STREAMING_WORDS_ARG)),
        SEISMIC_SCRATCH_STRIDE_1, q);
}
