// Expert streaming's scratch kind (streaming_words.seismic): a layer's
// scratch table `[FLIGHT, LANES SCRATCH_SLOT]`, the entry's `scratch`, over
// the backend's shim (streaming/contract.hpp), for `routed_scratch_request`,
// a round's wait (`routed_scratch_wait`), the round grids and the seed
// commit.

// The first word of lane `lane` of step `q`'s slot (`scratch_lane` in
// routed.seismic).
STREAMING_FN u64 streaming_scratch_lane(u64 q, u64 lane STREAMING_WORDS_PARAM) {
    return (q % u64(SEISMIC_DIM_FLIGHT)) * SEISMIC_SCRATCH_STRIDE_0
        + lane * u64(SEISMIC_CONST_SCRATCH_SLOT) * SEISMIC_SCRATCH_STRIDE_1;
}

// Step `q`'s round `round`, by one thread: lane SCRATCH_ROUND + round of
// the step's slot of the table `scratch` requested (the device's arrival
// at the round), then waited on (`streaming_await_slot`).
STREAMING_FN void streaming_round_wait(u64 q, u64 round, STREAMING_SYSTEM_WORDS scratch STREAMING_WORDS_PARAM) {
    STREAMING_SYSTEM_WORDS slot =
        STREAMING_AT(scratch, streaming_scratch_lane(q, u64(SEISMIC_CONST_SCRATCH_ROUND) + round STREAMING_WORDS_ARG));
    streaming_system_fence();
    streaming_request_slot(slot, SEISMIC_SCRATCH_STRIDE_1, q);
    streaming_await_slot(slot, SEISMIC_SCRATCH_STRIDE_1, q);
}
