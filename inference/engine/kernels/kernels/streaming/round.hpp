// Expert streaming's scratch kind (streaming_words.seismic): a layer's
// scratch table `[FLIGHT, LANES SCRATCH_SLOT]`, the entry's `scratch`, over
// the backend's shim (streaming/contract.hpp), for `routed_scratch_request`,
// a round's landing launch (`routed_round`), the round grids and the seed
// commit.

// The first word of lane `lane` of step `q`'s slot (`scratch_lane` in
// routed.seismic).
STREAMING_FN u64 streaming_scratch_lane(u64 q, u64 lane STREAMING_WORDS_PARAM) {
    return (q % u64(SEISMIC_DIM_FLIGHT)) * SEISMIC_SCRATCH_STRIDE_0
        + lane * u64(SEISMIC_CONST_SCRATCH_SLOT) * SEISMIC_SCRATCH_STRIDE_1;
}

// Step `q`'s lane SCRATCH_ROUND + `round` of its slot of the table
// `scratch` (`routed_round`'s slot).
STREAMING_FN STREAMING_SYSTEM_WORDS streaming_round_lane(u64 q, u64 round,
    STREAMING_SYSTEM_WORDS scratch STREAMING_WORDS_PARAM) {
    return STREAMING_AT(scratch,
        streaming_scratch_lane(q, u64(SEISMIC_CONST_SCRATCH_ROUND) + round STREAMING_WORDS_ARG));
}

// Step `q`'s arrival at the round whose lane is `lane` (its words `stride`
// apart), by one thread: the lane requested, after every earlier write of
// the device's (a system-scope fence), so the host serving the round sees
// the device past what came before it.
STREAMING_FN void streaming_round_arrive(STREAMING_SYSTEM_WORDS lane, u64 stride, u64 q) {
    streaming_system_fence();
    streaming_request_slot(lane, stride, q);
}
