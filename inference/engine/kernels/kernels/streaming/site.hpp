// Expert streaming's site request (streaming_words.seismic's site kind;
// `routed_site_request` in routed.seismic), over the backend's shim
// (streaming/contract.hpp): which choices a streamed decode layer asks the
// CPU for, and the request it publishes in its slot; the combine waits for
// the answer by the exchange's rule (`streaming_await_slot`).

// Whether choice c of the snapshot `sources` (strides `sources0`..`2`, K
// choices a row) is the CPU's: its SOURCE_SITE names a tier record.
STREAMING_FN bool streaming_site_choice(STREAMING_DEVICE_READ sources, u64 sources0, u64 sources1, u64 sources2,
    u32 c) {
    const u32 K = u32(SEISMIC_DIM_K);
    return streaming_device_load(STREAMING_AT(sources,
               SEISMIC_CONST_SOURCE_SITE * sources0 + u64(c / K) * sources1 + u64(c % K) * sources2)) >= 0;
}

// Whether any of the step's `choices` choices is the CPU's.
STREAMING_FN bool streaming_site_any(STREAMING_DEVICE_READ sources, u64 sources0, u64 sources1, u64 sources2,
    u32 choices) {
    bool any = false;
    for (u32 c = 0; c < choices && !any; ++c)
        any = streaming_site_choice(sources, sources0, sources1, sources2, c);
    return any;
}

// The request into its slot's words `slot` (stride `slot1`), by thread 0,
// after every thread's stores of the slot's rows (the caller's
// `streaming_publish_barrier`): each CPU choice's row, tier record, expert
// (of `routes`, strides `routes0`, `routes1`) and weight's bits (of
// `weights`, the f32 weights' words, strides `weights0`, `weights1`), the
// count, then the request, step `q` (`streaming_request_slot`). A step with
// no CPU choice requests nothing: its combine waits for nothing.
STREAMING_FN void streaming_site_publish(u64 q, STREAMING_DEVICE_READ routes, u64 routes0, u64 routes1,
    STREAMING_DEVICE_READ weights, u64 weights0, u64 weights1, STREAMING_DEVICE_READ sources, u64 sources0,
    u64 sources1, u64 sources2, u32 choices, STREAMING_SYSTEM_WORDS slot, u64 slot1) {
    const u32 K = u32(SEISMIC_DIM_K);
    int count = 0;
    for (u32 c = 0; c < choices; ++c) {
        if (!streaming_site_choice(sources, sources0, sources1, sources2, c))
            continue;
        const u64 at = u64(c / K) * sources1 + u64(c % K) * sources2;
        const u64 entry = SEISMIC_CONST_SITE_ENTRIES + SEISMIC_CONST_SITE_ENTRY_WORDS * u64(count);
        streaming_store(STREAMING_AT(slot, (entry + SEISMIC_CONST_SITE_ROW) * slot1), int(c / K));
        streaming_store(STREAMING_AT(slot, (entry + SEISMIC_CONST_SITE_TIER_RECORD) * slot1),
            streaming_device_load(STREAMING_AT(sources, SEISMIC_CONST_SOURCE_SITE * sources0 + at)));
        streaming_store(STREAMING_AT(slot, (entry + SEISMIC_CONST_SITE_EXPERT) * slot1),
            streaming_device_load(STREAMING_AT(routes, u64(c / K) * routes0 + u64(c % K) * routes1)));
        streaming_store(STREAMING_AT(slot, (entry + SEISMIC_CONST_SITE_WEIGHT_BITS) * slot1),
            streaming_device_load(STREAMING_AT(weights, u64(c / K) * weights0 + u64(c % K) * weights1)));
        ++count;
    }
    if (count == 0)
        return;
    streaming_store(STREAMING_AT(slot, SEISMIC_CONST_SITE_COUNT * slot1), count);
    streaming_request_slot(slot, slot1, q);
}
