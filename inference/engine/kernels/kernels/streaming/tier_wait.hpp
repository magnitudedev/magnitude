// Expert streaming: a streamed layer's tier wait (`routed_tier_wait` in
// routed.seismic), over the backend's shim (streaming/contract.hpp): the
// body of the route's wait grid (WAIT_GRID), one group. The grid is empty
// unless the route requested the step's slot of the layer's file table, so
// the slot is this step's request whenever it runs.
//
// Each thread waits on the slot (`streaming_await_slot`, which the host
// answers once it landed every absent choice's expert and wrote where each
// lies), then writes the sources of its choices among the absent ones (a
// choice's request word an expert, not -1 - a held one) into the step's
// snapshot (`sources`): a bank record (>= 0, unified memory) is read as a
// hit; a tier record t (-2 - t, across a link) goes where the route
// assigned its expert: to the CPU where its SOURCE_SITE is pending (-2:
// SOURCE_SITE t), else to the GPU in place (SOURCE_RECORD -2 - t). A failed
// read (record -1) or a failed step sources none: SOURCE_RECORD and
// SOURCE_SITE -1. Every thread reads the same words, so no thread waits for
// another.

// Thread `tid` of `threads`' wait and its choices' sources: `file` the
// layer's file table (strides `file0`, `file1`), `sources0`..`sources2` the
// route's sources' strides; `step` the step pair.
STREAMING_FN void streaming_tier_wait(STREAMING_DEVICE_READ step, u64 step0, STREAMING_SYSTEM_WORDS file, u64 file0,
    u64 file1, STREAMING_DEVICE_WORDS sources, u64 sources0, u64 sources1, u64 sources2, u32 tid,
    u32 threads STREAMING_WORDS_PARAM) {
    const u64 q = streaming_step(step, step0);
    STREAMING_SYSTEM_WORDS slot = STREAMING_AT(file, (q % u64(SEISMIC_DIM_FLIGHT)) * file0);
    const bool answered = streaming_await_slot(slot, file1, q);
    const u32 K = u32(SEISMIC_DIM_K);
    const u32 count = u32(streaming_load(STREAMING_AT(slot, SEISMIC_CONST_FILE_COUNT * file1)));
    for (u32 choice = tid; choice < count; choice += threads) {
        const u64 entry = SEISMIC_CONST_FILE_ENTRIES + SEISMIC_CONST_FILE_ENTRY_WORDS * u64(choice);
        const int expert = streaming_load(STREAMING_AT(slot, (entry + SEISMIC_CONST_FILE_CHOICE) * file1));
        if (expert < 0)
            continue;
        const u64 at = u64(choice / K) * sources1 + u64(choice % K) * sources2;
        STREAMING_DEVICE_WORDS record = STREAMING_AT(sources, SEISMIC_CONST_SOURCE_RECORD * sources0 + at);
        STREAMING_DEVICE_WORDS site = STREAMING_AT(sources, SEISMIC_CONST_SOURCE_SITE * sources0 + at);
        const int source =
            answered ? streaming_load(STREAMING_AT(slot, (entry + SEISMIC_CONST_FILE_RECORD) * file1)) : -1;
        if (source == -1) {
            streaming_device_store(record, -1);
            streaming_device_store(site, -1);
        } else if (source >= 0)
            streaming_device_store(record, source);
        else if (streaming_device_load(site) == -2)
            streaming_device_store(site, -2 - source);
        else
            streaming_device_store(record, source);
    }
}
