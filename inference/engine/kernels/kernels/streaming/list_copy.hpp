// Expert streaming's record copies of a host-written list (`routed_copy` in
// routed.seismic), over the backend's shim
// (streaming/contract.hpp) and the includer's record-address hooks
// (`STREAMING_COPY_FROM`, `STREAMING_COPY_TO`), byte for byte
// (`streaming_copy_bytes`): each listed (from, to) pair's record of `bytes`
// bytes of the source into the destination. The includer includes
// streaming/list.hpp first.

// A copy of the step's list (`routed_copy`): each group copies its own
// LIST_PART-byte parts of the listed records in turn, from item `first`, a
// group's items `groups` apart (an entry's parts its items), its threads
// striding over each part. A list written for another step copies nothing.
// Every thread of the group calls it.
STREAMING_FN void streaming_list_copy(STREAMING_RECORDS source, STREAMING_SYSTEM_READ list, int step,
    STREAMING_RECORDS destination, u64 bytes, u32 first, u32 groups, STREAMING_GROUP_PARAM, u32 tid,
    u32 threads STREAMING_WORDS_PARAM) {
    const u64 stride = SEISMIC_LIST_STRIDE_0;
    const u32 entries = streaming_step_entries(list, stride, step, streaming_group, tid);
    const u64 part = SEISMIC_CONST_LIST_PART;
    const u32 parts = streaming_narrow_quotient(bytes + part - 1, part);
    STREAMING_LIST_ITEMS(item, entries, parts, first, groups) {
        streaming_list_pair(list, stride, item, parts, entries, streaming_group, tid, threads);
        const int from = streaming_list_from(streaming_group), to = streaming_list_to(streaming_group);
        const u64 at = u64(item % parts) * part, length = bytes - at < part ? bytes - at : part;
        streaming_copy_bytes(STREAMING_COPY_FROM(source, from) + at, STREAMING_COPY_TO(destination, to) + at, length,
            u64(tid), u64(threads));
    }
}
