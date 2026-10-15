// Expert streaming's convert (`routed_convert`, `routed_convert_wide` in
// routed.seismic): the listed staging records into bank slots, by windows
// (lib/streaming/window.h): when list[0] is this step's sequence, record
// i < list[1] is staging record list[2 + 2 i] into slot list[3 + 2 i].
// Threadgroups of 8 simdgroups stride over the records' items, so a launch
// costs what its list holds.

// The including entry defines CONVERT_KERNEL, its kernel's name; the two
// entries differ only in their grids.

#include "bind_convert.h"
#include "window.h"

kernel void CONVERT_KERNEL(
    volatile coherent(system) device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device uchar *destination [[buffer(SEISMIC_BUFFER_DESTINATION)]],
    volatile coherent(system) device const int *list [[buffer(SEISMIC_BUFFER_LIST)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uchar window[streaming::Window<ConvertBinding>::bytes()];
    threadgroup int header[2];
    threadgroup int entry[2];
    // The list's header, one system-coherent load each.
    if (tid == 0) {
        header[0] = list[0];
        header[1] = list[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (header[0] != step[0])
        return;
    const ulong count = ulong(max(header[1], 0));
    const ulong per_record = streaming::record_items<ConvertBinding>(seismic_words);
    const ulong total = ulong(groups.x) * groups.y * groups.z;
    const ulong first = group.x + ulong(groups.x) * (group.y + ulong(groups.y) * group.z);
    for (ulong item = first; item < count * per_record; item += total) {
        const ulong record = item / per_record;
        if (tid == 0) {
            entry[0] = list[2 + 2 * record];
            entry[1] = list[3 + 2 * record];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        streaming::convert_item<ConvertBinding>(source, destination, seismic_words, ulong(entry[0]), ulong(entry[1]), item % per_record,
            window, tid, 256, simd, 8, lane);
    }
}
