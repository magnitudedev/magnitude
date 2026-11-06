// Expert streaming, a scratch round's stored records (`routed_convert`
// in routed.seismic): when the list's sequence is this step's, entry i below
// its count converts conveyor record `from` (below V), or, for a negative
// `from`, the stored bytes the host read into scratch slot -1 - from (a slot
// no pass reads, read through the same coherent pointer as a conveyor
// record), into scratch slot `to` (below S) by windows
// (lib/streaming/window.h), `repack_weight`'s
// conversion. The conveyor and the destination are record ports (views of
// lent records, the destination the scratch lender's), each record read or
// written at its own base. Threadgroups stride over the listed records'
// items, so a launch costs what its list holds.

#include <seismic/slab.h>
#include "lib/streaming/control.h"
#include "lib/streaming/window.h"
#include "../streaming/list.hpp"

REPACK_BINDING(ConveyorBinding, SEISMIC_CONVERSION_ST_U, SEISMIC_CONVEYOR, SEISMIC_DESTINATION, SEISMIC_DIM_N, SEISMIC_DIM_K)
static_assert(SEISMIC_CONST_CONVERT_TILE_ROWS == repack::TILE_ROWS, "a convert threadgroup's tile is repack's");

kernel void routed_convert(
    device const ulong *conveyor [[buffer(SEISMIC_BUFFER_CONVEYOR)]],
    SYSTEM const int *list [[buffer(SEISMIC_BUFFER_LIST)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    device const ulong *destination [[buffer(SEISMIC_BUFFER_DESTINATION)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *window [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 shape [[threads_per_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup int words[STREAMING_LIST_WORDS];
    const ulong stride = SEISMIC_LIST_STRIDE_0;
    const uint entries = streaming_step_entries(list, stride, step[0], words, tid);
    const uint per_record = uint(streaming::record_items<ConveyorBinding>(seismic_words));
    STREAMING_LIST_ITEMS(at, entries, per_record, group.x + groups.x * group.y, groups.x * groups.y) {
        streaming_list_pair(list, stride, at, per_record, entries, words, tid, shape.x);
        const int from = streaming_list_from(words), to = streaming_list_to(words);
        const ulong item = at % per_record;
        STAGED *source = from >= 0 ? streaming::source_record(conveyor, ulong(from))
                                   : streaming::source_record(destination, ulong(-1 - from));
        streaming::convert_item<ConveyorBinding>(source,
                slab::region<uchar>(destination, ulong(to)),
                seismic_words, item, window, tid, 32 * simdgroups, simd, simdgroups, lane);
    }
}
