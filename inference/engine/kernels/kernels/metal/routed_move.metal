// Expert streaming, a bank shrink's record moves (`routed_move` in
// routed.seismic): the listed records of a group's per-record tensor into its
// other records, byte for byte (streaming/list_copy.hpp). Threadgroup rows
// stride over the list's entries; the threads of a row over each record. No
// entry's destination is another's source, so the entries are independent.

#include <seismic/slab.h>
#include "lib/streaming/control.h"
#include "lib/repack/repack.h"
#include "../streaming/list.hpp"

#define STREAMING_COPY_FROM(table, record) slab::region<const uchar, SLAB_SPACE(SEISMIC_RECORDS)>(table, ulong(record))
#define STREAMING_COPY_TO(table, record) slab::region<uchar, SLAB_SPACE(SEISMIC_RECORDS)>(table, ulong(record))
#include "../streaming/list_copy.hpp"

kernel void routed_move(
    device const ulong *records [[buffer(SEISMIC_BUFFER_RECORDS)]],
    SYSTEM const int *list [[buffer(SEISMIC_BUFFER_LIST)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 shape [[threads_per_threadgroup]]) {
    threadgroup int words[STREAMING_LIST_WORDS];
    streaming_list_move(records, list, REPACK_MATRIX_BYTES(SEISMIC_RECORDS), group.y, groups.y,
        ulong(group.x) * shape.x + tid, ulong(groups.x) * shape.x, words, tid, shape.x, seismic_words);
}
