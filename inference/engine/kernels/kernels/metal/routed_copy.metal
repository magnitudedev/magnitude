// Expert streaming, a prompt group's seeds (`routed_copy` in
// routed.seismic): the listed records of the source into the destination,
// byte for byte (streaming/list_copy.hpp), each a record port: the source
// may be a host tier the host lands into, read in its port's space
// (<seismic/slab.h>).

#include <seismic/slab.h>
#include "lib/streaming/control.h"
#include "lib/repack/repack.h"
#include "../streaming/list.hpp"

#define STREAMING_COPY_FROM(table, record) slab::region<const uchar, SLAB_SPACE(SEISMIC_SOURCE)>(table, ulong(record))
#define STREAMING_COPY_TO(table, record) slab::region<uchar, SLAB_SPACE(SEISMIC_DESTINATION)>(table, ulong(record))
#include "../streaming/list_copy.hpp"

kernel void routed_copy(
    device const ulong *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    SYSTEM const int *list [[buffer(SEISMIC_BUFFER_LIST)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    device const ulong *destination [[buffer(SEISMIC_BUFFER_DESTINATION)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 shape [[threads_per_threadgroup]]) {
    threadgroup int words[STREAMING_LIST_WORDS];
    streaming_list_copy(source, list, step[0], destination, REPACK_MATRIX_BYTES(SEISMIC_SOURCE),
        group.x + groups.x * group.y, groups.x * groups.y, words, tid, shape.x, seismic_words);
}
