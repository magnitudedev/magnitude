// `repack_weight`: the exact GGUF -> resident conversion (the Seismic
// registry's registered `repack`, lib/repack/repack.h) of a whole weight: E
// is the GGUF source, U the resident (representation, layout). The entry
// sees every weight as a [B, N, K] view, threadgroup z its matrix; each
// threadgroup (32 threads) places one row tile. The result is a record port
// (bank records import through it): matrix z is written from its own
// record's base.

#if !defined(SEISMIC_ELEMENT_E_KIND_EXTERNAL) || !defined(SEISMIC_ELEMENT_U_KIND_PACKED)
#error "repack_weight converts an external GGUF source into packed resident storage"
#endif

#if !defined(SEISMIC_CONVERSION_E_U)
#error "repack_weight binding is not a registered GGUF-to-resident conversion"
#endif

#include <seismic/slab.h>
#include "lib/repack/repack.h"

REPACK_BINDING(RepackBinding, SEISMIC_CONVERSION_E_U, SEISMIC_SOURCE, SEISMIC_RESULT_0, SEISMIC_DIM_N, SEISMIC_DIM_K)

kernel void repack_weight(
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device const ulong *destination [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 tile [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
    threadgroup uint staged[repack::exchange_words<RepackBinding>()];
    repack::repack_tile<RepackBinding>(repack::Strided<RepackBinding>::packet(source, seismic_words, tile.z, 0, 0),
        slab::region<uchar>(destination, tile.z), seismic_words, 0, tile.xy, thread_position.x, staged);
}
