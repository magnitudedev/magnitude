// `repack_weight`: the registered GGUF -> resident conversion of a whole
// weight (lib/repack/repack.h), threadgroup z the [B, N, K] matrix.

#include "lib/repack/bind.h"

kernel void repack_weight(
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device uchar *destination [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 tile [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]) {
    repack::repack_tile<RepackBinding>(source, destination, seismic_words, tile.z, tile.xy, thread_position.x);
}
