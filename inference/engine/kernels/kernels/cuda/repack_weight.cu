// `repack_weight`: the registered GGUF -> resident conversion of a whole
// weight (lib/repack/repack.cuh): E is the GGUF source, U the resident
// (representation, layout); the entry sees every weight as a [B, N, K] view.
// Each block, one warp, places one row tile: block x its column tile, y its
// row tile, z its matrix.

#include "lib/repack/repack.cuh"

REPACK_BINDING(RepackBinding, SEISMIC_CONVERSION_E_U, SEISMIC_SOURCE, SEISMIC_RESULT_0, SEISMIC_DIM_N, SEISMIC_DIM_K)

extern "C" __global__ void repack_weight(SEISMIC_KERNEL_PARAMS) {
    // The warp's staged source bytes (then its record words, or its staged
    // code words).
    __shared__ repack::u32 shared[repack::Shape<RepackBinding>::SHARED_WORDS];
    repack::convert_tile<RepackBinding>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE), SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER),
                                        seismic_words_value, blockIdx.z, blockIdx.x, blockIdx.y, threadIdx.x, shared);
}
