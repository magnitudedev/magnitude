// `repack_weight`: the registered GGUF -> resident conversion of a whole
// weight (<seismic/repack.cuh>): E is the GGUF source, U the resident
// (representation, layout); the entry sees every weight as a [B, N, K] view.
// Each block, one warp, places one row tile: block x its column tile, y its
// row tile, z its matrix. The result is a record port (bank records import
// through it): matrix z is written from its own record's base.

#include <seismic/slab.cuh>
#include <seismic/repack.cuh>

REPACK_BINDING(RepackBinding, SEISMIC_CONVERSION_E_U, SEISMIC_SOURCE, SEISMIC_RESULT_0, SEISMIC_DIM_N, SEISMIC_DIM_K)

extern "C" __global__ void repack_weight(SEISMIC_KERNEL_PARAMS) {
    // The warp's staged source bytes (then its record words, or its staged
    // code words).
    __shared__ repack::u32 shared[repack::Shape<RepackBinding>::SHARED_WORDS];
    const repack::u64 matrix = blockIdx.z;
    repack::convert_tile<RepackBinding>(
        repack::Strided<RepackBinding>{SEISMIC_PTR(SEISMIC_BUFFER_SOURCE) +
                                       matrix * RepackBinding::source_matrix_bytes(seismic_words_value)},
        slab::region(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), matrix),
        seismic_words_value, 0, blockIdx.x, blockIdx.y, threadIdx.x, shared);
}
