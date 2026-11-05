// Expert streaming, a prompt group's seeds (`routed_copy` in
// routed.seismic); the CUDA form of `metal/routed_copy.metal`: the listed
// records of the source into the destination, byte for byte
// (streaming/list_copy.hpp). It is the body of one block of its list's
// written grid (`routed_list_grid`, `_block`: none for a list of no
// entries).

#include "lib/streaming/control.cuh"
#include "lib/repack/repack.cuh"
#include "../streaming/list.hpp"

#define STREAMING_COPY_FROM(table, record) ((table) + u64(record) * REPACK_MATRIX_BYTES(SEISMIC_SOURCE))
#define STREAMING_COPY_TO(table, record) ((table) + u64(record) * REPACK_MATRIX_BYTES(SEISMIC_DESTINATION))
#include "../streaming/list_copy.hpp"

#ifdef SEISMIC_FORMING_ROUTED_COPY
__device__ void routed_copy_block(SEISMIC_KERNEL_PARAMS) {
    __shared__ int words[STREAMING_LIST_WORDS];
    streaming_list_copy(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE), reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_LIST)),
                        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP))[0],
                        SEISMIC_PTR(SEISMIC_BUFFER_DESTINATION), REPACK_MATRIX_BYTES(SEISMIC_SOURCE),
                        blockIdx.x + gridDim.x * blockIdx.y, gridDim.x * gridDim.y, words, threadIdx.x, blockDim.x,
                        seismic_words_value);
}
#endif
