// Expert streaming, a bank shrink's record moves (`routed_move` in
// routed.seismic); the CUDA form of `metal/routed_move.metal`: the listed
// records of a group's per-record tensor into its other records, byte for
// byte (streaming/list_copy.hpp). Block rows stride over the list's
// entries; the threads of a row over each record. No entry's destination is
// another's source, so the entries are independent.

#include "lib/streaming/control.cuh"
#include "lib/repack/repack.cuh"
#include "../streaming/list.hpp"

#define STREAMING_COPY_FROM(table, record) ((table) + u64(record) * REPACK_MATRIX_BYTES(SEISMIC_RECORDS))
#define STREAMING_COPY_TO(table, record) ((table) + u64(record) * REPACK_MATRIX_BYTES(SEISMIC_RECORDS))
#include "../streaming/list_copy.hpp"

extern "C" __global__ void routed_move(SEISMIC_KERNEL_PARAMS) {
    __shared__ int words[STREAMING_LIST_WORDS];
    streaming_list_move(SEISMIC_PTR(SEISMIC_BUFFER_RECORDS), reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_LIST)),
                        REPACK_MATRIX_BYTES(SEISMIC_RECORDS), blockIdx.y, gridDim.y,
                        (u64)blockIdx.x * blockDim.x + threadIdx.x, (u64)gridDim.x * blockDim.x, words, threadIdx.x,
                        blockDim.x, seismic_words_value);
}
