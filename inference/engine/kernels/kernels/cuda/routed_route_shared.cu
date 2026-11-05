// Decode routing (M <= DECODE_ROWS) with the shared expert's gate/up in one launch;
// the CUDA form of `metal/routed_route_shared.metal`. Blocks [0,
// route::BLOCKS) are `routed_route`'s GEMV blocks (`lib/routed/route.cuh`),
// whose selection counts only them. Every block stages the decode rows in
// its dynamic shared memory (`route::stage_rows`): a router block's columns
// read them there, and the shared blocks after the router blocks run
// `routed_expand`'s shared row over them (`lib/routed/shared.cuh`).

#define KERNEL_W0 SEISMIC_ROUTER
#define KERNEL_W1 SEISMIC_SHARED_GATE
#define KERNEL_W2 SEISMIC_SHARED_UP
#include <seismic/packets.cuh>
#include "lib/routed/route.cuh"
#include "lib/routed/shared.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

extern "C" __global__ void routed_route_shared(SEISMIC_KERNEL_PARAMS) {
    __shared__ float parts[SEISMIC_CONST_DECODE_ROWS][route::WARPS];
    __shared__ float inverses[SEISMIC_CONST_DECODE_ROWS];
    __shared__ unsigned last;
    extern __shared__ uint4 memory[];
    if (blockIdx.x >= route::BLOCKS) {
        shared_expert::block(ROUTE_INPUT, SEISMIC_PTR(SEISMIC_BUFFER_SHARED_GATE), SEISMIC_PTR(SEISMIC_BUFFER_SHARED_UP),
            SEISMIC_PTR(SEISMIC_RESULT_2_BUFFER), parts, inverses, reinterpret_cast<projection::u8 *>(memory),
            blockIdx.x - route::BLOCKS, seismic_words_value);
        return;
    }
    route::route_block(ROUTE_INPUT, route::StageRows{reinterpret_cast<unsigned char *>(memory)}, ROUTE_COLUMNS,
        ROUTE_SELECTION, SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SEISMIC_RESULT_0_STRIDE_0, SEISMIC_RESULT_0_STRIDE_1,
        reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS)),
        reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS)), blockIdx.x, parts, inverses,
        &last, seismic_words_value);
}
