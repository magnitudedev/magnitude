// Expert streaming, a streamed decode layer's request to the CPU
// (`routed_site_request` in routed.seismic): when a choice is the CPU's, the
// threadgroup's threads widen the normalized rows into the step's slot's
// F32 rows through system-coherent stores, then thread 0 publishes the
// request in the slot (`streaming/site.hpp`).

#include <seismic/element.h>
#include "lib/streaming/control.h"
#include "../streaming/site.hpp"

kernel void routed_site_request(
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    device const int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],
    device const float *weights [[buffer(SEISMIC_BUFFER_WEIGHTS)]],
    device const uchar *normalized [[buffer(SEISMIC_BUFFER_NORMALIZED)]],
    SYSTEM int *site [[buffer(SEISMIC_BUFFER_SITE)]],
    SYSTEM float *site_rows [[buffer(SEISMIC_BUFFER_SITE_ROWS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    typedef ELEMENT_OF(SEISMIC_ELEMENT_A) A;
    const uint choices = uint(SEISMIC_DIM_M) * uint(SEISMIC_DIM_K);
    if (!streaming_site_any(sources, SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2,
            choices))
        return;
    device const typename A::storage *rows_in = reinterpret_cast<device const typename A::storage *>(normalized);
    const ulong q = streaming_step(step, SEISMIC_STEP_STRIDE_0);
    const ulong slot = q % ulong(SEISMIC_DIM_FLIGHT);
    const ulong rows = SEISMIC_DIM_M, columns = SEISMIC_DIM_H;
    for (ulong i = tid; i < rows * columns; i += threads) {
        const ulong m = i / columns, n = i % columns;
        site_rows[slot * SEISMIC_SITE_ROWS_STRIDE_0 + m * SEISMIC_SITE_ROWS_STRIDE_1
            + n * SEISMIC_SITE_ROWS_STRIDE_2] = A::load(rows_in[m * SEISMIC_NORMALIZED_STRIDE_0 + n * SEISMIC_NORMALIZED_STRIDE_1]);
    }
    streaming_publish_barrier();
    if (tid == 0)
        streaming_site_publish(q, routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1,
            reinterpret_cast<device const int *>(weights), SEISMIC_WEIGHTS_STRIDE_0, SEISMIC_WEIGHTS_STRIDE_1, sources,
            SEISMIC_SOURCES_STRIDE_0, SEISMIC_SOURCES_STRIDE_1, SEISMIC_SOURCES_STRIDE_2, choices,
            site + slot * SEISMIC_SITE_STRIDE_0, SEISMIC_SITE_STRIDE_1);
}
