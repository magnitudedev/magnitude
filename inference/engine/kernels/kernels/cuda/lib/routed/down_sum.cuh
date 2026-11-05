// The slot-order sum of the general routed feed-forward's decode combine,
// shared by `routed_down` and `routed_down_sited`: thread (n, m) publishes
// R(base + (sum_k w_k * projected_k + cpu)), the sum in slot order, over
// the projections in A at ROUTED_PROJECTED (`lib/routed/down.cuh`'s
// layout, `[M, K, H]`), `cpu` the CPU's answer where it answered the step.
// The includer includes `routed.cuh` first and defines ROUTED_PROJECTED and
// the answer hooks: `ROUTED_SITE_AWAIT(m)`, block-collective, and
// `ROUTED_SITE_ADD(m, n, selected)`, which adds row m's answer at channel n
// where the CPU answered (`lib/routed/sited.cuh`; nothing in `routed_down`).
using Published = ELEMENT_OF(SEISMIC_ELEMENT_R);

// Thread (n, m)'s body: R(base + (sum_k w_k * projected_k + cpu)), the sum
// in slot order. The includer defines the kernel `routed_down_combine` over it.
#define ROUTED_DOWN_COMBINE_BODY                                                                           \
    const projection::u64 n = (projection::u64)blockIdx.x * blockDim.x + threadIdx.x;                     \
    const projection::u64 m = blockIdx.y;                                                                  \
    ROUTED_SITE_AWAIT(m)                                                                                   \
    if (n >= SEISMIC_DIM_H)                                                                                \
        return;                                                                                            \
    const float *weights = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHTS));          \
    const element::Act::storage *projected = ROUTED_PROJECTED + m * SEISMIC_DIM_K * SEISMIC_DIM_H; \
    float selected = 0.0f;                                                                                 \
    for (projection::u64 k = 0; k < SEISMIC_DIM_K; ++k)                                                    \
        selected = __fmaf_rn(weights[m * SEISMIC_WEIGHTS_STRIDE_0 + k * SEISMIC_WEIGHTS_STRIDE_1],          \
                             element::Act::load(projected[k * SEISMIC_DIM_H + n]), selected);                   \
    ROUTED_SITE_ADD(m, n, selected)                                                                        \
    const float *base =                                                                                    \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BASE)) + m * SEISMIC_BASE_STRIDE_0;     \
    element::put<Published>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER) + m * SEISMIC_RESULT_0_STRIDE_0 * Published::bytes, \
                            n, base[n] + selected);
