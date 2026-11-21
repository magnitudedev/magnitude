// The slot-order sum of the general routed feed-forward's decode combine,
// shared by `routed_down` and `routed_down_sited`: thread (n, m) publishes
// R(base + (sum_k w_k * projected_k + cpu)), the sum in slot order, over
// the projections in A at ROUTED_PROJECTED (`lib/routed/down.cuh`'s
// layout, `[M, K, H]`), `cpu` the CPU's answer where it answered the step.
// The includer includes `routed.cuh` first and defines ROUTED_PROJECTED and
// the answer hooks: `ROUTED_SITE_AWAIT(m)`, block-collective, and
// `ROUTED_SITE_ADD(m, n, selected)`, which adds row m's answer at channel n
// where the CPU answered (`lib/routed/sited.cuh`; nothing in `routed_down`).
// Thread (t, m)'s body: R(base + (sum_k w_k * projected_k + cpu)), the sum
// in slot order, 4 contiguous channels per thread within the block's
// `blockDim.x`-channel tile (vectorized loads/stores when aligned, scalar
// fallback for unaligned or tail channels). The includer defines the kernel
// `routed_down_combine` over it.
#define ROUTED_DOWN_COMBINE_BODY                                                                           \
    seismic_dependency_start();                                                                            \
    const projection::u64 block_start = (projection::u64)blockIdx.x * blockDim.x;                          \
    const projection::u64 m = blockIdx.y;                                                                  \
    ROUTED_SITE_AWAIT(m)                                                                                   \
    const projection::u64 block_limit = block_start + blockDim.x;                                          \
    const projection::u64 block_end = block_limit < SEISMIC_DIM_H ? block_limit : SEISMIC_DIM_H;           \
    const projection::u64 n = block_start + (projection::u64)threadIdx.x * 4ull;                           \
    if (n >= block_end)                                                                                    \
        return;                                                                                            \
    const float *weights =                                                                                 \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHTS)) +                             \
        m * SEISMIC_WEIGHTS_STRIDE_0;                                                                      \
    const activation::Act::storage *projected = ROUTED_PROJECTED + m * SEISMIC_DIM_K * SEISMIC_DIM_H;      \
    const float *base =                                                                                    \
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BASE)) + m * SEISMIC_BASE_STRIDE_0;     \
    const auto row_out = SEISMIC_RESULT_0_OUT.offset(m * SEISMIC_RESULT_0_STRIDE_0);                       \
    using ResultElem = ELEMENT_OF(SEISMIC_RESULT_0);                                                       \
    const bool aligned =                                                                                   \
        (n + 4ull <= block_end) &&                                                                         \
        (SEISMIC_DIM_K <= 1ull || (SEISMIC_DIM_H & 3ull) == 0ull) &&                                       \
        ((reinterpret_cast<projection::u64>(projected + n) & (4ull * activation::Act::bytes - 1ull)) ==    \
         0ull) &&                                                                                          \
        ((reinterpret_cast<projection::u64>(base + n) & 15ull) == 0ull) &&                                 \
        ((reinterpret_cast<projection::u64>(row_out.stored(n)) & (4ull * ResultElem::bytes - 1ull)) ==     \
         0ull);                                                                                            \
    if (aligned) {                                                                                         \
        const element::u8 *projected_bytes = reinterpret_cast<const element::u8 *>(projected);             \
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;                                                  \
        for (projection::u64 k = 0; k < SEISMIC_DIM_K; ++k) {                                              \
            const float w = weights[k * SEISMIC_WEIGHTS_STRIDE_1];                                         \
            float vals[4];                                                                                 \
            element::span<activation::Act, 4, true>(projected_bytes, k * SEISMIC_DIM_H + n, vals);         \
            s0 = __fmaf_rn(w, vals[0], s0);                                                                \
            s1 = __fmaf_rn(w, vals[1], s1);                                                                \
            s2 = __fmaf_rn(w, vals[2], s2);                                                                \
            s3 = __fmaf_rn(w, vals[3], s3);                                                                \
        }                                                                                                  \
        ROUTED_SITE_ADD(m, n + 0ull, s0)                                                                   \
        ROUTED_SITE_ADD(m, n + 1ull, s1)                                                                   \
        ROUTED_SITE_ADD(m, n + 2ull, s2)                                                                   \
        ROUTED_SITE_ADD(m, n + 3ull, s3)                                                                   \
        const float4 b = *reinterpret_cast<const float4 *>(base + n);                                      \
        const float out4[4] = {b.x + s0, b.y + s1, b.z + s2, b.w + s3};                                    \
        row_out.put_span<4>(n, out4);                                                                      \
    } else {                                                                                               \
        const projection::u64 count = block_end - n < 4ull ? block_end - n : 4ull;                         \
        for (projection::u64 i = 0; i < count; ++i) {                                                      \
            const projection::u64 col = n + i;                                                             \
            float selected = 0.0f;                                                                         \
            for (projection::u64 k = 0; k < SEISMIC_DIM_K; ++k)                                            \
                selected = __fmaf_rn(weights[k * SEISMIC_WEIGHTS_STRIDE_1],                                \
                                     activation::Act::load(projected[k * SEISMIC_DIM_H + col]), selected); \
            ROUTED_SITE_ADD(m, col, selected)                                                              \
            row_out.put(col, base[col] + selected);                                                        \
        }                                                                                                  \
    }
