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
// Join one channel `n` of row `m` from row `m`'s `projected` projections.
#define ROUTED_DOWN_JOIN(m, n, projected)                                                                  \
    do {                                                                                                   \
        const projection::u64 down_m = (m);                                                                \
        const projection::u64 down_col = (n);                                                              \
        const float *down_weights =                                                                        \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHTS)) +                         \
            down_m * SEISMIC_WEIGHTS_STRIDE_0;                                                             \
        const float *down_base =                                                                           \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BASE)) +                            \
            down_m * SEISMIC_BASE_STRIDE_0;                                                                \
        const auto down_out = SEISMIC_RESULT_0_OUT.offset(down_m * SEISMIC_RESULT_0_STRIDE_0);             \
        const activation::Act::storage *down_proj = (projected);                                           \
        float selected = 0.0f;                                                                             \
        for (projection::u64 k = 0; k < SEISMIC_DIM_K; ++k)                                                \
            selected = __fmaf_rn(down_weights[k * SEISMIC_WEIGHTS_STRIDE_1],                               \
                                 activation::Act::load(down_proj[k * SEISMIC_DIM_H + down_col]), selected); \
        ROUTED_SITE_ADD(down_m, down_col, selected)                                                        \
        down_out.put(down_col, down_base[down_col] + selected);                                            \
    } while (0)

// Join four contiguous channels `n .. n + 3` of row `m` from row `m`'s
// `projected` projections (vectorized when aligned, scalar fallback otherwise).
#define ROUTED_DOWN_JOIN4(m, n, projected)                                                                 \
    do {                                                                                                   \
        const projection::u64 down_m = (m);                                                                \
        const projection::u64 down_n = (n);                                                                \
        const float *down_weights =                                                                        \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHTS)) +                         \
            down_m * SEISMIC_WEIGHTS_STRIDE_0;                                                             \
        const float *down_base =                                                                           \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BASE)) +                            \
            down_m * SEISMIC_BASE_STRIDE_0;                                                                \
        const auto down_out = SEISMIC_RESULT_0_OUT.offset(down_m * SEISMIC_RESULT_0_STRIDE_0);             \
        const activation::Act::storage *down_proj = (projected);                                           \
        using ResultElem = ELEMENT_OF(SEISMIC_RESULT_0);                                                   \
        const bool down_aligned =                                                                          \
            (SEISMIC_DIM_K <= 1ull || (SEISMIC_DIM_H & 3ull) == 0ull) &&                                   \
            ((reinterpret_cast<projection::u64>(down_proj + down_n) &                                      \
              (4ull * activation::Act::bytes - 1ull)) == 0ull) &&                                          \
            ((reinterpret_cast<projection::u64>(down_base + down_n) & 15ull) == 0ull) &&                   \
            ((reinterpret_cast<projection::u64>(down_out.stored(down_n)) &                                 \
              (4ull * ResultElem::bytes - 1ull)) == 0ull);                                                 \
        if (down_aligned) {                                                                                \
            const element::u8 *projected_bytes = reinterpret_cast<const element::u8 *>(down_proj);         \
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;                                              \
            for (projection::u64 k = 0; k < SEISMIC_DIM_K; ++k) {                                          \
                const float w = down_weights[k * SEISMIC_WEIGHTS_STRIDE_1];                                \
                float vals[4];                                                                             \
                element::span<activation::Act, 4, true>(projected_bytes, k * SEISMIC_DIM_H + down_n, vals); \
                s0 = __fmaf_rn(w, vals[0], s0);                                                            \
                s1 = __fmaf_rn(w, vals[1], s1);                                                            \
                s2 = __fmaf_rn(w, vals[2], s2);                                                            \
                s3 = __fmaf_rn(w, vals[3], s3);                                                            \
            }                                                                                              \
            ROUTED_SITE_ADD(down_m, down_n + 0ull, s0)                                                     \
            ROUTED_SITE_ADD(down_m, down_n + 1ull, s1)                                                     \
            ROUTED_SITE_ADD(down_m, down_n + 2ull, s2)                                                     \
            ROUTED_SITE_ADD(down_m, down_n + 3ull, s3)                                                     \
            const float4 b = *reinterpret_cast<const float4 *>(down_base + down_n);                        \
            const float out4[4] = {b.x + s0, b.y + s1, b.z + s2, b.w + s3};                                \
            down_out.put_span<4>(down_n, out4);                                                            \
        } else {                                                                                           \
            for (projection::u64 i = 0; i < 4ull; ++i)                                                     \
                ROUTED_DOWN_JOIN(down_m, down_n + i, down_proj);                                           \
        }                                                                                                  \
    } while (0)

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
    const activation::Act::storage *projected = ROUTED_PROJECTED + m * SEISMIC_DIM_K * SEISMIC_DIM_H;      \
    if (n + 4ull <= block_end) {                                                                           \
        ROUTED_DOWN_JOIN4(m, n, projected);                                                                \
    } else {                                                                                               \
        for (projection::u64 col = n; col < block_end; ++col)                                              \
            ROUTED_DOWN_JOIN(m, col, projected);                                                           \
    }
