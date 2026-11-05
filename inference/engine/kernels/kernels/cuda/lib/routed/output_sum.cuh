// The join of Qwen's fused routed feed-forward's decode output, shared by
// `routed_output` (its last arriving block's channels) and
// `routed_output_sited` (a thread a channel of a row): row m's channel n is
//     residual + selected + shared * c,
// `selected` the published choice projections weighted by their scores in
// slot order, plus the CPU's answer where it answered, `shared` slot K's
// projection, over the projections in A of `lib/routed/output.cuh`'s
// layout (`[M, K + 1, H]`, row-major). A macro, expanded where it is used
// (past `routed.cuh`), so an includer may include it before
// `lib/routed/output.cuh`, whose hooks expand it; the includer defines
// ROUTED_PUBLISHED and the answer hooks (`lib/routed/down_sum.cuh` states
// them).

// Row `row`'s channel `n` over its published projections `row_published`.
#define ROUTED_OUTPUT_JOIN(row, n, row_published)                                                          \
    do {                                                                                                   \
        const projection::u64 join_k = SEISMIC_DIM_K, join_h = SEISMIC_DIM_H;                              \
        const float *join_scores = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES));     \
        float selected = 0.0f;                                                                             \
        for (projection::u64 k = 0; k < join_k; ++k)                                                       \
            selected = __fmaf_rn(join_scores[(row) * SEISMIC_SCORES_STRIDE_0 + k * SEISMIC_SCORES_STRIDE_1], \
                                 element::Act::load(__ldcg((row_published) + k * join_h + (n))), selected); \
        ROUTED_SITE_ADD(row, n, selected)                                                                  \
        projection::routed_join<SEISMIC_DIM_SN, (SEISMIC_DIM_IN != 0)>(                                    \
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)) + (row) * SEISMIC_RESULT_0_STRIDE_0, \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)) +                        \
                (row) * SEISMIC_RESIDUAL_STRIDE_0,                                                         \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INJECT)) + (row) * SEISMIC_INJECT_STRIDE_0, \
            join_h, (n), selected,                                                                         \
            element::Act::load(__ldcg((row_published) + join_k * join_h + (n))) *                                             \
                reinterpret_cast<const float *>(                                                           \
                    SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENT))[(row) * SEISMIC_COEFFICIENT_STRIDE_0]);       \
    } while (0)
