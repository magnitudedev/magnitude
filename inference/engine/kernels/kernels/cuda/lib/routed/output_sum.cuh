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
                                 activation::Act::load(__ldcg((row_published) + k * join_h + (n))), selected); \
        ROUTED_SITE_ADD(row, n, selected)                                                                  \
        projection::routed_join<SEISMIC_DIM_SN, (SEISMIC_DIM_IN != 0)>(                                    \
            SEISMIC_RESULT_0_OUT.offset((row) * SEISMIC_RESULT_0_STRIDE_0),                                \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)) +                        \
                (row) * SEISMIC_RESIDUAL_STRIDE_0,                                                         \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INJECT)) + (row) * SEISMIC_INJECT_STRIDE_0, \
            join_h, (n), selected,                                                                         \
            activation::Act::load(__ldcg((row_published) + join_k * join_h + (n))) *                                             \
                reinterpret_cast<const float *>(                                                           \
                    SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENT))[(row) * SEISMIC_COEFFICIENT_STRIDE_0]);       \
    } while (0)

// Row `row`'s 4 contiguous channels `n..n+3` over its published projections `row_published`.
// Loads published projections with `__ldcg` (L2-coherent with other blocks' stores in the same
// launch), hoisting scores, shared coefficient, and inject gains once per 4 channels.
#define ROUTED_OUTPUT_JOIN4(row, n, row_published)                                                             \
    do {                                                                                                       \
        const projection::u64 join_row = (row), join_n = (n);                                                  \
        const projection::u64 join_k = SEISMIC_DIM_K, join_h = SEISMIC_DIM_H;                                  \
        const activation::Act::storage *join_pub = (row_published);                                            \
        const bool join_pub_aligned =                                                                          \
            ((join_h & 3ull) == 0ull) &&                                                                       \
            ((reinterpret_cast<projection::u64>(join_pub + join_n) &                                           \
              (4ull * activation::Act::bytes - 1ull)) == 0ull);                                                \
        if (!join_pub_aligned) {                                                                               \
            ROUTED_OUTPUT_JOIN(join_row, join_n + 0ull, join_pub);                                             \
            ROUTED_OUTPUT_JOIN(join_row, join_n + 1ull, join_pub);                                             \
            ROUTED_OUTPUT_JOIN(join_row, join_n + 2ull, join_pub);                                             \
            ROUTED_OUTPUT_JOIN(join_row, join_n + 3ull, join_pub);                                             \
            break;                                                                                             \
        }                                                                                                      \
        const float *join_scores =                                                                             \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES)) +                              \
            join_row * SEISMIC_SCORES_STRIDE_0;                                                                \
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;                                                      \
        for (projection::u64 k = 0; k < join_k; ++k) {                                                         \
            const float score = join_scores[k * SEISMIC_SCORES_STRIDE_1];                                      \
            const activation::Act::storage *slot_ptr = join_pub + k * join_h + join_n;                         \
            float vals[4];                                                                                     \
            if constexpr (activation::Act::bytes == 4) {                                                       \
                const float4 raw = __ldcg(reinterpret_cast<const float4 *>(slot_ptr));                         \
                vals[0] = raw.x;                                                                               \
                vals[1] = raw.y;                                                                               \
                vals[2] = raw.z;                                                                               \
                vals[3] = raw.w;                                                                               \
            } else {                                                                                           \
                const uint2 raw = __ldcg(reinterpret_cast<const uint2 *>(slot_ptr));                           \
                vals[0] = activation::Act::load(static_cast<activation::Act::storage>(raw.x & 0xffffu));       \
                vals[1] = activation::Act::load(static_cast<activation::Act::storage>(raw.x >> 16));           \
                vals[2] = activation::Act::load(static_cast<activation::Act::storage>(raw.y & 0xffffu));       \
                vals[3] = activation::Act::load(static_cast<activation::Act::storage>(raw.y >> 16));           \
            }                                                                                                  \
            s0 = __fmaf_rn(score, vals[0], s0);                                                                \
            s1 = __fmaf_rn(score, vals[1], s1);                                                                \
            s2 = __fmaf_rn(score, vals[2], s2);                                                                \
            s3 = __fmaf_rn(score, vals[3], s3);                                                                \
        }                                                                                                      \
        ROUTED_SITE_ADD(join_row, join_n + 0ull, s0)                                                           \
        ROUTED_SITE_ADD(join_row, join_n + 1ull, s1)                                                           \
        ROUTED_SITE_ADD(join_row, join_n + 2ull, s2)                                                           \
        ROUTED_SITE_ADD(join_row, join_n + 3ull, s3)                                                           \
        const float join_coeff =                                                                               \
            reinterpret_cast<const float *>(                                                                   \
                SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENT))[join_row * SEISMIC_COEFFICIENT_STRIDE_0];             \
        float sh0, sh1, sh2, sh3;                                                                              \
        {                                                                                                      \
            const activation::Act::storage *shared_ptr = join_pub + join_k * join_h + join_n;                  \
            float svals[4];                                                                                    \
            if constexpr (activation::Act::bytes == 4) {                                                       \
                const float4 raw = __ldcg(reinterpret_cast<const float4 *>(shared_ptr));                       \
                svals[0] = raw.x;                                                                              \
                svals[1] = raw.y;                                                                              \
                svals[2] = raw.z;                                                                              \
                svals[3] = raw.w;                                                                              \
            } else {                                                                                           \
                const uint2 raw = __ldcg(reinterpret_cast<const uint2 *>(shared_ptr));                         \
                svals[0] = activation::Act::load(static_cast<activation::Act::storage>(raw.x & 0xffffu));      \
                svals[1] = activation::Act::load(static_cast<activation::Act::storage>(raw.x >> 16));          \
                svals[2] = activation::Act::load(static_cast<activation::Act::storage>(raw.y & 0xffffu));      \
                svals[3] = activation::Act::load(static_cast<activation::Act::storage>(raw.y >> 16));          \
            }                                                                                                  \
            sh0 = svals[0] * join_coeff;                                                                       \
            sh1 = svals[1] * join_coeff;                                                                       \
            sh2 = svals[2] * join_coeff;                                                                       \
            sh3 = svals[3] * join_coeff;                                                                       \
        }                                                                                                      \
        const auto join_out = SEISMIC_RESULT_0_OUT.offset(join_row * SEISMIC_RESULT_0_STRIDE_0);               \
        const float *join_res = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)) +        \
                                join_row * SEISMIC_RESIDUAL_STRIDE_0;                                          \
        using JoinResultElem = ELEMENT_OF(SEISMIC_RESULT_0);                                                   \
        if constexpr (SEISMIC_DIM_IN == 0) {                                                                   \
            const float4 base =                                                                                \
                ((reinterpret_cast<projection::u64>(join_res + join_n) & 15ull) == 0ull)                       \
                    ? *reinterpret_cast<const float4 *>(join_res + join_n)                                     \
                    : make_float4(join_res[join_n], join_res[join_n + 1ull], join_res[join_n + 2ull],          \
                                  join_res[join_n + 3ull]);                                                    \
            const float out4[4] = {(base.x + s0) + sh0, (base.y + s1) + sh1, (base.z + s2) + sh2,              \
                                   (base.w + s3) + sh3};                                                       \
            if ((reinterpret_cast<projection::u64>(join_out.stored(join_n)) &                                  \
                 (4ull * JoinResultElem::bytes - 1ull)) == 0ull) {                                             \
                join_out.put_span<4>(join_n, out4);                                                            \
            } else {                                                                                           \
                join_out.put(join_n + 0ull, out4[0]);                                                          \
                join_out.put(join_n + 1ull, out4[1]);                                                          \
                join_out.put(join_n + 2ull, out4[2]);                                                          \
                join_out.put(join_n + 3ull, out4[3]);                                                          \
            }                                                                                                  \
        } else {                                                                                               \
            const float *join_inj = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INJECT)) +      \
                                    join_row * SEISMIC_INJECT_STRIDE_0;                                        \
            const float t0 = s0 + sh0, t1 = s1 + sh1, t2 = s2 + sh2, t3 = s3 + sh3;                            \
            _Pragma("unroll")                                                                                  \
            for (projection::u32 s = 0; s < SEISMIC_DIM_SN; ++s) {                                             \
                const float gain = join_inj[s];                                                                \
                const projection::u64 at = (projection::u64)s * join_h + join_n;                               \
                const float4 base =                                                                            \
                    ((reinterpret_cast<projection::u64>(join_res + at) & 15ull) == 0ull)                       \
                        ? *reinterpret_cast<const float4 *>(join_res + at)                                     \
                        : make_float4(join_res[at], join_res[at + 1ull], join_res[at + 2ull],                  \
                                      join_res[at + 3ull]);                                                    \
                const float out4[4] = {base.x + t0 * gain, base.y + t1 * gain, base.z + t2 * gain,             \
                                       base.w + t3 * gain};                                                    \
                if ((reinterpret_cast<projection::u64>(join_out.stored(at)) &                                  \
                     (4ull * JoinResultElem::bytes - 1ull)) == 0ull) {                                         \
                    join_out.put_span<4>(at, out4);                                                            \
                } else {                                                                                       \
                    join_out.put(at + 0ull, out4[0]);                                                          \
                    join_out.put(at + 1ull, out4[1]);                                                          \
                    join_out.put(at + 2ull, out4[2]);                                                          \
                    join_out.put(at + 3ull, out4[3]);                                                          \
                }                                                                                              \
            }                                                                                                  \
        }                                                                                                      \
    } while (0)
