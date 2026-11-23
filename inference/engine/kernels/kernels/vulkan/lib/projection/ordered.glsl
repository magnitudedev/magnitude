// Reference-ordered bodies of the `dense` projection for Vulkan (`dense.seismic`
// with `ordered (..)`), reading only the tuning parameters ordered entries
// declare (`SEISMIC_TUNE_SIMDGROUPS`, with `rows` and `lanes` passed by mixed
// segmented entries).

#ifndef PROJECTION_G
#define PROJECTION_G 1
#endif
#define PROJECTION_GROUP(z) (PROJECTION_G == 1 ? 0u : uint(z))

#ifndef PROJECTION_ORDERED_PUBLISH
#define PROJECTION_ORDERED_PUBLISH false
#endif
#ifndef PROJECTION_ORDERED_PUB_OUT
#define PROJECTION_ORDERED_PUB_OUT out_over_scratch(ELEMENT_F16, 0ul)
#define PROJECTION_ORDERED_PUB_S0 0ul
#define PROJECTION_ORDERED_PUB_S1 1ul
#endif
#ifndef PROJECTION_ORDERED_FIRST_TILE
#define PROJECTION_ORDERED_FIRST_TILE()
#endif

#ifdef PROJECTION_SINGLE
void projection_single_ordered_gemv() {
    packets_stage_tables();
    const uint g = PROJECTION_GROUP(gl_WorkGroupID.z);
    projection_ordered_gemv(PACKETS_W0, PROJECTION_ORDERED_PUBLISH, uint(SEISMIC_CONST_VERIFY_ROWS), projected_input(g), projected_output(g),
        projected_weight(g), PROJECTION_ORDERED_PUB_OUT, PROJECTION_ORDERED_PUB_S0, PROJECTION_ORDERED_PUB_S1,
        gl_WorkGroupID.x == 0u, PROJECTION_M, PROJECTION_N, PROJECTION_K, gl_WorkGroupID.x);
    if (gl_WorkGroupID.x == 0u) {
        PROJECTION_ORDERED_FIRST_TILE()
    }
}

void projection_single_ordered_gemm() {
    packets_stage_tables();
    const uint g = PROJECTION_GROUP(gl_WorkGroupID.z);
    projection_ordered_gemm(PACKETS_W0, projected_rows(g), projected_output(g), projected_weight(g), PROJECTION_M,
        PROJECTION_N, PROJECTION_K, gl_WorkGroupID.y, gl_WorkGroupID.x);
}
#endif

#if SEISMIC_PROJECTION_SEGMENTS
#include "segments.glsl"

#ifndef PROJECTION_SEGMENT_INPUT
#define PROJECTION_SEGMENT_INPUT(s) projected_input(0u)
#endif

#define PROJECTION_ORDERED_SEG_GEMV_ALL_STEP(s)                                                         \
    {                                                                                                   \
        const uint tiles = (uint(PROJECTION_N##s) + ord_per - 1u) / ord_per;                            \
        if (tile < tiles) {                                                                             \
            if (PROJECTION_ZEROED(s))                                                                   \
                projection_zero_tile(PROJECTION_OUTPUT##s, m, uint(PROJECTION_N##s), 0u, m,             \
                    tile * ord_per, ord_per);                                                           \
            else                                                                                        \
                projection_ordered_gemv(PROJECTION_KIND##s, PROJECTION_ORDERED_PUBLISH, uint(SEISMIC_CONST_VERIFY_ROWS), \
                    PROJECTION_SEGMENT_INPUT(s), PROJECTION_OUTPUT##s, PROJECTION_WEIGHT##s,            \
                    PROJECTION_ORDERED_PUB_OUT, PROJECTION_ORDERED_PUB_S0, PROJECTION_ORDERED_PUB_S1,   \
                    gl_WorkGroupID.x == 0u, m, uint(PROJECTION_N##s), PROJECTION_K, tile);              \
            if (gl_WorkGroupID.x == 0u) { PROJECTION_ORDERED_FIRST_TILE() }                             \
            return;                                                                                     \
        }                                                                                               \
        tile -= tiles;                                                                                  \
    }

void projection_segmented_ordered_gemv_all() {
    packets_stage_tables();
    const uint ord_per = 4u * SEISMIC_TUNE_SIMDGROUPS;
    const uint m = PROJECTION_M;
    uint tile = gl_WorkGroupID.x;
    PROJECTION_EACH_SEGMENT(PROJECTION_ORDERED_SEG_GEMV_ALL_STEP)
}

#define PROJECTION_ORDERED_SEG_GEMV_MIXED_STEP(s)                                                       \
    {                                                                                                   \
        const uint per_s = SEISMIC_SEGMENT_ORDERED_##s != 0 ? ord_per : unord_per;                      \
        const uint tiles = (uint(PROJECTION_N##s) + per_s - 1u) / per_s;                                \
        if (tile < tiles) {                                                                             \
            if (PROJECTION_ZEROED(s)) {                                                                 \
                projection_zero_tile(PROJECTION_OUTPUT##s, m, uint(PROJECTION_N##s), 0u, m,             \
                    tile * per_s, per_s);                                                               \
            } else if (SEISMIC_SEGMENT_ORDERED_##s != 0) {                                              \
                projection_ordered_gemv(PROJECTION_KIND##s, PROJECTION_ORDERED_PUBLISH,                 \
                    maxm == 8u ? uint(SEISMIC_CONST_VERIFY_ROWS) : maxm,                                \
                    PROJECTION_SEGMENT_INPUT(s), PROJECTION_OUTPUT##s, PROJECTION_WEIGHT##s,            \
                    PROJECTION_ORDERED_PUB_OUT, PROJECTION_ORDERED_PUB_S0, PROJECTION_ORDERED_PUB_S1,   \
                    gl_WorkGroupID.x == 0u, m, uint(PROJECTION_N##s), PROJECTION_K, tile);              \
            } else {                                                                                    \
                for (uint row_base = 0u; row_base < m; row_base += maxm) {                              \
                    const uint slice_m = min(maxm, m - row_base);                                       \
                    projection_prologue seg_in = PROJECTION_SEGMENT_INPUT(s);                           \
                    projection_epilogue seg_out = PROJECTION_OUTPUT##s;                                 \
                    if (row_base != 0u) {                                                               \
                        seg_in.x = stored_offset(seg_in.x, uint64_t(row_base) * seg_in.x0);             \
                        seg_out.y = out_offset(seg_out.y, uint64_t(row_base) * seg_out.y0);             \
                        seg_out.y_f32 = out_offset(seg_out.y_f32, uint64_t(row_base) * seg_out.y0);     \
                    }                                                                                   \
                    projection_gemv(PROJECTION_KIND##s, PROJECTION_PARTNER_KIND##s, PROJECTION_PAIRED##s, \
                        rows, lanes, maxm, SEISMIC_TUNE_STAGE == 1, SEISMIC_TUNE_DECODE == 1,           \
                        PROJECTION_NORM, seg_in, seg_out, PROJECTION_WEIGHT##s,                         \
                        PROJECTION_PARTNER##s, slice_m, uint(PROJECTION_N##s), PROJECTION_K, tile);     \
                    if (row_base + maxm < m)                                                            \
                        barrier();                                                                      \
                }                                                                                       \
            }                                                                                           \
            if (gl_WorkGroupID.x == 0u) { PROJECTION_ORDERED_FIRST_TILE() }                             \
            return;                                                                                     \
        }                                                                                               \
        tile -= tiles;                                                                                  \
    }

void projection_segmented_ordered_gemv_mixed(const uint maxm, const uint rows, const uint lanes) {
    packets_stage_tables();
    const uint ord_per = 4u * SEISMIC_TUNE_SIMDGROUPS;
    const uint unord_per = projection_gemv_rows(SEISMIC_TUNE_SIMDGROUPS, rows, lanes);
    const uint m = PROJECTION_M;
    uint tile = gl_WorkGroupID.x;
    PROJECTION_EACH_SEGMENT(PROJECTION_ORDERED_SEG_GEMV_MIXED_STEP)
}

#define PROJECTION_ORDERED_SEG_GEMM_STEP(s)                                                             \
    if (!found) {                                                                                       \
        const uint tiles = (uint(PROJECTION_N##s) + 31u) / 32u;                                         \
        if (tile < tiles) {                                                                             \
            if (PROJECTION_ZEROED(s)) {                                                                 \
                projection_zero_tile(PROJECTION_OUTPUT##s, m, uint(PROJECTION_N##s),                    \
                    gl_WorkGroupID.y * 64u, 64u, tile * 32u, 32u);                                      \
                return;                                                                                 \
            }                                                                                           \
            found = true;                                                                               \
            kind = PROJECTION_KIND##s;                                                                  \
            w = PROJECTION_WEIGHT##s;                                                                   \
            out_ = PROJECTION_OUTPUT##s;                                                                \
            rows = uint(PROJECTION_N##s);                                                               \
        } else {                                                                                        \
            tile -= tiles;                                                                              \
        }                                                                                               \
    }

void projection_segmented_ordered_gemm() {
    packets_stage_tables();
    const uint m = PROJECTION_M;
    const projection_prologue x = projected_rows(0u);
    uint tile = gl_WorkGroupID.x;
    bool found = false;
    int kind = PROJECTION_KIND0;
    projection_weights w;
    projection_epilogue out_;
    uint rows = 0u;
    PROJECTION_EACH_SEGMENT(PROJECTION_ORDERED_SEG_GEMM_STEP)
    projection_ordered_gemm(kind, x, out_, w, m, rows, PROJECTION_K, gl_WorkGroupID.y, tile);
}
#endif
