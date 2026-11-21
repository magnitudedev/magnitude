// The bodies of the `dense` projection's forms for Vulkan (`dense.seismic`),
// once for every entry that projects: an entry states its prologue (the rows
// it projects), its epilogue and its weights, and each of its launches runs
// the body its name gives here. A projection is `single` (one weight, which
// may be grouped), `paired` (a gate and an up weight one activation row
// serves, combined by the epilogue) or `segmented` (below).
//
// The includer includes `projection.glsl` first and defines, before including
// this file, its variant (PROJECTION_SINGLE or PROJECTION_PAIR; a
// `segmented` entry's prefix defines SEISMIC_PROJECTION_SEGMENTS, the number
// of its segments) and
//   PROJECTION_M, PROJECTION_K  the rows and their columns;
//   PROJECTION_N                each weight's rows (`single`, `paired`);
//   PROJECTION_G                the groups of a `single` weight (1 where it
//                               has none): group g projects its own columns of
//                               the rows by its own N weight rows into its own
//                               N columns, at grid z (z = g * SPLIT + part in
//                               the GEMM), into its own G-th of the partials;
//   PROJECTION_NORM             PROJECTION_NORM_SHARED where the GEMV
//                               normalizes the rows while staging them (`rms`
//                               rows), else PROJECTION_NORM_NONE;
//   PROJECTION_STAGED           the stage launch's scratch (`staged`);
//   PROJECTION_SPLIT, PROJECTION_PARTIALS
//                               the GEMM's split of K (SPLIT) and its
//                               partials (`single`);
// and the functions
//   projected_input(g)          the rows the GEMV forms its operands from and
//                               the stage launch stages (group g's), as the
//                               weights' operands;
//   projected_rows(g)           the rows every GEMM reads: the staged rows, or
//                               the rows in place;
//   projected_output(g), projected_weight(g)
//                               the epilogue and the weight (`single`; group
//                               g's), or the gate weight (`paired`, g = 0),
//   projected_partner(g)        the up weight (`paired`).
// A `segmented` projection defines instead the segments' macros (below).
// Every body that decodes the weights first stages their tables
// (`packets_stage_tables`), which an includer that defines
// PACKETS_CONFIG_STAGED_TABLES before including `projection.glsl` reads from the
// end of each such launch's shared region (`dense.seismic` holds them there).

// The GEMM's shape: the declaration's tuned tile and sub-tile.
#define PROJECTION_GEMM_TM SEISMIC_TUNE_TILE_M
#define PROJECTION_GEMM_TN SEISMIC_TUNE_TILE_N
#define PROJECTION_GEMM_WM SEISMIC_TUNE_SUB_M
#define PROJECTION_GEMM_WN SEISMIC_TUNE_SUB_N
#include "gemm.glsl"

#define PROJECTION_GROUP(z) (PROJECTION_G == 1 ? 0u : uint(z))

// The stage launch of `rms` and `plain` rows: one row a workgroup, the
// prologue's rows as the weights' operands into `staged` (group 0's: the
// forms that stage whole rows run only G = 1). `staged` rows are formed by the
// entry's own stage body.
void projection_dense_stage() {
    projection_stage(projected_input(0u), gl_WorkGroupID.x, PROJECTION_STAGED, PROJECTION_K);
}

// ---------------------------------------------------------------------------
// `single`.

#ifdef PROJECTION_SINGLE
// The GEMV with the activation-row bound `maxm` (one bound a pipeline, so one
// launch a bound), group g at grid z.
void projection_single_gemv(const uint maxm) {
    packets_stage_tables();
    const uint g = PROJECTION_GROUP(gl_WorkGroupID.z);
    const projection_weights w = projected_weight(g);
    projection_gemv(PACKETS_W0, PACKETS_W0, false, SEISMIC_TUNE_ROWS, SEISMIC_TUNE_LANES, maxm,
        SEISMIC_TUNE_STAGE == 1, SEISMIC_TUNE_DECODE == 1, PROJECTION_NORM, projected_input(g), projected_output(g), w,
        w, PROJECTION_M, PROJECTION_N, PROJECTION_K, gl_WorkGroupID.x);
}

// Group g's share of the split partials: SPLIT parts of M x N each.
uint64_t projection_single_partials(uint g) {
    return PROJECTION_PARTIALS
        + uint64_t(g) * PROJECTION_SPLIT * PROJECTION_M * PROJECTION_N * 4ul;
}

// The GEMM's tiles, whole or (SPLIT > 1) over K share `part` of SPLIT into
// partials that `finalize` sums in part order.
void projection_single_gemm() {
    packets_stage_tables();
    const uint g = PROJECTION_GROUP(gl_WorkGroupID.z / PROJECTION_SPLIT);
#if PROJECTION_SPLIT == 1
    projection_gemm(PACKETS_W0, projected_rows(g), projected_output(g), projected_weight(g), PROJECTION_M, PROJECTION_N, PROJECTION_K,
        gl_WorkGroupID.y, gl_WorkGroupID.x, 0xffffffffu);
#else
    projection_gemm_part(PACKETS_W0, projected_rows(g), projection_single_partials(g), projected_weight(g), PROJECTION_M, PROJECTION_N,
        PROJECTION_K, PROJECTION_SPLIT, gl_WorkGroupID.z % PROJECTION_SPLIT, gl_WorkGroupID.y, gl_WorkGroupID.x);
#endif
}

void projection_single_finalize() {
    const uint g = PROJECTION_GROUP(gl_WorkGroupID.z);
    projection_gemm_finalize(projected_output(g), projection_single_partials(g), PROJECTION_M, PROJECTION_N,
        PROJECTION_SPLIT, gl_GlobalInvocationID.y, gl_GlobalInvocationID.x);
}

#endif

// ---------------------------------------------------------------------------
// `paired`: the gate weight is the kernel's W0, the up weight its W1.

#ifdef PROJECTION_PAIR
void projection_paired_gemv(const uint maxm) {
    packets_stage_tables();
    projection_gemv(PACKETS_W0, PACKETS_W1, true, SEISMIC_TUNE_ROWS, SEISMIC_TUNE_LANES, maxm,
        SEISMIC_TUNE_STAGE == 1, SEISMIC_TUNE_DECODE == 1, PROJECTION_NORM, projected_input(0u), projected_output(0u),
        projected_weight(0u), projected_partner(0u), PROJECTION_M, PROJECTION_N, PROJECTION_K, gl_WorkGroupID.x);
}

// A pair's tile takes TILE_N / 2 features of each weight.
void projection_paired_gemm() {
    packets_stage_tables();
    projection_gemm_paired(PACKETS_W0, PACKETS_W1, projected_rows(0u), projected_output(0u), projected_weight(0u), projected_partner(0u),
        PROJECTION_M, PROJECTION_N, PROJECTION_K, gl_WorkGroupID.y, gl_WorkGroupID.x, 0xffffffffu);
}
#endif

// ---------------------------------------------------------------------------
// `segmented`: SEISMIC_PROJECTION_SEGMENTS segments (1 to 4), segment s of
// PROJECTION_N<s> weight rows (possibly 0) projecting the same rows into its
// own columns of the results, its workgroups after the earlier segments' in
// every launch. The ABI prefix states the declaration's grouping
// (SEISMIC_PROJECTION_SEGMENTS, and SEISMIC_SEGMENT_WEIGHTS_<s>, the weights
// segment s serves); no entry states it. The entry defines, for each segment
// s,
//   PROJECTION_N<s>             its weight rows;
//   PROJECTION_WEIGHT<s>        its weight, PROJECTION_OUTPUT<s> its epilogue;
//   PROJECTION_KIND<s>          its packet kind (default PACKETS_W<s>);
//   PROJECTION_PARTNER<s>       where it serves a pair, and only there, its
//                               weight's partner (kind
//                               PROJECTION_PARTNER_KIND<s>), the two combined
//                               by its epilogue, a GEMM tile taking TILE_N / 2
//                               features of each;
//   PROJECTION_ZEROED(s)        whether its results are zeros in this call
//                               (default never): its workgroups store zeros;
// and projected_input(0u), projected_rows(0u). A GEMM launch runs one GEMM
// body over every unpaired segment, the segment's kind a workgroup-uniform
// value (several bodies in one launch defeat the NVIDIA proprietary compiler,
// `projection.glsl`), and a pair's paired body.

#if SEISMIC_PROJECTION_SEGMENTS
#include "segments.glsl"

// Segment s's GEMV workgroups: `tile` counts down through the segments'
// tiles of `per` weight rows each.
#define PROJECTION_SEGMENT_GEMV(s)                                                                      \
    {                                                                                                   \
        const uint tiles = (uint(PROJECTION_N##s) + per - 1u) / per;                                    \
        if (tile < tiles) {                                                                             \
            if (PROJECTION_ZEROED(s))                                                                   \
                projection_zero_tile(PROJECTION_OUTPUT##s, m, uint(PROJECTION_N##s), 0u, m, tile * per, per); \
            else                                                                                        \
                projection_gemv(PROJECTION_KIND##s, PROJECTION_PARTNER_KIND##s, PROJECTION_PAIRED##s,   \
                    SEISMIC_TUNE_ROWS, SEISMIC_TUNE_LANES, maxm, SEISMIC_TUNE_STAGE == 1,               \
                    SEISMIC_TUNE_DECODE == 1, PROJECTION_NORM, x, PROJECTION_OUTPUT##s,                 \
                    PROJECTION_WEIGHT##s, PROJECTION_PARTNER##s, m, uint(PROJECTION_N##s), PROJECTION_K, tile); \
            return;                                                                                     \
        }                                                                                               \
        tile -= tiles;                                                                                  \
    }

void projection_segmented_gemv(const uint maxm) {
    packets_stage_tables();
    const uint per = projection_gemv_rows(SEISMIC_TUNE_SIMDGROUPS, SEISMIC_TUNE_ROWS, SEISMIC_TUNE_LANES);
    const uint m = PROJECTION_M;
    uint tile = gl_WorkGroupID.x;
    const projection_prologue x = projected_input(0u);
    PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_GEMV)
}

// Segment s's GEMM workgroup columns: a zeroed segment's store zeros, a
// pair's run its paired body, an unpaired one's name the segment the one
// GEMM body takes.
#define PROJECTION_SEGMENT_GEMM(s)                                                                      \
    if (!found) {                                                                                       \
        const uint per = PROJECTION_PAIRED##s ? tn / 2u : tn;                                           \
        const uint tiles = (uint(PROJECTION_N##s) + per - 1u) / per;                                    \
        if (tile < tiles) {                                                                             \
            if (PROJECTION_ZEROED(s)) {                                                                 \
                projection_zero_tile(PROJECTION_OUTPUT##s, m, uint(PROJECTION_N##s), gl_WorkGroupID.y * tm, tm, \
                    tile * per, per);                                                                   \
                return;                                                                                 \
            }                                                                                           \
            if (PROJECTION_PAIRED##s) {                                                                 \
                projection_gemm_paired(PROJECTION_KIND##s, PROJECTION_PARTNER_KIND##s, x,               \
                    PROJECTION_OUTPUT##s, PROJECTION_WEIGHT##s,                                         \
                    PROJECTION_PARTNER##s, m, uint(PROJECTION_N##s), PROJECTION_K, gl_WorkGroupID.y, tile, \
                    0xffffffffu);                                                                       \
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

void projection_segmented_gemm() {
    packets_stage_tables();
    const uint tm = SEISMIC_TUNE_TILE_M, tn = SEISMIC_TUNE_TILE_N, m = PROJECTION_M;
    const projection_prologue x = projected_rows(0u);
    uint tile = gl_WorkGroupID.x;
    bool found = false;
    int kind = PROJECTION_KIND0;
    projection_weights w;
    projection_epilogue out_;
    uint rows = 0u;
    PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_GEMM)
    projection_gemm(kind, x, out_, w, m, rows, PROJECTION_K,
        gl_WorkGroupID.y, tile, 0xffffffffu);
}

// Parts (PROJECTION_PARTS): K is PROJECTION_PARTS parts of PROJECTION_K /
// PROJECTION_PARTS columns, each part's rows of their own (its own norm
// group), and the GEMV splits K by part: workgroup `tile` serves part tile /
// tiles over its row tile tile % tiles of the segments (`tiles` a part's),
// into F32 partials (output n of row m from part p at PROJECTION_PARTIALS[(m
// * PARTS + p) * outputs + n], `outputs` the segments' rows in order), which
// the combine launch adds in part order through the segments' epilogues. The
// segments are unpaired. The entry defines PROJECTION_PARTS,
// PROJECTION_PARTIALS and the functions
//   projected_part(p)           part p's rows as the GEMV forms its operands;
//   projected_part_first(p, x)  what part p's first workgroup runs after its
//                               GEMV (publishing what its prologue publishes).
#ifdef PROJECTION_PARTS
#define PROJECTION_ADD_ROWS(s) + uint(PROJECTION_N##s)
#define PROJECTION_ADD_TILES(s) + (uint(PROJECTION_N##s) + per - 1u) / per

#define PROJECTION_PART_GEMV(s)                                                                         \
    if (!found) {                                                                                       \
        const uint tiles = (uint(PROJECTION_N##s) + per - 1u) / per;                                    \
        if (tile < tiles) {                                                                             \
            projection_weights w = PROJECTION_WEIGHT##s;                                                \
            w.packet_first = part * (width / 32u);                                                      \
            projection_gemv(PROJECTION_KIND##s, PROJECTION_KIND##s, false, SEISMIC_TUNE_ROWS,           \
                SEISMIC_TUNE_LANES, maxm, SEISMIC_TUNE_STAGE == 1, SEISMIC_TUNE_DECODE == 1,            \
                PROJECTION_NORM, x,                                                                     \
                projection_store(out_over_scratch(ELEMENT_F32, PROJECTION_PARTIALS), row, 1ul, column + first), w, w,  \
                m, uint(PROJECTION_N##s), width, tile);                                                 \
            found = true;                                                                               \
        } else {                                                                                        \
            tile -= tiles;                                                                              \
            first += uint(PROJECTION_N##s);                                                             \
        }                                                                                               \
    }

void projection_parts_gemv(const uint maxm) {
    packets_stage_tables();
    const uint per = projection_gemv_rows(SEISMIC_TUNE_SIMDGROUPS, SEISMIC_TUNE_ROWS, SEISMIC_TUNE_LANES);
    const uint m = PROJECTION_M, parts = uint(PROJECTION_PARTS), width = PROJECTION_K / parts;
    const uint outputs = 0u PROJECTION_EACH_SEGMENT(PROJECTION_ADD_ROWS);
    const uint tiles = 0u PROJECTION_EACH_SEGMENT(PROJECTION_ADD_TILES);
    const uint part = gl_WorkGroupID.x / tiles, local = gl_WorkGroupID.x % tiles;
    const projection_prologue x = projected_part(part);
    const uint64_t row = uint64_t(parts) * outputs, column = uint64_t(part) * outputs;
    uint tile = local, first = 0u;
    bool found = false;
    PROJECTION_EACH_SEGMENT(PROJECTION_PART_GEMV)
    if (local == 0u)
        projected_part_first(part, x);
}

#define PROJECTION_PART_PUT(s)                                                                          \
    if (!found) {                                                                                       \
        if (n < uint(PROJECTION_N##s)) {                                                                \
            projection_put(PROJECTION_OUTPUT##s, m, n, sum);                                            \
            found = true;                                                                               \
        } else {                                                                                        \
            n -= uint(PROJECTION_N##s);                                                                 \
        }                                                                                               \
    }

// Output n of row m: its parts added in part order, through the epilogue of
// its segment.
void projection_parts_combine() {
    const uint outputs = 0u PROJECTION_EACH_SEGMENT(PROJECTION_ADD_ROWS);
    const uint parts = uint(PROJECTION_PARTS);
    const uint item = gl_GlobalInvocationID.x;
    if (item >= PROJECTION_M * outputs)
        return;
    const uint m = item / outputs;
    uint n = item % outputs;
    float sum = 0.0;
    for (uint p = 0u; p < parts; ++p)
        sum += element_f32_at(PROJECTION_PARTIALS + (uint64_t(m * parts + p) * outputs + n) * 4ul);
    bool found = false;
    PROJECTION_EACH_SEGMENT(PROJECTION_PART_PUT)
}
#endif
#endif
