// The bodies of the `dense` projection's forms (`dense.seismic`), once
// for every entry that projects: an entry's kernel states its prologue (the
// rows `x` it projects), its epilogue (`out`) and its weights, and runs the
// form its launch names here. A projection is `paired` (two weights, `w0`
// and `w1`, one activation row serves, combined by `out.store_pair`),
// `single` (one weight, `w0`, which the bodies also take as `w1`) or
// `segmented` (one to four weights, below).
//
// The includer includes `projection.h` first, and `gemv.h` where it forms a
// GEMV launch.

// The GEMV of a launch that serves COUNT (ONE, PAIR, SEVERAL) rows of `x`:
// by the packets' own recipe where they name one (`gemv_recipe`, a single
// projection's), else with TILED = 1 the row tiles where every weight's rows
// are tiled, else the row mapping (walked by packet class with TILED = 2),
// its activations staged as F32 with STAGE = 1. `n` is the rows of each
// weight, `k` their columns; the kernel provides `tile`, `shared`,
// `simdgroups`, `sg`, `lane` and its template's STAGE.
#define PROJECTION_GEMV(W0, W1, PAIRED, ROWS, LANES, TILED, COUNT, x, out, w0, w1, rows, n, k)                  \
    if constexpr (projection::gemv_recipe<W0>::value) {                                                         \
        projection::gemv_recipe_rows<W0, ROWS, LANES>(x, out, w0, rows, n, k, tile, shared, simdgroups, sg,      \
            lane);                                                                                              \
    } else if (TILED == 1 && projection::gemv_tile_row_serves(w0) &&                                            \
        (!(PAIRED) || projection::gemv_tile_row_serves(w1))) {                                                  \
        PROJECTION_FOR_##COUNT##_ROWS(rows,                                                                     \
            projection::gemv_tile_row<W0, W1, PAIRED, ROWS, MAXM, LANES>(                                       \
                x, out, w0, w1, rows, n, k, tile, shared, simdgroups, sg, lane));                               \
    } else {                                                                                                    \
        PROJECTION_FOR_##COUNT##_ROWS(rows,                                                                     \
            projection::gemv_form<W0, W1, PAIRED, ROWS, MAXM, LANES, (TILED == 2), (STAGE == 1)>(              \
                x, out, w0, w1, rows, n, k, tile, shared, simdgroups, sg, lane));                               \
    }

// The batched launch (MATRIX_FROM..16 rows of `x`, `m` of them): when
// QMATRIX_STATIC holds and k % 32 == 0, the matrix GEMV (`gemv_matrix`), else
// the batched rows (`gemv_batch_runtime`), each launch's grid matching its
// form's threadgroups.
#define PROJECTION_BATCH(W0, W1, PAIRED, BATCH_ROWS, BATCH_PARTS, QMATRIX_STATIC, x, out, w0, w1, m, n, k)      \
    if constexpr (QMATRIX_STATIC) {                                                                             \
        if ((k) % 32u == 0u) {                                                                                  \
            if constexpr (PAIRED)                                                                               \
                projection::gemv_matrix_paired<W0, W1, BATCH_PARTS>(x, out, w0, w1, m, n, k, tile, shared,      \
                    simdgroups, sg, lane);                                                                      \
            else                                                                                                \
                projection::gemv_matrix<W0, BATCH_PARTS>(x, out, w0, m, n, k, tile, shared, simdgroups, sg,     \
                    lane);                                                                                      \
        } else if constexpr (PAIRED) {                                                                          \
            projection::gemv_batch_paired_runtime<W0, W1, BATCH_ROWS>(x, out, w0, w1, m, n, k, tile, shared,    \
                simdgroups, sg, lane);                                                                          \
        } else {                                                                                                \
            projection::gemv_batch_runtime<W0, BATCH_ROWS>(x, out, w0, m, n, k, tile, shared, simdgroups, sg,   \
                lane);                                                                                          \
        }                                                                                                       \
    } else if constexpr (PAIRED) {                                                                              \
        projection::gemv_batch_paired_runtime<W0, W1, BATCH_ROWS>(x, out, w0, w1, m, n, k, tile, shared,        \
            simdgroups, sg, lane);                                                                              \
    } else {                                                                                                    \
        projection::gemv_batch_runtime<W0, BATCH_ROWS>(x, out, w0, m, n, k, tile, shared, simdgroups, sg, lane); \
    }

// ---------------------------------------------------------------------------
// The kernels of a `single` projection. An entry writes each launch's kernel
// `entry_<launch>` under its forming guard, with the launch's code parameters
// (`dense.seismic`) as its template, `PROJECTION_ARGUMENTS` and the
// launch's argument macro as its arguments, and the launch's body macro as
// its body. The entry defines
//   PROJECTION_ARGUMENTS        its kernel arguments, the projection's scratch
//                               (`partials`, `small_partials`, `fragments`,
//                               `quantized`, `row_scales`, `block_sums`,
//                               `packed`, `token_factors`, `token_scales`)
//                               included;
//   PROJECTION_GROUPS           G, the groups of its weight (1 where it has
//                               none): group g projects its own columns of
//                               the rows by its own N weight rows into its
//                               own N columns, at grid z (g * SPLIT + part in
//                               the GEMMs), into its own share of the split
//                               partials and of the tall form's fragments;
//                               the forms that stage or pack whole rows or
//                               weights run only G = 1, as group 0;
//   PROJECTION_OPERANDS(group)  group `group`'s `raw` (the rows as given),
//                               `in` (the rows every form past the GEMV
//                               reads), `out`, `w`, and `m`, `n`, `k`;
//   PROJECTION_GEMV_INPUT(x)    `x`, the rows the GEMV forms its operands
//                               from (`raw`, or `raw` under a prologue the
//                               GEMV applies as it stages them);
//   PROJECTION_INT8_SCRATCH     `scratch`, an `int8_scratch` over its weight's
//                               coefficients and biases;
//   PROJECTION_PACKING_SCRATCH  `scratch`, a `packing_scratch` over its
//                               weight's factors and scales;
//   PROJECTION_BOUNDS           `BOUNDS` and `bounds`, its packing bound;
// and `Operands`, its operand recipe.

// The group grid z names: literally 0 at G = 1, so every ungrouped address
// and split part compiles to its ungrouped form.
#ifndef PROJECTION_GROUPS
#define PROJECTION_GROUPS 1
#endif
#ifndef PROJECTION_GATHERED
#define PROJECTION_GATHERED 0
#endif
#define PROJECTION_GROUP(z) (PROJECTION_GROUPS == 1 ? 0u : uint(z))

// The arguments of the row launches (the GEMVs, the batch).
#define PROJECTION_ROW_ARGUMENTS                                                                                \
    threadgroup uchar *shared [[threadgroup(0)]],                                                               \
    uint3 position [[threadgroup_position_in_grid]],                                                            \
    uint simdgroups [[simdgroups_per_threadgroup]],                                                             \
    uint sg [[simdgroup_index_in_threadgroup]],                                                                 \
    uint lane [[thread_index_in_simdgroup]]

// The arguments of the tiled launches (the GEMMs, TALL, INT8, unpacked).
#define PROJECTION_TILE_ARGUMENTS                                                                               \
    uint3 tile [[threadgroup_position_in_grid]],                                                                \
    uint sg [[simdgroup_index_in_threadgroup]],                                                                 \
    uint lane [[thread_index_in_simdgroup]]

// The arguments of the elementwise launches (the reductions, the relayout,
// the INT8 coefficients).
#define PROJECTION_ITEM_ARGUMENTS uint3 position [[thread_position_in_grid]]

// The arguments of the threadgroup launches (the INT8 quantization, the PACK
// operand, coefficients and tiles).
#define PROJECTION_THREADGROUP_ARGUMENTS                                                                        \
    uint index [[threadgroup_position_in_grid]],                                                                \
    uint thread_index [[thread_index_in_threadgroup]],                                                          \
    uint sg [[simdgroup_index_in_threadgroup]],                                                                 \
    uint lane [[thread_index_in_simdgroup]]

// The GEMV of a launch that serves COUNT (ONE, PAIR, SEVERAL) rows in tile
// form TILED.
#define PROJECTION_SINGLE_GEMV(COUNT, TILED)                                                                    \
    const uint tile = position.x;                                                                               \
    PROJECTION_OPERANDS(PROJECTION_GROUP(position.z));                                                          \
    PROJECTION_GEMV_INPUT(input);                                                                               \
    PROJECTION_OPERANDS_SHARED(globals, NVFP4);                                                                 \
    const auto x = Operands::rows(input, m, k, globals, simdgroups, sg, lane);                                  \
    PROJECTION_GEMV(packets::W0, packets::W0, false, ROWS, LANES, TILED, COUNT, x, out, w, w, m, n, k)

#define PROJECTION_SINGLE_BATCH                                                                                 \
    const uint tile = position.x;                                                                               \
    PROJECTION_OPERANDS(PROJECTION_GROUP(position.z));                                                          \
    constexpr bool QMATRIX_STATIC = (PROJECTION_GROUPS == 1) && (PROJECTION_GATHERED == 0)                      \
        && projection::matrix_form<packets::W0>::available;                                                     \
    PROJECTION_BATCH(packets::W0, packets::W0, false, BATCH_ROWS, BATCH_PARTS, QMATRIX_STATIC, in, out, w, w,   \
        m, n, k)

// The batched launch over the rows each threadgroup forms as the GEMV does
// (`PROJECTION_GEMV_INPUT`), for an entry that stages none to 16 rows.
#define PROJECTION_SINGLE_BATCH_FORMED                                                                          \
    const uint tile = position.x;                                                                               \
    PROJECTION_OPERANDS(PROJECTION_GROUP(position.z));                                                          \
    PROJECTION_GEMV_INPUT(formed);                                                                              \
    PROJECTION_BATCH(packets::W0, packets::W0, false, BATCH_ROWS, BATCH_PARTS, false, formed, out, w, w, m, n,  \
        k)

// A group's share of a split's partials: SPLIT parts of m x n each.
#define PROJECTION_PARTIALS(PARTIALS, group, SPLIT) ((PARTIALS) + ulong(group) * (SPLIT) * m * n)

// The staged tiles over SPLIT shares of the columns (whole at SPLIT = 1),
// grid z group * SPLIT + part.
#define PROJECTION_SINGLE_TILES(TM, TN, SPLIT, PARTIALS)                                                        \
    PROJECTION_GEMM_SHARED(shared, TM, TN);                                                                     \
    const uint group = PROJECTION_GROUP(tile.z / (SPLIT));                                                      \
    PROJECTION_OPERANDS(group);                                                                                 \
    if (SPLIT == 1)                                                                                             \
        projection::gemm<packets::W0, TM, TN>(in, out, w, m, n, k, tile.y, tile.x, shared, sg, lane);           \
    else                                                                                                        \
        projection::gemm_part<packets::W0, TM, TN>(in, PROJECTION_PARTIALS(PARTIALS, group, SPLIT), w, m, n, k, \
            SPLIT, PROJECTION_GROUPS == 1 ? tile.z : tile.z % (SPLIT), tile.y, tile.x, shared, sg, lane)

// 17..64 rows: the fixed small-row tile and split, then their sum.
#define PROJECTION_SINGLE_GEMM_SMALL                                                                            \
    PROJECTION_SINGLE_TILES(projection::small_tile_m, projection::small_tile_n, projection::small_split,        \
        small_partials)

#define PROJECTION_SINGLE_REDUCE_SMALL                                                                          \
    const uint group = PROJECTION_GROUP(position.z);                                                            \
    PROJECTION_OPERANDS(group);                                                                                 \
    projection::gemm_reduce(out, PROJECTION_PARTIALS(small_partials, group, projection::small_split), m, n,     \
        projection::small_split, position.x)

// The tuned tiles over SPLIT (the entry's runtime `split`) shares, then their
// sum.
#define PROJECTION_SINGLE_GEMM(SPLIT) PROJECTION_SINGLE_TILES(TILE_M, TILE_N, SPLIT, partials)

// The staged tiles whole, for an entry whose GEMM never splits (SPLIT 1).
#define PROJECTION_SINGLE_GEMM_WHOLE                                                                            \
    PROJECTION_GEMM_SHARED(shared, TILE_M, TILE_N);                                                             \
    PROJECTION_OPERANDS(PROJECTION_GROUP(tile.z));                                                              \
    projection::gemm<packets::W0, TILE_M, TILE_N>(in, out, w, m, n, k, tile.y, tile.x, shared, sg, lane)

#define PROJECTION_SINGLE_REDUCE(SPLIT)                                                                         \
    const uint group = PROJECTION_GROUP(position.z);                                                            \
    PROJECTION_OPERANDS(group);                                                                                 \
    projection::gemm_reduce(out, PROJECTION_PARTIALS(partials, group, SPLIT), m, n, SPLIT, position.x)

// The TALL form past 64 rows: the rows in the tall GEMM's order, then its
// tiles. A group's rows lie in its own share of the fragments, ceil(m / 64)
// * 64 rows of k halves each.
#define PROJECTION_GROUP_FRAGMENTS(group) (fragments + ulong(group) * ((m + 63u) / 64u) * 64u * k * 2u)

#define PROJECTION_SINGLE_RELAYOUT                                                                              \
    const uint group = PROJECTION_GROUP(position.z);                                                            \
    PROJECTION_OPERANDS(group);                                                                                 \
    projection::tall_relayout(in, PROJECTION_GROUP_FRAGMENTS(group), m, k, position.x)

#define PROJECTION_SINGLE_TALL                                                                                  \
    PROJECTION_GEMM_TALL_SHARED(shared, TALL_K);                                                                \
    const uint group = PROJECTION_GROUP(tile.z);                                                                \
    PROJECTION_OPERANDS(group);                                                                                 \
    projection::gemm_tall<packets::W0, TALL_M, TALL_K, STAGERS>(                                                \
        projection::tall_operand(in, PROJECTION_GROUP_FRAGMENTS(group)), out, w, m, n, k, tile.y, tile.x, shared, \
        sg, lane)

// The INT8 form past 64 rows: the rows quantized per (row, 32 columns), the
// weight's block scales and biases, then the int8 tiles.
#define PROJECTION_SINGLE_QUANTIZE                                                                              \
    if constexpr (!projection::int8_path<packets::W0>())                                                        \
        return;                                                                                                 \
    PROJECTION_OPERANDS(0u);                                                                                    \
    PROJECTION_INT8_SCRATCH;                                                                                    \
    projection::int8_quantize<packets::int8<packets::W0>::interleaved>(in, scratch, m, k, index,                \
        thread_index, lane, 1.0f)

#define PROJECTION_SINGLE_COEFFICIENTS                                                                          \
    if constexpr (!projection::int8_path<packets::W0>())                                                        \
        return;                                                                                                 \
    PROJECTION_OPERANDS(0u);                                                                                    \
    PROJECTION_INT8_SCRATCH;                                                                                    \
    projection::int8_coefficients(w, scratch, n, k, position.x)

// The default form's tile only where the launch takes that form: threadgroup
// memory a kernel does not use still slows its products.
#define PROJECTION_SINGLE_INT8                                                                                  \
    constexpr bool PATH = projection::int8_path<packets::W0>();                                                 \
    threadgroup float4 shared_words[PATH ? 1 : projection::gemm_tile<64, 64>::bytes / 16];                      \
    threadgroup uchar *shared = reinterpret_cast<threadgroup uchar *>(shared_words);                            \
    PROJECTION_OPERANDS(0u);                                                                                    \
    PROJECTION_INT8_SCRATCH;                                                                                    \
    projection::gemm_int8<packets::W0, INT8_TILE, PATH>(in, out, w, scratch, m, n, k, tile.x, tile.y, shared,   \
        sg, lane)

// The PACK form past 64 rows: the rows as integer codes, two rows
// packed per operand element (RESIDUE: the operand's residue width for
// weights with residue bits), the weight's block scales and biases, then the
// packed tiles over the rows padded to two operand tiles; on a device without
// the packed path, the staged tiles of the default form, with its results.
#define PROJECTION_SINGLE_PACK                                                                                  \
    if constexpr (!projection::packing_path<packets::W0>())                                                     \
        return;                                                                                                 \
    threadgroup float peaks[projection::packing_peaks()];                                                       \
    PROJECTION_OPERANDS(0u);                                                                                    \
    PROJECTION_PACKING_SCRATCH;                                                                                 \
    PROJECTION_BOUNDS;                                                                                          \
    typedef projection::packing_codes<packets::W0> coded;                                                       \
    projection::packing_operand<coded::centre, coded::folds, coded::sums, (coded::residue != 0 ? RESIDUE : 0),   \
        BOUNDS>(in, scratch, bounds, m, k, index, peaks, thread_index, sg, lane)

#define PROJECTION_SINGLE_PACK_COEFFICIENTS                                                                     \
    if constexpr (!projection::packing_path<packets::W0>())                                                     \
        return;                                                                                                 \
    threadgroup float peaks[8];                                                                                 \
    PROJECTION_OPERANDS(0u);                                                                                    \
    PROJECTION_PACKING_SCRATCH;                                                                                 \
    projection::packing_tile_coefficients(w, scratch, k, index, peaks, thread_index, sg, lane)

#define PROJECTION_SINGLE_PACKED                                                                                \
    if constexpr (!projection::packing_path<packets::W0>())                                                     \
        return;                                                                                                 \
    PROJECTION_OPERANDS(0u);                                                                                    \
    PROJECTION_PACKING_SCRATCH;                                                                                 \
    projection::gemm_packed<packets::W0, RESIDUE, PACK_TOKENS, WEIGHTS_AHEAD>(out, w, scratch, m,               \
        projection::packing_padded(m, 2u), n, k, index * projection::packing_simdgroups + sg, lane)

#define PROJECTION_SINGLE_UNPACKED                                                                              \
    if constexpr (projection::packing_path<packets::W0>())                                                      \
        return;                                                                                                 \
    PROJECTION_SINGLE_TILES(64, 64, 1, partials)

// ---------------------------------------------------------------------------
// The kernels of a `paired` projection: a gate and an up weight, `w0` and
// `w1`, of the same n rows, one activation row serves, combined by
// `out.store_pair`. An entry writes its kernels as for `single`, the
// `PROJECTION_PAIRED_` bodies, and defines
//   PROJECTION_ARGUMENTS        its kernel arguments, the projection's scratch
//                               (`quantized`, `row_scales`, `block_sums`,
//                               `packed`, `token_factors`, `token_scales`)
//                               included;
//   PROJECTION_OPERANDS         `raw` (the rows as given), `in` (the staged
//                               rows every form past the GEMV reads), `out`,
//                               `w0`, `w1`, and `m`, `n`, `k`;
//   PROJECTION_GEMV_INPUT(x)    as for `single`;
//   PROJECTION_FRAGMENTS        the staged rows in the tall GEMM's order;
//   PROJECTION_INT8_SCRATCH     `scratch0` and `scratch1`, each weight's
//                               `int8_scratch` over the one quantized operand;
//   PROJECTION_PACKING_SCRATCH  `scratch0` and `scratch1`, each weight's
//                               `packing_scratch` over the one packed operand;
//   PROJECTION_BOUNDED          the weights with a packing bound (bit 0 the
//                               gate's, bit 1 the up's), and
//   PROJECTION_BOUND_TABLES     their bounds' tables, gate first;
// and `Operands`, its operand recipe.

#define PROJECTION_PAIRED_GEMV(COUNT, TILED)                                                                    \
    const uint tile = position.x;                                                                               \
    PROJECTION_OPERANDS;                                                                                        \
    PROJECTION_GEMV_INPUT(input);                                                                               \
    PROJECTION_OPERANDS_SHARED(globals, NVFP4);                                                                 \
    const auto x = Operands::rows(input, m, k, globals, simdgroups, sg, lane);                                  \
    PROJECTION_GEMV(packets::W0, packets::W1, true, ROWS, LANES, TILED, COUNT, x, out, w0, w1, m, n, k)

#define PROJECTION_PAIRED_BATCH                                                                                 \
    const uint tile = position.x;                                                                               \
    PROJECTION_OPERANDS;                                                                                        \
    constexpr bool QMATRIX_STATIC = (PROJECTION_GROUPS == 1) && (PROJECTION_GATHERED == 0)                      \
        && projection::matrix_form<packets::W0>::available && projection::matrix_form<packets::W1>::available;  \
    PROJECTION_BATCH(packets::W0, packets::W1, true, BATCH_ROWS, BATCH_PARTS, QMATRIX_STATIC, in, out, w0, w1,  \
        m, n, k)

// The staged paired tiles, TM x TN.
#define PROJECTION_PAIRED_TILES(TM, TN)                                                                         \
    PROJECTION_GEMM_SHARED(shared, TM, TN);                                                                     \
    PROJECTION_OPERANDS;                                                                                        \
    projection::gemm_paired<packets::W0, packets::W1, TM, TN>(in, out, w0, w1, m, n, k, tile.y, tile.x, shared, \
        sg, lane)

// 17..64 rows: the fixed small-row tile.
#define PROJECTION_PAIRED_GEMM_SMALL PROJECTION_PAIRED_TILES(projection::small_tile_m, projection::small_tile_n)

// The tuned tiles past 64 rows.
#define PROJECTION_PAIRED_GEMM PROJECTION_PAIRED_TILES(TILE_M, TILE_N)

// The TALL form past 64 rows: the tall GEMM's tiles over the rows staged in
// its order.
#define PROJECTION_PAIRED_TALL                                                                                  \
    PROJECTION_GEMM_TALL_SHARED(shared, TALL_K);                                                                \
    PROJECTION_OPERANDS;                                                                                        \
    projection::gemm_tall_paired<packets::W0, packets::W1, TALL_M, TALL_K, STAGERS>(                            \
        projection::tall_operand(in, PROJECTION_FRAGMENTS), out, w0, w1, m, n, k, tile.y, tile.x, shared, sg,   \
        lane)

// The INT8 form past 64 rows: the staged rows quantized per (row, 32
// columns) once for both weights, both weights' block scales and biases,
// then the paired int8 tiles, where the weights take the int8 products
// (`int8_path`).
#define PROJECTION_PAIRED_INT8_PATH projection::int8_path<packets::W0, packets::W1>()

#define PROJECTION_PAIRED_QUANTIZE                                                                              \
    if constexpr (!PROJECTION_PAIRED_INT8_PATH)                                                                 \
        return;                                                                                                 \
    PROJECTION_OPERANDS;                                                                                        \
    PROJECTION_INT8_SCRATCH;                                                                                    \
    projection::int8_quantize<packets::int8<packets::W0>::interleaved>(in, scratch0, m, k, index,               \
        thread_index, lane, 1.0f)

#define PROJECTION_PAIRED_COEFFICIENTS                                                                          \
    if constexpr (!PROJECTION_PAIRED_INT8_PATH)                                                                 \
        return;                                                                                                 \
    PROJECTION_OPERANDS;                                                                                        \
    PROJECTION_INT8_SCRATCH;                                                                                    \
    projection::int8_coefficients(w0, scratch0, n, k, position.x);                                              \
    projection::int8_coefficients(w1, scratch1, n, k, position.x)

// The default form's tile only where the launch takes that form:
// threadgroup memory a kernel does not use still slows its products.
#define PROJECTION_PAIRED_INT8                                                                                  \
    constexpr bool PATH = PROJECTION_PAIRED_INT8_PATH;                                                          \
    threadgroup float exchange[PATH ? 2 * 32 * 32 : 1];                                                         \
    threadgroup float4 shared_words[PATH ? 1 : projection::gemm_tile<64, 64>::bytes / 16];                      \
    threadgroup uchar *shared = reinterpret_cast<threadgroup uchar *>(shared_words);                            \
    PROJECTION_OPERANDS;                                                                                        \
    PROJECTION_INT8_SCRATCH;                                                                                    \
    projection::gemm_int8_paired<packets::W0, packets::W1, PATH, 64>(in, out, w0, w1, scratch0, scratch1, m, n, k, \
        tile.y, tile.x, shared, exchange, sg, lane)

// The PACK form past 64 rows: the staged rows as integer codes, two
// rows packed per operand element, once for both weights (RESIDUE: the operand's
// residue width for weights with residue bits), both weights' block scales
// and biases, then the paired packed tiles; on a device without the packed
// path, the staged tiles of the default form, with its results. PACK = 1
// names them only for a gate and an up weight one operand serves, of one
// layout (`PACK_FOLDS`, `PACK_RESIDUE`, `PACK_SUMS`), which the projection's
// `where` requires.
#define PROJECTION_PAIRED_PACKING_PATH projection::packing_path<packets::W0, packets::W1>()
#define PROJECTION_PAIRED_PACKING_ADMITTED                                                                      \
    static_assert(projection::packing_codes<packets::W0>::folds == projection::packing_codes<packets::W1>::folds \
            && (projection::packing_codes<packets::W0>::residue != 0)                                           \
                == (projection::packing_codes<packets::W1>::residue != 0)                                       \
            && projection::packing_codes<packets::W0>::sums == projection::packing_codes<packets::W1>::sums,    \
        "PACK = 1 names the packed GEMM of weights one operand serves")

// The operand serves both weights: their bounds, where both have one.
#define PROJECTION_PAIRED_PACK                                                                                  \
    PROJECTION_PAIRED_PACKING_ADMITTED;                                                                         \
    if constexpr (!PROJECTION_PAIRED_PACKING_PATH)                                                              \
        return;                                                                                                 \
    threadgroup float peaks[projection::packing_peaks()];                                                       \
    PROJECTION_OPERANDS;                                                                                        \
    PROJECTION_PACKING_SCRATCH;                                                                                 \
    typedef projection::packing_shared<RESIDUE, packets::W0, 1, packets::W1, 1> coded;                          \
    constexpr uint BOUNDS = projection::packing_bounded<3u, PROJECTION_BOUNDED>::value;                         \
    const projection::packing_bounds<BOUNDS> bounds{PROJECTION_BOUND_TABLES};                                   \
    projection::packing_operand<coded::centre, coded::folds, coded::sums, coded::residue, BOUNDS>(              \
        in, scratch0, bounds, m, k, index, peaks, thread_index, sg, lane)

#define PROJECTION_PAIRED_PACK_COEFFICIENTS                                                                     \
    PROJECTION_PAIRED_PACKING_ADMITTED;                                                                         \
    if constexpr (!PROJECTION_PAIRED_PACKING_PATH)                                                              \
        return;                                                                                                 \
    threadgroup float peaks0[8];                                                                                \
    threadgroup float peaks1[8];                                                                                \
    PROJECTION_OPERANDS;                                                                                        \
    PROJECTION_PACKING_SCRATCH;                                                                                 \
    projection::packing_tile_coefficients(w0, scratch0, k, index, peaks0, thread_index, sg, lane);              \
    projection::packing_tile_coefficients(w1, scratch1, k, index, peaks1, thread_index, sg, lane)

// The operand's rows padded to the paired tile's one operand tile.
#define PROJECTION_PAIRED_PACKED                                                                                \
    PROJECTION_PAIRED_PACKING_ADMITTED;                                                                         \
    if constexpr (!PROJECTION_PAIRED_PACKING_PATH)                                                              \
        return;                                                                                                 \
    threadgroup float exchange[projection::packing_simdgroups / 2 * projection::packing_exchange()];            \
    PROJECTION_OPERANDS;                                                                                        \
    PROJECTION_PACKING_SCRATCH;                                                                                 \
    projection::gemm_packed_paired<packets::W0, packets::W1, RESIDUE, PACK_TOKENS, WEIGHTS_AHEAD>(              \
        out, w0, w1, scratch0, scratch1, m, projection::packing_padded(m, 1u), n, k, index,                     \
        projection::packing_simdgroups, exchange, sg, lane)

#define PROJECTION_PAIRED_UNPACKED                                                                              \
    PROJECTION_PAIRED_PACKING_ADMITTED;                                                                         \
    if constexpr (PROJECTION_PAIRED_PACKING_PATH)                                                               \
        return;                                                                                                 \
    PROJECTION_PAIRED_TILES(64, 64)

// ---------------------------------------------------------------------------
// The kernels of a `segmented` projection: SEISMIC_PROJECTION_SEGMENTS
// segments (1 to 4), segment s of PROJECTION_N<s> weight rows (a constant,
// possibly 0) projecting the same rows into its own columns of the results,
// its threadgroups after the earlier segments' in every launch. The ABI
// prefix states the declaration's grouping (`SEISMIC_PROJECTION_SEGMENTS`,
// and `SEISMIC_SEGMENT_WEIGHTS_<s>`, the weights segment s serves); no
// kernel states it. An entry writes its kernels as for `single`, the
// `PROJECTION_SEGMENTED_` bodies, and defines
//   PROJECTION_ARGUMENTS        its kernel arguments, the projection's scratch
//                               (`packed`, `token_factors`, `token_scales`,
//                               `factors`, `scales`) included;
//   PROJECTION_N<s>             segment s's weight rows;
//   PROJECTION_PACKET_<s>       segment s's weight packet (default
//                               `packets::W<s>`: the entry names it where its
//                               kernel's weights are not one a segment);
//   PROJECTION_PARTNER_<s>      where segment s serves a pair of weights, and
//                               only there, the partner of `w<s>` (packet
//                               `PROJECTION_PARTNER_PACKET_<s>`), the two
//                               combined by `out<s>.store_pair`. A pair's tile
//                               forms take half the features of a single
//                               weight's, and the PACK form is not offered for
//                               a projection with a pair (`SERVED`);
//   PROJECTION_OPERANDS         `raw` (the rows as given), `in` (the staged
//                               rows: every form past the GEMV, and the
//                               batched launch's matrix GEMV), `m`, `k`, and
//                               segment s's `w<s>` and `out<s>`;
//   PROJECTION_GEMV_INPUT(x)    as for `single`, the rows of the GEMVs and of
//                               the batched rows;
//   PROJECTION_FRAGMENTS        the staged rows in the tall GEMM's order;
//   PROJECTION_ZEROED(s)        whether segment s's results are zeros in this
//                               call: its threadgroups store zeros instead;
//   PROJECTION_BOUNDED          the segments with a packing bound (bit s), and
//   PROJECTION_BOUND_TABLES     their bounds' tables, in segment order;
//   PROJECTION_COLUMNS(s)       whether `out<s>` is a column epilogue
//                               (`store_column`) in the batched launch;
//   PROJECTION_GEMV_PROLOGUE(s) what a GEMV threadgroup of segment s runs
//                               first (no barrier: the GEMV's own publishes
//                               it to the epilogue);
//   PROJECTION_BATCH_PROLOGUE(s) what a batched threadgroup of segment s runs
//                               first, its own barrier included;
//   PROJECTION_GEMV_EPILOGUE(s, first, count)  what a GEMV threadgroup of
//                               segment s runs after storing its columns.

// A partner exactly where the declaration groups a pair.
#if SEISMIC_PROJECTION_SEGMENTS
#if SEISMIC_SEGMENT_WEIGHTS_0 == 2 && !defined(PROJECTION_PARTNER_0)
#error "segment 0 serves a pair: its kernel names the partner (PROJECTION_PARTNER_0)"
#elif SEISMIC_SEGMENT_WEIGHTS_0 != 2 && defined(PROJECTION_PARTNER_0)
#error "segment 0 serves one weight: its kernel names no partner"
#endif
#if SEISMIC_SEGMENT_WEIGHTS_1 == 2 && !defined(PROJECTION_PARTNER_1)
#error "segment 1 serves a pair: its kernel names the partner (PROJECTION_PARTNER_1)"
#elif SEISMIC_SEGMENT_WEIGHTS_1 != 2 && defined(PROJECTION_PARTNER_1)
#error "segment 1 serves one weight: its kernel names no partner"
#endif
#if SEISMIC_SEGMENT_WEIGHTS_2 == 2 && !defined(PROJECTION_PARTNER_2)
#error "segment 2 serves a pair: its kernel names the partner (PROJECTION_PARTNER_2)"
#elif SEISMIC_SEGMENT_WEIGHTS_2 != 2 && defined(PROJECTION_PARTNER_2)
#error "segment 2 serves one weight: its kernel names no partner"
#endif
#if SEISMIC_SEGMENT_WEIGHTS_3 == 2 && !defined(PROJECTION_PARTNER_3)
#error "segment 3 serves a pair: its kernel names the partner (PROJECTION_PARTNER_3)"
#elif SEISMIC_SEGMENT_WEIGHTS_3 != 2 && defined(PROJECTION_PARTNER_3)
#error "segment 3 serves one weight: its kernel names no partner"
#endif
#if SEISMIC_SEGMENT_WEIGHTS_4 == 2 && !defined(PROJECTION_PARTNER_4)
#error "segment 4 serves a pair: its kernel names the partner (PROJECTION_PARTNER_4)"
#elif SEISMIC_SEGMENT_WEIGHTS_4 != 2 && defined(PROJECTION_PARTNER_4)
#error "segment 4 serves one weight: its kernel names no partner"
#endif
#if SEISMIC_SEGMENT_WEIGHTS_5 == 2 && !defined(PROJECTION_PARTNER_5)
#error "segment 5 serves a pair: its kernel names the partner (PROJECTION_PARTNER_5)"
#elif SEISMIC_SEGMENT_WEIGHTS_5 != 2 && defined(PROJECTION_PARTNER_5)
#error "segment 5 serves one weight: its kernel names no partner"
#endif
#if SEISMIC_SEGMENT_WEIGHTS_6 == 2 && !defined(PROJECTION_PARTNER_6)
#error "segment 6 serves a pair: its kernel names the partner (PROJECTION_PARTNER_6)"
#elif SEISMIC_SEGMENT_WEIGHTS_6 != 2 && defined(PROJECTION_PARTNER_6)
#error "segment 6 serves one weight: its kernel names no partner"
#endif
#if SEISMIC_SEGMENT_WEIGHTS_7 == 2 && !defined(PROJECTION_PARTNER_7)
#error "segment 7 serves a pair: its kernel names the partner (PROJECTION_PARTNER_7)"
#elif SEISMIC_SEGMENT_WEIGHTS_7 != 2 && defined(PROJECTION_PARTNER_7)
#error "segment 7 serves one weight: its kernel names no partner"
#endif
#endif
#ifndef PROJECTION_PACKET_0
#define PROJECTION_PACKET_0 packets::W0
#endif
#ifndef PROJECTION_PARTNER_0
#define PROJECTION_PARTNER_0 w0
#endif
#ifndef PROJECTION_PARTNER_PACKET_0
#define PROJECTION_PARTNER_PACKET_0 PROJECTION_PACKET_0
#endif
#ifndef PROJECTION_PACKET_1
#define PROJECTION_PACKET_1 packets::W1
#endif
#ifndef PROJECTION_PARTNER_1
#define PROJECTION_PARTNER_1 w1
#endif
#ifndef PROJECTION_PARTNER_PACKET_1
#define PROJECTION_PARTNER_PACKET_1 PROJECTION_PACKET_1
#endif
#ifndef PROJECTION_PACKET_2
#define PROJECTION_PACKET_2 packets::W2
#endif
#ifndef PROJECTION_PARTNER_2
#define PROJECTION_PARTNER_2 w2
#endif
#ifndef PROJECTION_PARTNER_PACKET_2
#define PROJECTION_PARTNER_PACKET_2 PROJECTION_PACKET_2
#endif
#ifndef PROJECTION_PACKET_3
#define PROJECTION_PACKET_3 packets::W3
#endif
#ifndef PROJECTION_PARTNER_3
#define PROJECTION_PARTNER_3 w3
#endif
#ifndef PROJECTION_PARTNER_PACKET_3
#define PROJECTION_PARTNER_PACKET_3 PROJECTION_PACKET_3
#endif
#ifndef PROJECTION_PACKET_4
#define PROJECTION_PACKET_4 packets::W4
#endif
#ifndef PROJECTION_PARTNER_4
#define PROJECTION_PARTNER_4 w4
#endif
#ifndef PROJECTION_PARTNER_PACKET_4
#define PROJECTION_PARTNER_PACKET_4 PROJECTION_PACKET_4
#endif
#ifndef PROJECTION_PACKET_5
#define PROJECTION_PACKET_5 packets::W5
#endif
#ifndef PROJECTION_PARTNER_5
#define PROJECTION_PARTNER_5 w5
#endif
#ifndef PROJECTION_PARTNER_PACKET_5
#define PROJECTION_PARTNER_PACKET_5 PROJECTION_PACKET_5
#endif
#ifndef PROJECTION_PACKET_6
#define PROJECTION_PACKET_6 packets::W6
#endif
#ifndef PROJECTION_PARTNER_6
#define PROJECTION_PARTNER_6 w6
#endif
#ifndef PROJECTION_PARTNER_PACKET_6
#define PROJECTION_PARTNER_PACKET_6 PROJECTION_PACKET_6
#endif
#ifndef PROJECTION_PACKET_7
#define PROJECTION_PACKET_7 packets::W7
#endif
#ifndef PROJECTION_PARTNER_7
#define PROJECTION_PARTNER_7 w7
#endif
#ifndef PROJECTION_PARTNER_PACKET_7
#define PROJECTION_PARTNER_PACKET_7 PROJECTION_PACKET_7
#endif
#define PROJECTION_PACKET(s) PROJECTION_PACKET_##s
#define PROJECTION_PAIRED(s) (SEISMIC_SEGMENT_WEIGHTS_##s == 2)
#define PROJECTION_PARTNER(s) PROJECTION_PARTNER_##s
#define PROJECTION_PARTNER_PACKET(s) PROJECTION_PARTNER_PACKET_##s
#ifndef PROJECTION_ZEROED
#define PROJECTION_ZEROED(s) false
#endif

namespace projection {

// Zeros into `out`'s rows row0 .. row0 + rows and columns column0 ..
// column0 + columns, within its m x n.
template <typename Out>
inline void zero_tile(thread const Out &out, uint m, uint n, uint row0, uint rows, uint column0, uint columns,
    uint thread_index, uint threads) {
    for (uint item = thread_index; item < rows * columns; item += threads) {
        const uint row = row0 + item / columns, column = column0 + item % columns;
        if (row < m && column < n)
            out.store(row, column, 0.0f);
    }
}

// A zeroed segment's tile: a pair's results are never zeros.
template <bool PAIRED, typename Out>
inline void zero_segment(thread const Out &out, uint m, uint n, uint row0, uint rows, uint column0, uint columns,
    uint thread_index, uint threads) {
    if constexpr (!PAIRED)
        zero_tile(out, m, n, row0, rows, column0, columns, thread_index, threads);
}

// A segment's batched threadgroup `tile`: the matrix GEMV where MATRIX takes
// it and it serves the weights (`matrix_form`, over whole 32-column groups),
// else the batched rows, whose threadgroups past the
// segment's rows (the matrix GEMV's grid) return. A pair takes `w` and `v`
// together.
template <typename W, typename V, bool PAIRED, uint BATCH_ROWS, uint PARTS, bool QMATRIX, bool COLUMNS, typename In,
    typename Out>
inline void segment_batch(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const Weights<V> &v, uint m, uint n, uint k, uint tile, threadgroup uchar *shared, uint simdgroups,
    uint sg, uint lane) {
    if constexpr (QMATRIX && matrix_form<W>::available && (!PAIRED || matrix_form<V>::available)) {
        if (k % 32u == 0) {
            if constexpr (PAIRED)
                gemv_matrix_paired<W, V, PARTS>(in, out, w, v, m, n, k, tile, shared, simdgroups, sg, lane);
            else if constexpr (COLUMNS)
                gemv_matrix_columns<W, PARTS>(in, out, w, m, n, k, tile, shared, simdgroups, sg, lane);
            else
                gemv_matrix<W, PARTS>(in, out, w, m, n, k, tile, shared, simdgroups, sg, lane);
            return;
        }
    }
    if (tile * simdgroups * BATCH_ROWS * 8u >= n)
        return;
    if constexpr (PAIRED)
        gemv_batch_paired_runtime<W, V, BATCH_ROWS>(in, out, w, v, m, n, k, tile, shared, simdgroups, sg, lane);
    else if constexpr (COLUMNS)
        gemv_batch_columns_runtime<W, BATCH_ROWS>(in, out, w, m, n, k, tile, shared, simdgroups, sg, lane);
    else
        gemv_batch_runtime<W, BATCH_ROWS>(in, out, w, m, n, k, tile, shared, simdgroups, sg, lane);
}

// A segment's staged tile: a pair's takes both weights.
template <typename W, typename V, bool PAIRED, uint TM, uint TN, typename In, typename Out>
inline void segment_gemm(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const Weights<V> &v, uint m, uint n, uint k, uint tile_m, uint tile_n, threadgroup uchar *shared,
    uint sg, uint lane) {
    if constexpr (PAIRED)
        gemm_paired<W, V, TM, TN>(in, out, w, v, m, n, k, tile_m, tile_n, shared, sg, lane);
    else
        gemm<W, TM, TN>(in, out, w, m, n, k, tile_m, tile_n, shared, sg, lane);
}

// A segment's tall tile.
template <typename W, typename V, bool PAIRED, uint TALL_M, uint TALL_K, uint STAGERS, typename In, typename Out>
inline void segment_tall(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const Weights<V> &v, uint m, uint n, uint k, uint tile_m, uint tile_n, threadgroup uchar *shared,
    uint sg, uint lane) {
    if constexpr (PAIRED)
        gemm_tall_paired<W, V, TALL_M, TALL_K, STAGERS>(in, out, w, v, m, n, k, tile_m, tile_n, shared, sg, lane);
    else
        gemm_tall<W, TALL_M, TALL_K, STAGERS>(in, out, w, m, n, k, tile_m, tile_n, shared, sg, lane);
}

} // namespace projection

// The head of a chain of segment blocks: each block is an `else if`, so the
// segments are exclusive to the compiler as well as by construction (four
// independent `if`s cost the tall GEMM 0.2% at 256 and 512 rows).
#define PROJECTION_SEGMENT_CHAIN if (false) {}

// X(s, ...) for every segment s, in order.
#define PROJECTION_EACH_1(X, ...) X(0, __VA_ARGS__)
#define PROJECTION_EACH_2(X, ...) PROJECTION_EACH_1(X, __VA_ARGS__) X(1, __VA_ARGS__)
#define PROJECTION_EACH_3(X, ...) PROJECTION_EACH_2(X, __VA_ARGS__) X(2, __VA_ARGS__)
#define PROJECTION_EACH_4(X, ...) PROJECTION_EACH_3(X, __VA_ARGS__) X(3, __VA_ARGS__)
#define PROJECTION_EACH_5(X, ...) PROJECTION_EACH_4(X, __VA_ARGS__) X(4, __VA_ARGS__)
#define PROJECTION_EACH_6(X, ...) PROJECTION_EACH_5(X, __VA_ARGS__) X(5, __VA_ARGS__)
#define PROJECTION_EACH_7(X, ...) PROJECTION_EACH_6(X, __VA_ARGS__) X(6, __VA_ARGS__)
#define PROJECTION_EACH_8(X, ...) PROJECTION_EACH_7(X, __VA_ARGS__) X(7, __VA_ARGS__)
#define PROJECTION_EACH_OF(COUNT, X, ...) PROJECTION_EACH_##COUNT(X, __VA_ARGS__)
#define PROJECTION_EACH_EXPANDED(COUNT, X, ...) PROJECTION_EACH_OF(COUNT, X, __VA_ARGS__)
#define PROJECTION_EACH_SEGMENT(X, ...) PROJECTION_EACH_EXPANDED(SEISMIC_PROJECTION_SEGMENTS, X, __VA_ARGS__)

// The weight rows before segment s: its offset in the factor tables.
#define PROJECTION_FIRST_0 0u
#define PROJECTION_FIRST_1 uint(PROJECTION_N0)
#define PROJECTION_FIRST_2 uint(PROJECTION_N0 + PROJECTION_N1)
#define PROJECTION_FIRST_3 uint(PROJECTION_N0 + PROJECTION_N1 + PROJECTION_N2)
#define PROJECTION_FIRST_4 uint(PROJECTION_N0 + PROJECTION_N1 + PROJECTION_N2 + PROJECTION_N3)
#define PROJECTION_FIRST_5 uint(PROJECTION_N0 + PROJECTION_N1 + PROJECTION_N2 + PROJECTION_N3 + PROJECTION_N4)
#define PROJECTION_FIRST_6 uint(PROJECTION_N0 + PROJECTION_N1 + PROJECTION_N2 + PROJECTION_N3 + PROJECTION_N4 + PROJECTION_N5)
#define PROJECTION_FIRST_7 uint(PROJECTION_N0 + PROJECTION_N1 + PROJECTION_N2 + PROJECTION_N3 + PROJECTION_N4 + PROJECTION_N5 + PROJECTION_N6)

// `segment`, the segment of threadgroup TILE of a launch whose threadgroups
// each take PER columns of their segment, and `local`, the threadgroup
// within it.
#define PROJECTION_SEGMENT_FOUND(s, PER)                                                                        \
    if (segment == s##u) {                                                                                      \
        const uint tiles = (uint(PROJECTION_N##s) + (PER) - 1u) / (PER);                                        \
        if (local >= tiles) {                                                                                   \
            local -= tiles;                                                                                     \
            segment = s##u + 1u;                                                                                \
        }                                                                                                       \
    }
#define PROJECTION_SEGMENT_FIND(TILE, PER)                                                                      \
    uint segment = 0u, local = (TILE);                                                                          \
    PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_FOUND, PER)
// The same over tiles of PER features of a single weight, a pair's tile
// taking half (it covers both weights' features).
#define PROJECTION_SEGMENT_FOUND_TILES(s, PER)                                                                  \
    PROJECTION_SEGMENT_FOUND(s, (PROJECTION_PAIRED(s) ? (PER) / 2u : (PER)))
#define PROJECTION_SEGMENT_FIND_TILES(TILE, PER)                                                                \
    uint segment = 0u, local = (TILE);                                                                          \
    PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_FOUND_TILES, PER)

// A zeroed segment's threadgroup stores zeros over its rows ROW0 .. ROW0 +
// ROWS and its PER columns, THREADS threads, and returns.
#define PROJECTION_SEGMENT_ZEROS(s, ROW0, ROWS, PER, THREADS)                                                   \
    else if (segment == s##u && PROJECTION_ZEROED(s)) {                                                         \
        projection::zero_segment<PROJECTION_PAIRED(s)>(out##s, m, uint(PROJECTION_N##s), ROW0, ROWS,           \
            local * (PER), PER,                                                                                 \
            sg * 32u + lane, THREADS);                                                                          \
        return;                                                                                                 \
    }
#define PROJECTION_SEGMENTED_ZEROS(ROW0, ROWS, PER, THREADS)                                                    \
    PROJECTION_SEGMENT_CHAIN                                                                                    \
    PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_ZEROS, ROW0, ROWS, PER, THREADS)

#define PROJECTION_SEGMENT_GEMV(s, COUNT, TILED, PER)                                                           \
    else if (segment == s##u) {                                                                                 \
        PROJECTION_GEMV_PROLOGUE(s);                                                                            \
        {                                                                                                       \
            const uint tile = local;                                                                            \
            PROJECTION_GEMV(PROJECTION_PACKET(s), PROJECTION_PARTNER_PACKET(s), PROJECTION_PAIRED(s), ROWS, LANES,  \
                TILED, COUNT, x, out##s, w##s, PROJECTION_PARTNER(s), m, uint(PROJECTION_N##s), k);              \
        }                                                                                                       \
        PROJECTION_GEMV_EPILOGUE(s, local * (PER), metal::min(PER, uint(PROJECTION_N##s) - local * (PER)));     \
    }

// The GEMV of a launch that serves COUNT (ONE, PAIR, SEVERAL) rows in tile
// form TILED.
#define PROJECTION_SEGMENTED_GEMV(COUNT, TILED)                                                                 \
    PROJECTION_OPERANDS;                                                                                        \
    const uint per = simdgroups * ROWS * (32u / LANES);                                                         \
    PROJECTION_SEGMENT_FIND(position.x, per);                                                                   \
    PROJECTION_SEGMENTED_ZEROS(0u, m, per, simdgroups * 32u);                                                   \
    PROJECTION_GEMV_INPUT(input);                                                                               \
    PROJECTION_OPERANDS_SHARED(globals, NVFP4);                                                                 \
    const auto x = Operands::rows(input, m, k, globals, simdgroups, sg, lane);                                  \
    PROJECTION_SEGMENT_CHAIN                                                                                    \
    PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_GEMV, COUNT, TILED, per)

#define PROJECTION_SEGMENT_HAS_MATRIX(s, ...)                                                                   \
    || (uint(PROJECTION_N##s) > 0u && projection::matrix_form<PROJECTION_PACKET(s)>::available                  \
        && (!PROJECTION_PAIRED(s) || projection::matrix_form<PROJECTION_PARTNER_PACKET(s)>::available))

#define PROJECTION_SEGMENT_BATCH(s, x, QMATRIX)                                                                 \
    else if (segment == s##u) {                                                                                 \
        PROJECTION_BATCH_PROLOGUE(s);                                                                           \
        projection::segment_batch<PROJECTION_PACKET(s), PROJECTION_PARTNER_PACKET(s), PROJECTION_PAIRED(s),     \
            BATCH_ROWS, BATCH_PARTS, QMATRIX, PROJECTION_COLUMNS(s)>(x, out##s, w##s, PROJECTION_PARTNER(s), m, \
            uint(PROJECTION_N##s), k, local, shared, simdgroups, sg, lane);                                     \
    }

// The batched launch over the staged rows: when any active segment supports
// `matrix_form` and k % 32 == 0, uses the matrix GEMV grid (`QMATRIX`), else
// the batched-rows grid.
#define PROJECTION_SEGMENTED_BATCH                                                                              \
    PROJECTION_OPERANDS;                                                                                        \
    constexpr bool QMATRIX_STATIC = false PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_HAS_MATRIX, );             \
    const bool qmatrix = QMATRIX_STATIC && (k % 32u == 0u);                                                     \
    const uint per = qmatrix ? 8u * simdgroups / BATCH_PARTS : 8u * simdgroups * BATCH_ROWS;                    \
    PROJECTION_SEGMENT_FIND(position.x, per);                                                                   \
    PROJECTION_SEGMENTED_ZEROS(0u, m, per, simdgroups * 32u);                                                   \
    if constexpr (QMATRIX_STATIC) {                                                                             \
        if (qmatrix) {                                                                                          \
            PROJECTION_SEGMENT_CHAIN                                                                            \
            PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_BATCH, in, true)                                         \
        } else {                                                                                                \
            PROJECTION_SEGMENT_CHAIN                                                                            \
            PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_BATCH, in, false)                                        \
        }                                                                                                       \
    } else {                                                                                                    \
        PROJECTION_SEGMENT_CHAIN                                                                                \
        PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_BATCH, in, false)                                            \
    }

// SKIP(s): whether the tiles leave segment s to another launch.
#define PROJECTION_SEGMENT_TILES(s, TM, TN, SKIP)                                                               \
    else if (segment == s##u && !(SKIP(s)))                                                                     \
        projection::segment_gemm<PROJECTION_PACKET(s), PROJECTION_PARTNER_PACKET(s), PROJECTION_PAIRED(s), TM, TN>(in,  \
            out##s, w##s, PROJECTION_PARTNER(s), m, uint(PROJECTION_N##s), k, tile.y, local, shared, sg, lane);
#define PROJECTION_NONE(s) false

// The staged tiles of the segments not SKIP, TM x TN each, THREADS threads.
#define PROJECTION_SEGMENTED_TILES(TM, TN, THREADS, SKIP)                                                       \
    PROJECTION_GEMM_SHARED(shared, TM, TN);                                                                     \
    PROJECTION_OPERANDS;                                                                                        \
    PROJECTION_SEGMENT_FIND_TILES(tile.x, TN);                                                              \
    PROJECTION_SEGMENTED_ZEROS(tile.y * (TM), TM, TN, THREADS);                                                 \
    PROJECTION_SEGMENT_CHAIN                                                                                    \
    PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_TILES, TM, TN, SKIP)

// 17..64 rows: the fixed small-row tile.
#define PROJECTION_SEGMENTED_GEMM_SMALL                                                                         \
    PROJECTION_SEGMENTED_TILES(projection::small_tile_m, projection::small_tile_n, 128u, PROJECTION_NONE)

// The tuned tiles past 64 rows.
#define PROJECTION_SEGMENTED_GEMM                                                                               \
    PROJECTION_SEGMENTED_TILES(TILE_M, TILE_N, max(TILE_M, 64u) * TILE_N / 32u, PROJECTION_NONE)

#define PROJECTION_SEGMENT_TALL(s, ...)                                                                         \
    else if (segment == s##u)                                                                                   \
        projection::segment_tall<PROJECTION_PACKET(s), PROJECTION_PARTNER_PACKET(s), PROJECTION_PAIRED(s), TALL_M,     \
            TALL_K, STAGERS>(x, out##s, w##s, PROJECTION_PARTNER(s), m, uint(PROJECTION_N##s), k, tile.y, local,  \
            shared, sg, lane);

// The TALL form past 64 rows: the tall GEMM's tiles over the rows staged in
// its order, 32 weight rows of one segment each.
#define PROJECTION_SEGMENTED_TALL                                                                               \
    PROJECTION_GEMM_TALL_SHARED(shared, TALL_K);                                                                \
    PROJECTION_OPERANDS;                                                                                        \
    const auto x = projection::tall_operand(in, PROJECTION_FRAGMENTS);                                          \
    PROJECTION_SEGMENT_FIND_TILES(tile.x, projection::tall_n);                                                        \
    PROJECTION_SEGMENTED_ZEROS(tile.y * TALL_M, TALL_M, projection::tall_n, (TALL_M / 32u + STAGERS) * 32u);    \
    PROJECTION_SEGMENT_CHAIN                                                                                    \
    PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_TALL, )

// The PACK form past 64 rows: the staged rows as integer codes under the
// gain limits of the weights the operand serves (one operand for every
// segment, laid out for the weights that serve the most weight rows, of
// segments whose rows are whole 32-row tiles: `PROJECTION_SERVED`), the
// served weights' block scales and biases (each table in segment order),
// then their packed tiles in segment order; the staged 64 x 64 tiles
// (`unpacked`) run every other segment, and every zeroed one.
// The weights and the rows the operand may serve, of each segment (spelled
// out: the serving test runs inside `PROJECTION_EACH_SEGMENT`, which does not
// nest).
#define PROJECTION_PACKING_ROWS(s) , PROJECTION_PACKET(s), uint(PROJECTION_N##s % 32 == 0 ? PROJECTION_N##s : 0)
#define PROJECTION_PACKING_1 PROJECTION_PACKING_ROWS(0)
#define PROJECTION_PACKING_2 PROJECTION_PACKING_1 PROJECTION_PACKING_ROWS(1)
#define PROJECTION_PACKING_3 PROJECTION_PACKING_2 PROJECTION_PACKING_ROWS(2)
#define PROJECTION_PACKING_4 PROJECTION_PACKING_3 PROJECTION_PACKING_ROWS(3)
#define PROJECTION_PACKING_5 PROJECTION_PACKING_4 PROJECTION_PACKING_ROWS(4)
#define PROJECTION_PACKING_6 PROJECTION_PACKING_5 PROJECTION_PACKING_ROWS(5)
#define PROJECTION_PACKING_7 PROJECTION_PACKING_6 PROJECTION_PACKING_ROWS(6)
#define PROJECTION_PACKING_8 PROJECTION_PACKING_7 PROJECTION_PACKING_ROWS(7)
#define PROJECTION_PACKING_OF_COUNT(COUNT) PROJECTION_PACKING_##COUNT
#define PROJECTION_PACKING_EXPANDED(COUNT) PROJECTION_PACKING_OF_COUNT(COUNT)
// The segmented operand classes coded for WIDTH residue bits where present;
// every `N % 32 == 0` packable segment is served by its own class slot.
#define PROJECTION_PACKING_CLASSES_OF(WIDTH)                                                                    \
    projection::packing_classes<WIDTH PROJECTION_PACKING_EXPANDED(SEISMIC_PROJECTION_SEGMENTS)>
#define PROJECTION_PACKING_CLASSES PROJECTION_PACKING_CLASSES_OF(1)
#define PROJECTION_SERVED(s)                                                                                    \
    (PROJECTION_N##s != 0 && PROJECTION_N##s % 32 == 0                                                          \
        && projection::packing_codes<PROJECTION_PACKET(s)>::available)
#define PROJECTION_ADMITTED(s, ...)                                                                             \
    (packets::admits<PROJECTION_PACKET(s)>::pack && PROJECTION_N##s != 0 && PROJECTION_N##s % 32 == 0) ||
// PACK = 1 names the packed GEMM of a segment of weights admitted to it (the
// projection's `where`).
#define PROJECTION_PACKING_ADMITTED                                                                             \
    static_assert(PROJECTION_EACH_SEGMENT(PROJECTION_ADMITTED, ) false,                                         \
        "PACK = 1 names the packed GEMM of a segment of weights admitted to it")
// Segment s's factor tables beside class `slot`'s operand slice: each class
// holds its own `packed`, `token_factors`, and `token_scales` slice in
// canonical class order; segment weights' factors are laid out with 4 halfs
// per (row, 32 cols) when any segment uses `sums`, else 2.
#define PROJECTION_SEGMENT_SCRATCH_AT(s, slot, factor_mult, padded_rows)                                        \
    projection::packing_scratch{                                                                                \
        packed + ulong(slot) * (ulong(padded_rows) * k / 2u + 256u),                                            \
        token_factors + ulong(slot) * ulong(padded_rows) * (k / 32u) * 2u * (factor_mult),                      \
        token_scales + ulong(slot) * ulong(padded_rows),                                                        \
        factors + ulong(PROJECTION_FIRST_##s) * (k / 32u) * (PROJECTION_PACKING_CLASSES::any_sums ? 4u : 2u),   \
        scales + PROJECTION_FIRST_##s}

#define PROJECTION_SEGMENTED_PACK_CLASS(CLS, SLOT)                                                              \
    if constexpr (CLS::active) {                                                                                \
        if (slot == (SLOT)) {                                                                                   \
            constexpr uint BOUNDS = projection::packing_bounded<CLS::mask, PROJECTION_BOUNDED>::value;          \
            const projection::packing_bounds<BOUNDS> bounds{PROJECTION_BOUND_TABLES};                           \
            projection::packing_operand<CLS::centre, CLS::folds, CLS::sums, CLS::residue, BOUNDS>(              \
                in, PROJECTION_SEGMENT_SCRATCH_AT(0, SLOT, factor_mult, padded), bounds, m, k, local_index,     \
                peaks, thread_index, sg, lane);                                                                 \
            return;                                                                                             \
        }                                                                                                       \
    }

#define PROJECTION_SEGMENTED_PACK                                                                               \
    PROJECTION_PACKING_ADMITTED;                                                                                \
    if constexpr (!projection::packing_forms)                                                                   \
        return;                                                                                                 \
    threadgroup float peaks[projection::packing_peaks()];                                                       \
    PROJECTION_OPERANDS;                                                                                        \
    typedef PROJECTION_PACKING_CLASSES_OF(RESIDUE) classes;                                                     \
    const uint padded = projection::packing_padded(m, 2u);                                                      \
    const uint groups_per_class = padded / 4u;                                                                  \
    const uint slot = index / groups_per_class;                                                                 \
    const uint local_index = index % groups_per_class;                                                          \
    constexpr uint factor_mult = RESIDUE > (classes::any_sums ? 2u : 1u) ? RESIDUE : (classes::any_sums ? 2u : 1u); \
    PROJECTION_SEGMENTED_PACK_CLASS(classes::c_one, classes::slot_one)                                          \
    PROJECTION_SEGMENTED_PACK_CLASS(classes::c_two, classes::slot_two)                                          \
    PROJECTION_SEGMENTED_PACK_CLASS(classes::c_summed, classes::slot_summed)                                    \
    PROJECTION_SEGMENTED_PACK_CLASS(classes::c_bytes, classes::slot_bytes)

#define PROJECTION_SEGMENT_COEFFICIENTS(s, ...)                                                                 \
    if (row < uint(PROJECTION_N##s)) {                                                                          \
        if constexpr (PROJECTION_SERVED(s))                                                                     \
            projection::packing_coefficients(w##s, PROJECTION_SEGMENT_SCRATCH_AT(s, 0u, 1u, 0u),                \
                uint(PROJECTION_N##s), k, row, lane);                                                           \
        return;                                                                                                 \
    }                                                                                                           \
    row -= uint(PROJECTION_N##s);

// One weight row of the segments' in turn a threadgroup.
#define PROJECTION_SEGMENTED_PACK_COEFFICIENTS                                                                  \
    PROJECTION_PACKING_ADMITTED;                                                                                \
    PROJECTION_OPERANDS;                                                                                        \
    uint row = index;                                                                                           \
    PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_COEFFICIENTS, )

#define PROJECTION_SEGMENT_PACKED(s, ...)                                                                       \
    if (projection::gemm_packed_segment<PROJECTION_PACKET(s), PROJECTION_SERVED(s),                             \
            classes::template of<PROJECTION_PACKET(s)>::residue, PACK_TOKENS, WEIGHTS_AHEAD>(                   \
            out##s, w##s,                                                                                       \
            PROJECTION_SEGMENT_SCRATCH_AT(s, classes::template of<PROJECTION_PACKET(s)>::slot, factor_mult, padded), \
            PROJECTION_ZEROED(s) ? 0u : m, padded, uint(PROJECTION_N##s), k, group,                             \
            projection::packing_simdgroups, sg, lane))                                                          \
        return;

#define PROJECTION_SEGMENTED_PACKED                                                                             \
    PROJECTION_PACKING_ADMITTED;                                                                                \
    PROJECTION_OPERANDS;                                                                                        \
    typedef PROJECTION_PACKING_CLASSES_OF(RESIDUE) classes;                                                     \
    const uint padded = projection::packing_padded(m, 2u);                                                      \
    constexpr uint factor_mult = RESIDUE > (classes::any_sums ? 2u : 1u) ? RESIDUE : (classes::any_sums ? 2u : 1u); \
    uint group = index;                                                                                         \
    PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_PACKED, )

#define PROJECTION_SEGMENTED_UNPACKED                                                                           \
    PROJECTION_PACKING_ADMITTED;                                                                                \
    PROJECTION_SEGMENTED_TILES(64u, 64u, 128u, PROJECTION_SERVED)

// ---------------------------------------------------------------------------
// Parts: a `segmented` projection whose K is PROJECTION_PARTS parts of
// k / PROJECTION_PARTS columns, each part's rows of their own (its own norm
// group), splits its GEMV along K by part. Threadgroup `position.x` serves
// part position.x / tiles over its row tile position.x % tiles of the
// segments (`tiles` a part's), normalizing only its part, and stores F32
// partial sums over the part's packets; each row tile has its own arrival
// counter, and the last of its PROJECTION_PARTS threadgroups to arrive adds
// the tile's outputs' partials in part order through the segments'
// epilogues, so no one threadgroup adds every output after the launch. The
// segments are unpaired. The entry includes `lib/core/arrive.h` and defines,
// beside the segmented contract (whose GEMMs read `in`),
//   PROJECTION_PARTS                    the parts;
//   PROJECTION_PART_INPUT(x, part, first)
//                                       `x`, part `part`'s rows as the GEMV
//                                       reads them (`first`: the part's first
//                                       row tile, which publishes whatever
//                                       the part's prologue publishes);
// and, among its arguments, `partials` (arrive::coherent_device<float>, m *
// PARTS * outputs: output n of row m from part p at (m * PARTS + p) *
// outputs + n, outputs the segments' rows in order) and `arrivals` (one
// counter a row tile of a part).

#ifdef PROJECTION_PARTS
namespace projection {

// One part's partial sums of the segments' outputs, from column `column`.
struct PartPartial {
    arrive::coherent_device<float> y;
    ulong row;
    ulong column;
    void store(uint m, uint n, float value) const { y[ulong(m) * row + column + n] = value; }
};

} // namespace projection

#define PROJECTION_ADD_ROWS(s, ...) + uint(PROJECTION_N##s)
#define PROJECTION_ADD_TILES(s, PER) + (uint(PROJECTION_N##s) + (PER) - 1u) / (PER)

#define PROJECTION_SEGMENT_PART_GEMV(s, COUNT)                                                                  \
    else if (segment == s##u)                                                                                   \
        PROJECTION_FOR_##COUNT##_ROWS(m,                                                                        \
            projection::gemv_runtime<PROJECTION_PACKET(s), ROWS, MAXM, LANES, (STAGE == 1)>(x,                  \
            projection::PartPartial{sums.y, sums.row, sums.column + PROJECTION_FIRST_##s}, w##s, m,             \
            uint(PROJECTION_N##s), width, local, shared, simdgroups, sg, lane, part * (width / 32u)));

#define PROJECTION_SEGMENT_PART_COMBINE(s, ...)                                                                 \
    else if (segment == s##u) {                                                                                 \
        const uint column0 = local * per;                                                                       \
        const uint count = metal::min(per, uint(PROJECTION_N##s) - column0);                                    \
        for (uint item = thread_index; item < m * count; item += threads) {                                     \
            const uint row = item / count, column = column0 + item % count;                                     \
            float sum = 0.0f;                                                                                   \
            for (uint p = 0; p < parts; ++p)                                                                    \
                sum += partials[(ulong(row) * parts + p) * outputs + PROJECTION_FIRST_##s + column];            \
            out##s.store(row, column, sum);                                                                     \
        }                                                                                                       \
    }

// The GEMV of a parted row class (`ONE`, `PAIR`, `SEVERAL`).
#define PROJECTION_SEGMENTED_PARTS_GEMV(COUNT)                                                                  \
    PROJECTION_OPERANDS;                                                                                        \
    const uint parts = uint(PROJECTION_PARTS), width = k / parts;                                               \
    const uint per = simdgroups * ROWS * (32u / LANES);                                                         \
    const uint outputs = 0u PROJECTION_EACH_SEGMENT(PROJECTION_ADD_ROWS, );                                     \
    const uint tiles = 0u PROJECTION_EACH_SEGMENT(PROJECTION_ADD_TILES, per);                                   \
    const uint part = position.x / tiles, row_tile = position.x % tiles;                                        \
    PROJECTION_PART_INPUT(x, part, row_tile == 0u);                                                             \
    const projection::PartPartial sums{partials, ulong(parts) * outputs, ulong(part) * outputs};                \
    PROJECTION_SEGMENT_FIND(row_tile, per);                                                                     \
    PROJECTION_SEGMENT_CHAIN                                                                                    \
    PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_PART_GEMV, COUNT)                                                \
    threadgroup uint last;                                                                                      \
    const uint thread_index = sg * 32u + lane, threads = simdgroups * 32u;                                      \
    if (!arrive::last(arrivals + row_tile, parts, &last, thread_index))                                         \
        return;                                                                                                 \
    PROJECTION_SEGMENT_CHAIN                                                                                    \
    PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_PART_COMBINE, )

#define PROJECTION_SEGMENT_PART_BATCH(s, ...)                                                                   \
    else if (segment == s##u)                                                                                   \
        projection::gemv_batch_runtime<PROJECTION_PACKET(s), BATCH_ROWS>(x,                                     \
            projection::PartPartial{sums.y, sums.row, sums.column + PROJECTION_FIRST_##s}, w##s, m,             \
            uint(PROJECTION_N##s), width, local, shared, simdgroups, sg, lane, part * (width / 32u));

#define PROJECTION_SEGMENTED_PARTS_BATCH                                                                        \
    PROJECTION_OPERANDS;                                                                                        \
    const uint parts = uint(PROJECTION_PARTS), width = k / parts;                                               \
    const uint per = 8u * simdgroups * BATCH_ROWS;                                                              \
    const uint outputs = 0u PROJECTION_EACH_SEGMENT(PROJECTION_ADD_ROWS, );                                     \
    const uint tiles = 0u PROJECTION_EACH_SEGMENT(PROJECTION_ADD_TILES, per);                                   \
    const uint part = position.x / tiles, row_tile = position.x % tiles;                                        \
    PROJECTION_PART_INPUT(x, part, row_tile == 0u);                                                             \
    const projection::PartPartial sums{partials, ulong(parts) * outputs, ulong(part) * outputs};                \
    PROJECTION_SEGMENT_FIND(row_tile, per);                                                                     \
    PROJECTION_SEGMENT_CHAIN                                                                                    \
    PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_PART_BATCH, )                                                    \
    threadgroup uint last;                                                                                      \
    const uint thread_index = sg * 32u + lane, threads = simdgroups * 32u;                                      \
    if (!arrive::last(arrivals + row_tile, parts, &last, thread_index))                                         \
        return;                                                                                                 \
    PROJECTION_SEGMENT_CHAIN                                                                                    \
    PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_PART_COMBINE, )

// Split-K GEMM and reduction for parted segmented projections (`stream_mix_down`,
// SERVED == 1).
//
// Threadgroup memory math for TM x TN staged tile (gemm_k = 32, gemm_lda = 40):
//   A tile = TM * gemm_lda * 2 bytes; B tile = TN * gemm_lda * 2 bytes.
//   64x32: A = 64 * 40 * 2 = 5,120 B, B = 32 * 40 * 2 = 2,560 B -> 7,680 B (7.5 KB <= 32 KB).
//   64x64: A = 5,120 B, B = 5,120 B -> 10,240 B (10.0 KB <= 32 KB).
//   128x128: A = 10,240 B, B = 10,240 B -> 20,480 B (20.0 KB <= 32 KB).
// Single `PROJECTION_GEMM_SHARED` declaration outside the `SPLIT == 1` branch so
// MSL never allocates two static threadgroup buffers.
//
// Tail bounds & reduction order:
//   - M tail: `gemm_load_a` zero-pads rows >= m, `gemm_store` clips `m0 + r < m`.
//   - N tail: `gemm_load_b` clamps weight rows to `rows - 1`, `gemm_store` clips `first + c < rows`.
//   - K tail: `steps = ceil_div(k, 32)` is partitioned contiguously across
//     `part = 0 .. split - 1` as `[part * steps / split, (part + 1) * steps / split)`;
//     `gemm_load_a` / `gemm_load_b` zero-pad columns >= k in the final step.
//   - Reduction: `PROJECTION_SEGMENTED_PARTS_REDUCE` sums raw F32 partials in
//     strict ascending `part` order (0 .. split - 1) before applying each
//     segment's epilogue once, matching `PartPartial` layout
//     `partials[(row * split + part) * outputs + PROJECTION_FIRST_s + col]`.
#define PROJECTION_SEGMENT_PART_GEMM(s, TM, TN)                                                                 \
    else if (segment == s##u) {                                                                                 \
        const projection::PartPartial seg_sums{sums.y, sums.row, sums.column + PROJECTION_FIRST_##s};           \
        projection::gemm_run<PROJECTION_PACKET(s), PROJECTION_PACKET(s), false, TM, TN>(in, w##s, w##s,         \
            local * (TN), uint(PROJECTION_N##s), tile.y * (TM), m, k, step_begin, step_end, m, shared, sg,      \
            lane, projection::gemm_store<projection::PartPartial>{seg_sums, tile.y * (TM), m, local * (TN),     \
                uint(PROJECTION_N##s)});                                                                        \
    }

#define PROJECTION_SEGMENTED_PARTS_GEMM(SPLIT)                                                                  \
    PROJECTION_GEMM_SHARED(shared, TILE_M, TILE_N);                                                             \
    PROJECTION_OPERANDS;                                                                                        \
    PROJECTION_SEGMENT_FIND_TILES(tile.x, TILE_N);                                                              \
    if ((SPLIT) == 1u) {                                                                                        \
        PROJECTION_SEGMENTED_ZEROS(tile.y * (TILE_M), TILE_M, TILE_N, max(TILE_M, 64u) * TILE_N / 32u);         \
        PROJECTION_SEGMENT_CHAIN                                                                                \
        PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_TILES, TILE_M, TILE_N, PROJECTION_NONE)                      \
    } else {                                                                                                    \
        const uint split = (SPLIT);                                                                             \
        const uint outputs = 0u PROJECTION_EACH_SEGMENT(PROJECTION_ADD_ROWS, );                                 \
        const uint steps = (k + projection::gemm_k - 1u) / projection::gemm_k;                                  \
        const uint step_begin = tile.z * steps / split;                                                         \
        const uint step_end = (tile.z + 1u) * steps / split;                                                    \
        const projection::PartPartial sums{partials, ulong(split) * outputs, ulong(tile.z) * outputs};          \
        PROJECTION_SEGMENT_CHAIN                                                                                \
        PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_PART_GEMM, TILE_M, TILE_N)                                   \
    }

#define PROJECTION_SEGMENT_PART_REDUCE(s, ...)                                                                  \
    else if (col < PROJECTION_FIRST_##s + uint(PROJECTION_N##s))                                                \
        out##s.store(row, col - PROJECTION_FIRST_##s, sum);

#define PROJECTION_SEGMENTED_PARTS_REDUCE(SPLIT)                                                                \
    PROJECTION_OPERANDS;                                                                                        \
    const uint outputs = 0u PROJECTION_EACH_SEGMENT(PROJECTION_ADD_ROWS, );                                     \
    const uint index = position.x;                                                                              \
    if (index >= m * outputs)                                                                                   \
        return;                                                                                                 \
    const uint row = index / outputs, col = index % outputs;                                                    \
    const uint split = (SPLIT);                                                                                 \
    float sum = 0.0f;                                                                                           \
    for (uint p = 0u; p < split; ++p)                                                                           \
        sum += partials[(ulong(row) * split + p) * outputs + col];                                              \
    PROJECTION_SEGMENT_CHAIN                                                                                    \
    PROJECTION_EACH_SEGMENT(PROJECTION_SEGMENT_PART_REDUCE, )
#endif

// ---------------------------------------------------------------------------
// Reference-order (`ordered`) forms (`ordered_stage`, `ordered_gemv`,
// `ordered_gemm`): 8-lane dot products (`((l0+l4)+(l2+l6))+((l1+l5)+(l3+l7))`)
// matching the CPU reference order bit for bit.

namespace projection {

template <typename W>
struct OrderedWeight {
    // Whether a lane fetches a block's columns in the reference order's
    // stride (`fetch_strided`), so the walk needs no transpose.
    static constant constexpr bool strided = false;
    struct fetched {
        typename W::packet pkt;
        bool valid;
    };
    static fetched empty() { return fetched{{}, false}; }
    static fetched fetch_full(thread const Weights<W> &w, uint row, uint p, uint) {
        return fetched{w.packet(row, p), true};
    }
    static fetched fetch(thread const Weights<W> &w, uint row, uint p, uint) {
        return fetched{w.packet(row, p), true};
    }
    // A fetch is decoded by value: `GemmPrefetch` holds it as a member,
    // which Metal 4.1 reaches through `this` in no stated address space.
    static void decode_full(fetched f, uint step, thread float (&wt)[8]) {
        if (!f.valid) {
            PROJECTION_UNROLL
            for (uint j = 0; j < 8; ++j)
                wt[j] = 0.0f;
            return;
        }
        float4 even, odd;
        W::exact_values(f.pkt, step, even, odd);
        PROJECTION_UNROLL
        for (uint j = 0; j < 4; ++j) {
            wt[2u * j] = even[j];
            wt[2u * j + 1u] = odd[j];
        }
    }
    static void decode(fetched f, uint step, uint base_col, uint k, thread float (&wt)[8]) {
        if (!f.valid) {
            PROJECTION_UNROLL
            for (uint j = 0; j < 8; ++j)
                wt[j] = 0.0f;
            return;
        }
        float4 even, odd;
        W::exact_values(f.pkt, step, even, odd);
        if (base_col + 8u <= k) {
            PROJECTION_UNROLL
            for (uint j = 0; j < 4; ++j) {
                wt[2u * j] = even[j];
                wt[2u * j + 1u] = odd[j];
            }
        } else {
            PROJECTION_UNROLL
            for (uint j = 0; j < 4; ++j) {
                wt[2u * j] = (base_col + 2u * j < k) ? even[j] : 0.0f;
                wt[2u * j + 1u] = (base_col + 2u * j + 1u < k) ? odd[j] : 0.0f;
            }
        }
    }
};

template <typename E>
struct OrderedWeight<packets::Dense<E>> {
    static constant constexpr uint words = E::bytes / 2u;
    struct fetched {
        uint4 raw[words];
    };
    static fetched empty() {
        fetched out;
        PROJECTION_UNROLL
        for (uint i = 0; i < words; ++i)
            out.raw[i] = uint4(0u);
        return out;
    }
    static fetched fetch_full(thread const Weights<packets::Dense<E>> &w, uint row, uint p, uint step) {
        fetched out;
        const uint base_col = 32u * p + 8u * step;
        const auto at = w.locate(row);
        const packets::weight_bytes ptr = packets::Dense<E>::row_values(at) + ulong(base_col) * E::bytes;
        if (packets::Dense<E>::aligned16(at)) {
            const auto v = slab::as<uint4>(ptr);
            PROJECTION_UNROLL
            for (uint i = 0; i < words; ++i)
                out.raw[i] = v[i];
            return out;
        }
        const auto s = slab::as<typename E::storage>(ptr);
        if constexpr (E::bytes == 2u) {
            const uint w0 = uint(as_type<ushort>(s[0])) | (uint(as_type<ushort>(s[1])) << 16u);
            const uint w1 = uint(as_type<ushort>(s[2])) | (uint(as_type<ushort>(s[3])) << 16u);
            const uint w2 = uint(as_type<ushort>(s[4])) | (uint(as_type<ushort>(s[5])) << 16u);
            const uint w3 = uint(as_type<ushort>(s[6])) | (uint(as_type<ushort>(s[7])) << 16u);
            out.raw[0] = uint4(w0, w1, w2, w3);
        } else {
            out.raw[0] = as_type<uint4>(float4(s[0], s[1], s[2], s[3]));
            out.raw[1] = as_type<uint4>(float4(s[4], s[5], s[6], s[7]));
        }
        return out;
    }
    static constant constexpr bool strided = true;
    // Columns 64 block + 8 s + u (s = 0..7) of a whole block, value s in
    // place s: lane u's operands after `ordered_transpose8`, each read where
    // the eight lanes of a row read 16 consecutive bytes.
    static fetched fetch_strided(thread const Weights<packets::Dense<E>> &w, uint row, uint block, uint u) {
        fetched out;
        const auto s = slab::as<typename E::storage>(packets::Dense<E>::row_values(w.locate(row))
            + ulong(64u * block + u) * E::bytes);
        if constexpr (E::bytes == 2u) {
            uint pair[4];
            PROJECTION_UNROLL
            for (uint j = 0; j < 4u; ++j)
                pair[j] = uint(as_type<ushort>(s[16u * j])) | (uint(as_type<ushort>(s[16u * j + 8u])) << 16u);
            out.raw[0] = uint4(pair[0], pair[1], pair[2], pair[3]);
        } else {
            out.raw[0] = as_type<uint4>(float4(s[0], s[8], s[16], s[24]));
            out.raw[1] = as_type<uint4>(float4(s[32], s[40], s[48], s[56]));
        }
        return out;
    }
    static fetched fetch(thread const Weights<packets::Dense<E>> &w, uint row, uint p, uint step) {
        const uint base_col = 32u * p + 8u * step;
        if (base_col + 8u <= w.matrix.k)
            return fetch_full(w, row, p, step);
        fetched out = empty();
        if (base_col >= w.matrix.k)
            return out;
        const packets::weight_bytes ptr = packets::Dense<E>::row_values(w.locate(row)) + ulong(base_col) * E::bytes;
        const auto s = slab::as<typename E::storage>(ptr);
        const uint rem = w.matrix.k - base_col;
        if constexpr (E::bytes == 2u) {
            uint w0 = 0u, w1 = 0u, w2 = 0u, w3 = 0u;
            if (0u < rem) w0 |= uint(as_type<ushort>(s[0]));
            if (1u < rem) w0 |= uint(as_type<ushort>(s[1])) << 16u;
            if (2u < rem) w1 |= uint(as_type<ushort>(s[2]));
            if (3u < rem) w1 |= uint(as_type<ushort>(s[3])) << 16u;
            if (4u < rem) w2 |= uint(as_type<ushort>(s[4]));
            if (5u < rem) w2 |= uint(as_type<ushort>(s[5])) << 16u;
            if (6u < rem) w3 |= uint(as_type<ushort>(s[6]));
            if (7u < rem) w3 |= uint(as_type<ushort>(s[7])) << 16u;
            out.raw[0] = uint4(w0, w1, w2, w3);
        } else {
            const float v0 = (0u < rem) ? s[0] : 0.0f;
            const float v1 = (1u < rem) ? s[1] : 0.0f;
            const float v2 = (2u < rem) ? s[2] : 0.0f;
            const float v3 = (3u < rem) ? s[3] : 0.0f;
            const float v4 = (4u < rem) ? s[4] : 0.0f;
            const float v5 = (5u < rem) ? s[5] : 0.0f;
            const float v6 = (6u < rem) ? s[6] : 0.0f;
            const float v7 = (7u < rem) ? s[7] : 0.0f;
            out.raw[0] = as_type<uint4>(float4(v0, v1, v2, v3));
            out.raw[1] = as_type<uint4>(float4(v4, v5, v6, v7));
        }
        return out;
    }
    static void decode_full(fetched f, uint s, thread float (&wt)[8]) {
        decode(f, s, 0u, 8u, wt);
    }
    static void decode(fetched f, uint, uint, uint, thread float (&wt)[8]) {
        if constexpr (E::bytes == 2u) {
            float4 even, odd;
            E::split8(f.raw[0], even, odd);
            PROJECTION_UNROLL
            for (uint j = 0; j < 4; ++j) {
                wt[2u * j] = even[j];
                wt[2u * j + 1u] = odd[j];
            }
        } else {
            const float4 lo = as_type<float4>(f.raw[0]);
            const float4 hi = as_type<float4>(f.raw[1]);
            wt[0] = lo.x;
            wt[1] = lo.y;
            wt[2] = lo.z;
            wt[3] = lo.w;
            wt[4] = hi.x;
            wt[5] = hi.y;
            wt[6] = hi.z;
            wt[7] = hi.w;
        }
    }
};

template <typename W>
struct Dense16Trait {
    static constant constexpr bool value = false;
    typedef element::Bf16 element;
};

template <typename E>
struct Dense16Trait<packets::Dense<E>> {
    static constant constexpr bool value = (E::bytes == 2u);
    typedef E element;
};

template <typename W>
inline float ordered_weight_at(thread const Weights<W> &w, uint row, uint col) {
    float4 even, odd;
    W::exact_values(w.packet(row, col >> 5u), (col >> 3u) & 3u, even, odd);
    const uint u = col & 7u;
    const float4 hv = (u & 1u) != 0u ? odd : even;
    const uint i = u >> 1u;
    return i == 0u ? hv.x : (i == 1u ? hv.y : (i == 2u ? hv.z : hv.w));
}

template <typename E>
inline float ordered_weight_at(thread const Weights<packets::Dense<E>> &w, uint row, uint col) {
    return E::load(slab::as<typename E::storage>(packets::Dense<E>::row_values(w.locate(row)))[col]);
}

// Blocks a 1-row ordered walk fetches together (`ordered_gemv_maxm_run`):
// as many as 64 bytes of fetched weights hold, so a batch and the next one in
// flight fit a lane's registers; one where a block's fetch is larger.
template <typename W>
constexpr uint ordered_walk_batch() {
    constexpr uint fit = 64u / uint(sizeof(typename OrderedWeight<W>::fetched));
    return fit < 1u ? 1u : (fit > 8u ? 8u : fit);
}

template <uint MAXM>
constexpr uint ordered_mixed_w_per_sg() {
    return 4u;
}
// The simdgroups of a threadgroup that hold an ordered segment's rows: two
// at 1..2 rows (spreading a small router across many cores), four at 3..4
// rows, and up to eight at 5..8 rows so compute-heavy verify steps keep every
// simdgroup active and avoid redundant RMS/staging passes. `dense.seismic`'s
// `ordered_gemv1..8` grids match.
template <uint MAXM, uint SIMDGROUPS>
constexpr uint ordered_mixed_sgs() {
    constexpr uint cap = (MAXM >= 8u) ? 8u : ((MAXM >= 4u) ? 4u : 2u);
    return (SIMDGROUPS > cap) ? cap : SIMDGROUPS;
}
template <uint MAXM, uint SIMDGROUPS>
constexpr uint ordered_mixed_rows() {
    return ordered_mixed_sgs<MAXM, SIMDGROUPS>() * ordered_mixed_w_per_sg<MAXM>();
}

// Unified reference-order GEMV (`M <= 8`): four weight rows per simdgroup
// (`q = lane >> 3`, `u = lane & 7`, `w_row = local_tile * (4 * SIMDGROUPS) + sg * 4 + q`,
// matching CUDA and Vulkan), streaming 64 contiguous weight columns per quarter
// via coalesced `OrderedWeight<W>` register loads + `ordered_transpose8` against
// conflict-free lane-strided `uint4` activations in `8,192`B threadgroup memory.
template <typename W, bool PUBLISH, uint SIMDGROUPS, typename In, typename Out, typename PubOut>
inline void ordered_gemv_run(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    PubOut pub_out, ulong pub_stride0, ulong pub_stride1, bool publish_rows, uint m, uint n, uint k, uint local_tile,
    threadgroup uchar *shared, uint sg, uint lane) {
    typedef typename In::activation A;
    typedef OrderedWeight<W> Ow;
    const uint q = lane >> 3u;
    const uint u = lane & 7u;
    const uint sg_base_row = local_tile * (SIMDGROUPS * 4u) + sg * 4u;
    const uint w_row = sg_base_row + q;
    const bool sg_has_rows = sg_base_row < n;
    const bool has_row = w_row < n;
    const uint p_own = u >> 2u;
    const uint s_own = u & 3u;
    typename Ow::fetched cur_w = Ow::empty();
    if (has_row && k > 0u) {
        cur_w = (k >= 64u) ? Ow::fetch_full(w, w_row, p_own, s_own)
                           : ((8u * u < k) ? Ow::fetch(w, w_row, p_own, s_own) : Ow::empty());
    }
    if (m == 1u) {
        float inv0 = 0.0f;
        if (ordered_stage_single_gemv_raw<PUBLISH, SIMDGROUPS>(
                in, k, shared, pub_out, pub_stride0, pub_stride1, publish_rows, inv0, sg, lane)) {
            if (sg_has_rows) {
                threadgroup const uint4 *x_packed = reinterpret_cast<threadgroup const uint4 *>(shared);
                float acc = 0.0f;
                uint col0 = 0u;
                for (; col0 + 64u <= k; col0 += 64u) {
                    float wt[8];
                    if (has_row)
                        Ow::decode_full(cur_w, s_own, wt);
                    else {
                        PROJECTION_UNROLL
                        for (uint j = 0u; j < 8u; ++j)
                            wt[j] = 0.0f;
                    }
                    const uint next_col = col0 + 64u;
                    if (has_row) {
                        if (next_col + 64u <= k)
                            cur_w = Ow::fetch_full(w, w_row, (next_col >> 5u) + p_own, s_own);
                        else if (next_col + 8u * u < k)
                            cur_w = Ow::fetch(w, w_row, (next_col >> 5u) + p_own, s_own);
                        else
                            cur_w = Ow::empty();
                    }
                    ordered_transpose8(wt, u);
                    float4 xe, xo;
                    A::split8(x_packed[(col0 >> 6u) * 8u + u], xe, xo);
                    acc = metal::fma(xe.x, wt[0], acc);
                    acc = metal::fma(xo.x, wt[1], acc);
                    acc = metal::fma(xe.y, wt[2], acc);
                    acc = metal::fma(xo.y, wt[3], acc);
                    acc = metal::fma(xe.z, wt[4], acc);
                    acc = metal::fma(xo.z, wt[5], acc);
                    acc = metal::fma(xe.w, wt[6], acc);
                    acc = metal::fma(xo.w, wt[7], acc);
                }
                if (col0 < k) {
                    float wt[8];
                    if (has_row && col0 + 8u * u < k)
                        Ow::decode(cur_w, s_own, col0 + 8u * u, k, wt);
                    else {
                        PROJECTION_UNROLL
                        for (uint j = 0u; j < 8u; ++j)
                            wt[j] = 0.0f;
                    }
                    ordered_transpose8(wt, u);
                    float4 xe, xo;
                    A::split8(x_packed[(col0 >> 6u) * 8u + u], xe, xo);
                    const float xv[8] = {xe.x, xo.x, xe.y, xo.y, xe.z, xo.z, xe.w, xo.w};
                    PROJECTION_UNROLL
                    for (uint s = 0u; s < 8u; ++s) {
                        if (col0 + 8u * s + u < k)
                            acc = metal::fma(xv[s], wt[s], acc);
                    }
                }
                const float sum = ordered_combine8(acc);
                if (has_row && u == 0u)
                    out.store(0u, w_row, sum);
            }
            if (local_tile == 0u) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg == 0u && lane == 0u)
                    reinterpret_cast<threadgroup float *>(shared)[0] = inv0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            return;
        }
    }
    threadgroup float *smem_inv = reinterpret_cast<threadgroup float *>(shared);
    in.ordered_inverses_sram(m, smem_inv, reinterpret_cast<threadgroup float *>(shared + 64), 2032u, sg, lane, SIMDGROUPS);
    const float inv_lane = (lane < m) ? smem_inv[lane] : 0.0f;
    constexpr uint threads = SIMDGROUPS * 32u;
    const uint tid = sg * 32u + lane;
    if constexpr (PUBLISH) {
        if (publish_rows && k > 0u) {
            if (pub_stride1 == 1ul && (pub_stride0 & 7ul) == 0ul && (k & 7u) == 0u) {
                const uint vecs = k >> 3u;
                for (uint idx = tid; idx < m * vecs; idx += threads) {
                    const uint r = idx / vecs, c = (idx % vecs) * 8u;
                    float4 even, odd;
                    in.load8(r, c, smem_inv[r], even, odd);
                    pub_out.put8(ulong(r) * pub_stride0 + ulong(c), even, odd);
                }
            } else {
                for (uint idx = tid; idx < m * k; idx += threads) {
                    const uint r = idx / k, c = idx % k;
                    pub_out.put(ulong(r) * pub_stride0 + ulong(c) * pub_stride1, in.ordered_at(r, c, smem_inv[r]));
                }
            }
        }
    }
    if (m > 0u && k > 0u) {
        const uint total_blocks = (k + 63u) >> 6u;
        const uint max_blocks = metal::max(1u, metal::min(total_blocks, 63u / m));
        const uint row_cols = max_blocks * 64u;
        threadgroup uint4 *x_packed = reinterpret_cast<threadgroup uint4 *>(shared + 64);
        float acc[SEISMIC_CONST_VERIFY_ROWS];
        PROJECTION_UNROLL
        for (uint r = 0u; r < uint(SEISMIC_CONST_VERIFY_ROWS); ++r)
            acc[r] = 0.0f;
        for (uint k0 = 0u; k0 < k; k0 += row_cols) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint chunk = metal::min(row_cols, k - k0);
            const uint chunk_blocks = (chunk + 63u) >> 6u;
            for (uint base = 0u; base < m * chunk_blocks * 8u; base += threads) {
                const uint idx = base + tid;
                const uint blk = idx >> 3u;
                const uint xr = (blk < m * chunk_blocks) ? (blk / chunk_blocks) : 0u;
                const float x_inv = simd_shuffle(inv_lane, ushort(xr & 15u));
                if (blk < m * chunk_blocks) {
                    const uint xb = blk - xr * chunk_blocks;
                    const uint xs = xb * 64u + 8u * u;
                    float4 xe = float4(0.0f), xo = float4(0.0f);
                    if (xs < chunk)
                        in.load8(xr, k0 + xs, x_inv, xe, xo);
                    float xv[8] = {xe.x, xo.x, xe.y, xo.y, xe.z, xo.z, xe.w, xo.w};
                    ordered_transpose8(xv, u);
                    x_packed[(xr * chunk_blocks + xb) * 8u + u] = A::pack8(
                        float4(xv[0], xv[2], xv[4], xv[6]),
                        float4(xv[1], xv[3], xv[5], xv[7]));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg_has_rows) {
                for (uint xb = 0u; xb < chunk_blocks; ++xb) {
                    const uint sub = xb * 64u;
                    const uint col0 = k0 + sub;
                    float wt[8];
                    if (has_row) {
                        if (col0 + 64u <= k)
                            Ow::decode_full(cur_w, s_own, wt);
                        else if (col0 + 8u * u < k)
                            Ow::decode(cur_w, s_own, col0 + 8u * u, k, wt);
                        else {
                            PROJECTION_UNROLL
                            for (uint j = 0u; j < 8u; ++j)
                                wt[j] = 0.0f;
                        }
                        const uint next_col = col0 + 64u;
                        if (next_col + 64u <= k)
                            cur_w = Ow::fetch_full(w, w_row, (next_col >> 5u) + p_own, s_own);
                        else if (next_col + 8u * u < k)
                            cur_w = Ow::fetch(w, w_row, (next_col >> 5u) + p_own, s_own);
                        else
                            cur_w = Ow::empty();
                    } else {
                        PROJECTION_UNROLL
                        for (uint j = 0u; j < 8u; ++j)
                            wt[j] = 0.0f;
                    }
                    ordered_transpose8(wt, u);
                    if (sub + 64u <= chunk) {
                        PROJECTION_UNROLL
                        for (uint r = 0u; r < uint(SEISMIC_CONST_VERIFY_ROWS); ++r) {
                            if (r < m) {
                                float4 xe, xo;
                                A::split8(x_packed[(r * chunk_blocks + xb) * 8u + u], xe, xo);
                                acc[r] = metal::fma(xe.x, wt[0], acc[r]);
                                acc[r] = metal::fma(xo.x, wt[1], acc[r]);
                                acc[r] = metal::fma(xe.y, wt[2], acc[r]);
                                acc[r] = metal::fma(xo.y, wt[3], acc[r]);
                                acc[r] = metal::fma(xe.z, wt[4], acc[r]);
                                acc[r] = metal::fma(xo.z, wt[5], acc[r]);
                                acc[r] = metal::fma(xe.w, wt[6], acc[r]);
                                acc[r] = metal::fma(xo.w, wt[7], acc[r]);
                            }
                        }
                    } else {
                        PROJECTION_UNROLL
                        for (uint r = 0u; r < uint(SEISMIC_CONST_VERIFY_ROWS); ++r) {
                            if (r < m) {
                                float4 xe, xo;
                                A::split8(x_packed[(r * chunk_blocks + xb) * 8u + u], xe, xo);
                                const float xv[8] = {xe.x, xo.x, xe.y, xo.y, xe.z, xo.z, xe.w, xo.w};
                                PROJECTION_UNROLL
                                for (uint s = 0u; s < 8u; ++s) {
                                    if (sub + 8u * s + u < chunk)
                                        acc[r] = metal::fma(xv[s], wt[s], acc[r]);
                                }
                            }
                        }
                    }
                }
            }
        }
        if (sg_has_rows) {
            for (uint r = 0u; r < m; ++r) {
                const float sum = ordered_combine8(acc[r]);
                if (has_row && u == 0u)
                    out.store(r, w_row, sum);
            }
        }
    }
}

// Row-specialized reference-order GEMV (`MAXM` in {1, 2, 4, 8}) for mixed
// ordered/unordered segmented launches (`ordered_gemv1..8`): each of the four
// 8-lane quarters of a simdgroup owns a distinct weight row (`W_PER_SG == 4`),
// fetches and decodes its weights once per 64-column block, and reuses them
// across all `m <= MAXM` activation rows, reading lane-strided `uint4` staged
// activations conflict-free.
template <typename W, bool PUBLISH, uint MAXM, uint SIMDGROUPS, typename In, typename Out, typename PubOut>
inline void ordered_gemv_maxm_run(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    PubOut pub_out, ulong pub_stride0, ulong pub_stride1, bool publish_rows, uint m, uint n, uint k, uint local_tile,
    threadgroup uchar *shared, uint sg, uint lane) {
    typedef typename In::activation A;
    typedef OrderedWeight<W> Ow;
    constexpr uint W_PER_SG = ordered_mixed_w_per_sg<MAXM>();
    const uint q = lane >> 3u;
    const uint u = lane & 7u;
    const uint sg_base_row = local_tile * ordered_mixed_rows<MAXM, SIMDGROUPS>() + sg * W_PER_SG;
    const uint w_row = sg_base_row + q;
    const bool sg_has_rows = sg < ordered_mixed_sgs<MAXM, SIMDGROUPS>() && sg_base_row < n;
    const bool has_row = sg_has_rows && w_row < n;
    const uint p_own = u >> 2u;
    const uint s_own = u & 3u;
    typename Ow::fetched cur_w = Ow::empty();
    if (has_row && k > 0u) {
        cur_w = (k >= 64u) ? Ow::fetch_full(w, w_row, p_own, s_own)
                           : ((8u * u < k) ? Ow::fetch(w, w_row, p_own, s_own) : Ow::empty());
    }
    if constexpr (MAXM == 1u) {
        if (m == 1u) {
            float inv0 = 0.0f;
            if (ordered_stage_single_gemv_raw<PUBLISH, SIMDGROUPS>(
                    in, k, shared, pub_out, pub_stride0, pub_stride1, publish_rows, inv0, sg, lane)) {
                if (sg_has_rows) {
                    threadgroup const uint4 *x_packed = reinterpret_cast<threadgroup const uint4 *>(shared);
                    float acc = 0.0f;
                    // The row's whole blocks in batches of WALK, the next
                    // batch's loads in flight while this one is summed, so
                    // the walk waits on memory once a batch, not once a
                    // block; the sum keeps the block order.
                    constexpr uint WALK = ordered_walk_batch<W>();
                    const uint blocks = k >> 6u;
                    // A block's fetch: lane u's own columns where the weights
                    // fetch in stride, else its eight consecutive ones, which
                    // `ordered_transpose8` exchanges.
                    const auto fetch_block = [&](uint block) {
                        if constexpr (Ow::strided)
                            return Ow::fetch_strided(w, w_row, block, u);
                        else
                            return Ow::fetch_full(w, w_row, 2u * block + p_own, s_own);
                    };
                    typename Ow::fetched cur[WALK];
                    cur[0] = Ow::strided ? (has_row && blocks > 0u ? fetch_block(0u) : Ow::empty()) : cur_w;
                    PROJECTION_UNROLL
                    for (uint d = 1u; d < WALK; ++d)
                        cur[d] = (has_row && d < blocks) ? fetch_block(d) : Ow::empty();
                    for (uint b = 0u; b < blocks; b += WALK) {
                        typename Ow::fetched next[WALK];
                        PROJECTION_UNROLL
                        for (uint d = 0u; d < WALK; ++d) {
                            const uint nb = b + WALK + d;
                            next[d] = (has_row && nb < blocks) ? fetch_block(nb) : Ow::empty();
                        }
                        PROJECTION_UNROLL
                        for (uint d = 0u; d < WALK; ++d) {
                            if (b + d < blocks) {
                                float wt[8];
                                if (has_row)
                                    Ow::decode_full(cur[d], s_own, wt);
                                else {
                                    PROJECTION_UNROLL
                                    for (uint j = 0u; j < 8u; ++j)
                                        wt[j] = 0.0f;
                                }
                                if constexpr (!Ow::strided)
                                    ordered_transpose8(wt, u);
                                float4 xe, xo;
                                A::split8(x_packed[(b + d) * 8u + u], xe, xo);
                                acc = metal::fma(xe.x, wt[0], acc);
                                acc = metal::fma(xo.x, wt[1], acc);
                                acc = metal::fma(xe.y, wt[2], acc);
                                acc = metal::fma(xo.y, wt[3], acc);
                                acc = metal::fma(xe.z, wt[4], acc);
                                acc = metal::fma(xo.z, wt[5], acc);
                                acc = metal::fma(xe.w, wt[6], acc);
                                acc = metal::fma(xo.w, wt[7], acc);
                            }
                        }
                        PROJECTION_UNROLL
                        for (uint d = 0u; d < WALK; ++d)
                            cur[d] = next[d];
                    }
                    const uint col0 = blocks << 6u;
                    if (col0 < k)
                        cur_w = (has_row && col0 + 8u * u < k) ? Ow::fetch(w, w_row, (col0 >> 5u) + p_own, s_own)
                                                               : Ow::empty();
                    if (col0 < k) {
                        float wt[8];
                        if (has_row && col0 + 8u * u < k)
                            Ow::decode(cur_w, s_own, col0 + 8u * u, k, wt);
                        else {
                            PROJECTION_UNROLL
                            for (uint j = 0u; j < 8u; ++j)
                                wt[j] = 0.0f;
                        }
                        ordered_transpose8(wt, u);
                        float4 xe, xo;
                        A::split8(x_packed[(col0 >> 6u) * 8u + u], xe, xo);
                        const float xv[8] = {xe.x, xo.x, xe.y, xo.y, xe.z, xo.z, xe.w, xo.w};
                        PROJECTION_UNROLL
                        for (uint s = 0u; s < 8u; ++s) {
                            if (col0 + 8u * s + u < k)
                                acc = metal::fma(xv[s], wt[s], acc);
                        }
                    }
                    const float sum = ordered_combine8(acc);
                    if (has_row && u == 0u)
                        out.store(0u, w_row, sum);
                }
                if (local_tile == 0u) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (sg == 0u && lane == 0u)
                        reinterpret_cast<threadgroup float *>(shared)[0] = inv0;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                return;
            }
        }
    }
    constexpr uint min_shared_bytes = (MAXM == 1u) ? 1024u : ((MAXM == 2u) ? 1536u : ((MAXM == 4u) ? 2560u : ((MAXM == 8u) ? 4608u : 8704u)));
    const uint packets = (k + 31u) >> 5u;
    const uint avail_bytes = metal::max(min_shared_bytes, metal::min(packets * MAXM, 256u) * 64u);
    threadgroup float *smem_inv = reinterpret_cast<threadgroup float *>(shared);
    in.ordered_inverses_sram(m, smem_inv, smem_inv, avail_bytes >> 2u, sg, lane, SIMDGROUPS);
    const float inv_lane = (lane < MAXM && lane < m) ? smem_inv[lane] : 0.0f;
    constexpr uint threads = SIMDGROUPS * 32u;
    const uint tid = sg * 32u + lane;
    if constexpr (PUBLISH) {
        if (publish_rows && k > 0u) {
            if (pub_stride1 == 1ul && (pub_stride0 & 7ul) == 0ul && (k & 7u) == 0u) {
                const uint vecs = k >> 3u;
                for (uint idx = tid; idx < m * vecs; idx += threads) {
                    const uint r = idx / vecs, c = (idx % vecs) * 8u;
                    float4 even, odd;
                    in.load8(r, c, smem_inv[r], even, odd);
                    pub_out.put8(ulong(r) * pub_stride0 + ulong(c), even, odd);
                }
            } else {
                for (uint idx = tid; idx < m * k; idx += threads) {
                    const uint r = idx / k, c = idx % k;
                    pub_out.put(ulong(r) * pub_stride0 + ulong(c) * pub_stride1, in.ordered_at(r, c, smem_inv[r]));
                }
            }
        }
    }
    if (m > 0u && k > 0u) {
        const uint total_blocks = (k + 63u) >> 6u;
        const uint max_blocks = metal::min(total_blocks, avail_bytes / (MAXM * 128u));
        const uint row_cols = max_blocks * 64u;
        threadgroup uint4 *x_packed = reinterpret_cast<threadgroup uint4 *>(shared);
        float acc[MAXM];
        PROJECTION_UNROLL
        for (uint r = 0u; r < MAXM; ++r)
            acc[r] = 0.0f;
        for (uint k0 = 0u; k0 < k; k0 += row_cols) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint chunk = metal::min(row_cols, k - k0);
            const uint chunk_blocks = (chunk + 63u) >> 6u;
            for (uint base = 0u; base < m * chunk_blocks * 8u; base += threads) {
                const uint idx = base + tid;
                const uint blk = idx >> 3u;
                const uint xr = (blk < m * chunk_blocks) ? (blk / chunk_blocks) : 0u;
                const float x_inv = simd_shuffle(inv_lane, ushort(xr & 15u));
                if (blk < m * chunk_blocks) {
                    const uint xb = blk - xr * chunk_blocks;
                    const uint xs = xb * 64u + 8u * u;
                    float4 xe = float4(0.0f), xo = float4(0.0f);
                    if (xs < chunk)
                        in.load8(xr, k0 + xs, x_inv, xe, xo);
                    float xv[8] = {xe.x, xo.x, xe.y, xo.y, xe.z, xo.z, xe.w, xo.w};
                    ordered_transpose8(xv, u);
                    x_packed[(xr * chunk_blocks + xb) * 8u + u] = A::pack8(
                        float4(xv[0], xv[2], xv[4], xv[6]),
                        float4(xv[1], xv[3], xv[5], xv[7]));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg_has_rows) {
                for (uint xb = 0u; xb < chunk_blocks; ++xb) {
                    const uint sub = xb * 64u;
                    const uint col0 = k0 + sub;
                    float wt[8];
                    if (has_row) {
                        if (col0 + 64u <= k)
                            Ow::decode_full(cur_w, s_own, wt);
                        else if (col0 + 8u * u < k)
                            Ow::decode(cur_w, s_own, col0 + 8u * u, k, wt);
                        else {
                            PROJECTION_UNROLL
                            for (uint j = 0u; j < 8u; ++j)
                                wt[j] = 0.0f;
                        }
                        const uint next_col = col0 + 64u;
                        if (next_col + 64u <= k)
                            cur_w = Ow::fetch_full(w, w_row, (next_col >> 5u) + p_own, s_own);
                        else if (next_col + 8u * u < k)
                            cur_w = Ow::fetch(w, w_row, (next_col >> 5u) + p_own, s_own);
                        else
                            cur_w = Ow::empty();
                    } else {
                        PROJECTION_UNROLL
                        for (uint j = 0u; j < 8u; ++j)
                            wt[j] = 0.0f;
                    }
                    ordered_transpose8(wt, u);
                    if (sub + 64u <= chunk) {
                        PROJECTION_UNROLL
                        for (uint r = 0u; r < MAXM; ++r) {
                            if (r < m) {
                                float4 xe, xo;
                                A::split8(x_packed[(r * chunk_blocks + xb) * 8u + u], xe, xo);
                                acc[r] = metal::fma(xe.x, wt[0], acc[r]);
                                acc[r] = metal::fma(xo.x, wt[1], acc[r]);
                                acc[r] = metal::fma(xe.y, wt[2], acc[r]);
                                acc[r] = metal::fma(xo.y, wt[3], acc[r]);
                                acc[r] = metal::fma(xe.z, wt[4], acc[r]);
                                acc[r] = metal::fma(xo.z, wt[5], acc[r]);
                                acc[r] = metal::fma(xe.w, wt[6], acc[r]);
                                acc[r] = metal::fma(xo.w, wt[7], acc[r]);
                            }
                        }
                    } else {
                        PROJECTION_UNROLL
                        for (uint r = 0u; r < MAXM; ++r) {
                            if (r < m) {
                                float4 xe, xo;
                                A::split8(x_packed[(r * chunk_blocks + xb) * 8u + u], xe, xo);
                                const float xv[8] = {xe.x, xo.x, xe.y, xo.y, xe.z, xo.z, xe.w, xo.w};
                                PROJECTION_UNROLL
                                for (uint s = 0u; s < 8u; ++s) {
                                    if (sub + 8u * s + u < chunk)
                                        acc[r] = metal::fma(xv[s], wt[s], acc[r]);
                                }
                            }
                        }
                    }
                }
            }
        }
        if (sg_has_rows) {
            PROJECTION_UNROLL
            for (uint r = 0u; r < MAXM; ++r) {
                if (r < m) {
                    const float sum = ordered_combine8(acc[r]);
                    if (has_row && u == 0u)
                        out.store(r, w_row, sum);
                }
            }
        }
    }
    if (local_tile == 0u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u && lane < MAXM && lane < m)
            reinterpret_cast<threadgroup float *>(shared)[lane] = inv_lane;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <typename W>
struct GemmPrefetch {
    typedef OrderedWeight<W> Ow;
    uint4 x;
    typename Ow::fetched w;
    template <typename In>
    void fetch_x_full(thread const In &in, uint row, uint col, uint m) {
        x = (row < m) ? in.words8_full(row, col) : uint4(0u);
    }
    template <typename In>
    void fetch_x_tail(thread const In &in, uint row, uint col, uint m, uint k) {
        x = (row < m && col < k) ? in.words8(row, col) : uint4(0u);
    }
    uint4 take_x() const { return x; }
    void fetch_w_full(thread const Weights<W> &weights, uint row, uint p, uint step, bool active) {
        w = active ? Ow::fetch_full(weights, row, p, step) : Ow::empty();
    }
    void fetch_w_tail(thread const Weights<W> &weights, uint row, uint p, uint step, bool active, uint base_col,
        uint k) {
        w = (active && base_col < k) ? Ow::fetch(weights, row, p, step) : Ow::empty();
    }
    void decode_w_full(uint step, bool active, thread float (&wt)[8]) const {
        if (active)
            Ow::decode_full(w, step, wt);
        else {
            PROJECTION_UNROLL
            for (uint j = 0; j < 8; ++j)
                wt[j] = 0.0f;
        }
    }
    void decode_w_tail(uint step, uint base_col, uint k, bool active, thread float (&wt)[8]) const {
        if (active && base_col < k)
            Ow::decode(w, step, base_col, k, wt);
        else {
            PROJECTION_UNROLL
            for (uint j = 0; j < 8; ++j)
                wt[j] = 0.0f;
        }
    }
};

template <typename E>
struct GemmPrefetch<packets::Dense<E>> {
    typedef OrderedWeight<packets::Dense<E>> Ow;
    typename Ow::fetched buf;
    template <typename In>
    void fetch_x_full(thread const In &in, uint row, uint col, uint m) {
        buf.raw[0] = (row < m) ? in.words8_full(row, col) : uint4(0u);
    }
    template <typename In>
    void fetch_x_tail(thread const In &in, uint row, uint col, uint m, uint k) {
        buf.raw[0] = (row < m && col < k) ? in.words8(row, col) : uint4(0u);
    }
    uint4 take_x() const { return buf.raw[0]; }
    void fetch_w_full(thread const Weights<packets::Dense<E>> &weights, uint row, uint p, uint step, bool active) {
        buf = active ? Ow::fetch_full(weights, row, p, step) : Ow::empty();
    }
    void fetch_w_tail(thread const Weights<packets::Dense<E>> &weights, uint row, uint p, uint step, bool active,
        uint base_col, uint k) {
        buf = (active && base_col < k) ? Ow::fetch(weights, row, p, step) : Ow::empty();
    }
    void decode_w_full(uint step, bool active, thread float (&wt)[8]) const {
        if (active)
            Ow::decode_full(buf, step, wt);
        else {
            PROJECTION_UNROLL
            for (uint j = 0; j < 8; ++j)
                wt[j] = 0.0f;
        }
    }
    void decode_w_tail(uint step, uint base_col, uint k, bool active, thread float (&wt)[8]) const {
        if (active && base_col < k)
            Ow::decode(buf, step, base_col, k, wt);
        else {
            PROJECTION_UNROLL
            for (uint j = 0; j < 8; ++j)
                wt[j] = 0.0f;
        }
    }
};

template <typename W, typename In, typename Out>
inline void ordered_gemm_col1_run(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    uint m, uint n, uint k, uint tile_m, uint tile_n, threadgroup uchar *shared, uint sg, uint lane) {
    typedef typename In::activation A;
    if (k == 0u || (tile_n * 32u) >= n)
        return;
    const uint full_packets = k >> 5u;
    const bool has_tail = (k & 31u) != 0u;
    constexpr uint BUF_FLOATS = 3072u; // 64*32 X (2048) + 32*32 W (1024)
    threadgroup float *base = reinterpret_cast<threadgroup float *>(shared);
    const uint tid = sg * 32u + lane;
    const uint u = lane & 7u;
    const uint gn = lane >> 3u;
    const uint act_lo = tile_m * 64u + sg * 8u;
    const uint dec_row = tile_n * 32u;
    const uint x_row = tile_m * 64u + (tid >> 2u);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint4 x_pref = uint4(0u);
    float w_pref = 0.0f;
    if (full_packets > 0u) {
        x_pref = (x_row < m) ? in.words8_full(x_row, 8u * (tid & 3u)) : uint4(0u);
        if (sg == 0u)
            w_pref = ordered_weight_at(w, dec_row, lane);
    }
    for (uint p = 0; p < full_packets; ++p) {
        const uint k0 = 32u * p;
        threadgroup float *x_tile = base + (p & 1u) * BUF_FLOATS;
        threadgroup float *w_tile = x_tile + 2048u;
        {
            float4 xe, xo;
            A::split8(x_pref, xe, xo);
            threadgroup float4 *xdst4 = reinterpret_cast<threadgroup float4 *>(x_tile) + 2u * tid;
            xdst4[0] = float4(xe.x, xo.x, xe.y, xo.y);
            xdst4[1] = float4(xe.z, xo.z, xe.w, xo.w);
            const uint next_col = k0 + 32u + 8u * (tid & 3u);
            if (p + 1u < full_packets)
                x_pref = (x_row < m) ? in.words8_full(x_row, next_col) : uint4(0u);
            else if (has_tail)
                x_pref = (x_row < m && next_col < k) ? in.words8(x_row, next_col) : uint4(0u);
        }
        if (sg == 0u) {
            w_tile[lane] = w_pref;
            const uint next_col = k0 + 32u + lane;
            if (p + 1u < full_packets)
                w_pref = ordered_weight_at(w, dec_row, next_col);
            else if (has_tail)
                w_pref = (next_col < k) ? ordered_weight_at(w, dec_row, next_col) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (act_lo < m && gn == 0u) {
            const threadgroup float *my_x = x_tile + sg * 256u + u;
            PROJECTION_UNROLL
            for (uint s = 0; s < 4; ++s) {
                const float wv0 = w_tile[8u * s + u];
                PROJECTION_UNROLL
                for (uint i = 0; i < 8; ++i)
                    acc[i] = metal::fma(my_x[i * 32u + 8u * s], wv0, acc[i]);
            }
        }
    }
    if (has_tail) {
        const uint k0 = 32u * full_packets;
        threadgroup float *x_tile = base + (full_packets & 1u) * BUF_FLOATS;
        threadgroup float *w_tile = x_tile + 2048u;
        if (full_packets == 0u) {
            const uint col = 8u * (tid & 3u);
            x_pref = (x_row < m && col < k) ? in.words8(x_row, col) : uint4(0u);
            if (sg == 0u)
                w_pref = (lane < k) ? ordered_weight_at(w, dec_row, lane) : 0.0f;
        }
        {
            float4 xe, xo;
            A::split8(x_pref, xe, xo);
            threadgroup float4 *xdst4 = reinterpret_cast<threadgroup float4 *>(x_tile) + 2u * tid;
            xdst4[0] = float4(xe.x, xo.x, xe.y, xo.y);
            xdst4[1] = float4(xe.z, xo.z, xe.w, xo.w);
        }
        if (sg == 0u)
            w_tile[lane] = w_pref;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (act_lo < m && gn == 0u) {
            const threadgroup float *my_x = x_tile + sg * 256u + u;
            PROJECTION_UNROLL
            for (uint s = 0; s < 4; ++s) {
                if (k0 + 8u * s + u < k) {
                    const float wv0 = w_tile[8u * s + u];
                    PROJECTION_UNROLL
                    for (uint i = 0; i < 8; ++i)
                        acc[i] = metal::fma(my_x[i * 32u + 8u * s], wv0, acc[i]);
                }
            }
        }
    }
    if (act_lo < m) {
        PROJECTION_UNROLL
        for (uint i = 0; i < 8; ++i) {
            const uint out_r = act_lo + i;
            if (out_r < m) {
                const float sum = ordered_combine8(acc[i]);
                if (gn == 0u && u == 0u)
                    out.store(out_r, dec_row, sum);
            }
        }
    }
}

template <typename W, typename In, typename Out>
inline void ordered_gemm_run(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    uint m, uint n, uint k, uint tile_m, uint tile_n, threadgroup uchar *shared, uint sg, uint lane) {
    typedef typename In::activation A;
    if (k == 0u || (tile_n * 32u) >= n)
        return;
    const uint full_packets = k >> 5u;
    const bool has_tail = (k & 31u) != 0u;
    constexpr uint BUF_FLOATS = 3072u; // 64*32 X (2048) + 32*32 W (1024)
    threadgroup float *base = reinterpret_cast<threadgroup float *>(shared);
    const uint tid = sg * 32u + lane;
    const uint u = lane & 7u;
    const uint gn = lane >> 3u;
    const uint act_lo = tile_m * 64u + sg * 8u;
    const uint x_row = tile_m * 64u + (tid >> 2u);
    const uint c_dec = tid >> 2u, s_dec = tid & 3u;
    const uint dec_row = tile_n * 32u + c_dec;
    const uint j_dec = c_dec & 7u, gn_dec = c_dec >> 3u;
    const uint w_slot = (s_dec * 8u + j_dec) * 4u + (gn_dec ^ (j_dec & 3u));
    float acc[8][8];
    PROJECTION_UNROLL
    for (uint i = 0; i < 8; ++i)
        PROJECTION_UNROLL
        for (uint j = 0; j < 8; ++j)
            acc[i][j] = 0.0f;
    uint4 x_pref = uint4(0u);
    GemmPrefetch<W> w_pref;
    if (full_packets > 0u) {
        x_pref = (x_row < m) ? in.words8_full(x_row, 8u * (tid & 3u)) : uint4(0u);
        if (sg < 4u)
            w_pref.fetch_w_full(w, dec_row, 0u, s_dec, dec_row < n);
    }
    for (uint p = 0; p < full_packets; ++p) {
        const uint k0 = 32u * p;
        threadgroup float *x_tile = base + (p & 1u) * BUF_FLOATS;
        threadgroup float *w_tile = x_tile + 2048u;
        {
            float4 xe, xo;
            A::split8(x_pref, xe, xo);
            threadgroup float4 *xdst4 = reinterpret_cast<threadgroup float4 *>(x_tile) + 2u * tid;
            xdst4[0] = float4(xe.x, xo.x, xe.y, xo.y);
            xdst4[1] = float4(xe.z, xo.z, xe.w, xo.w);
            const uint next_col = k0 + 32u + 8u * (tid & 3u);
            if (p + 1u < full_packets)
                x_pref = (x_row < m) ? in.words8_full(x_row, next_col) : uint4(0u);
            else if (has_tail)
                x_pref = (x_row < m && next_col < k) ? in.words8(x_row, next_col) : uint4(0u);
        }
        if (sg < 4u) {
            float w_vals[8];
            w_pref.decode_w_full(s_dec, dec_row < n, w_vals);
            threadgroup float4 *wdst4 = reinterpret_cast<threadgroup float4 *>(w_tile) + 2u * w_slot;
            wdst4[0] = float4(w_vals[0], w_vals[1], w_vals[2], w_vals[3]);
            wdst4[1] = float4(w_vals[4], w_vals[5], w_vals[6], w_vals[7]);
            if (p + 1u < full_packets)
                w_pref.fetch_w_full(w, dec_row, p + 1u, s_dec, dec_row < n);
            else if (has_tail)
                w_pref.fetch_w_tail(w, dec_row, p + 1u, s_dec, dec_row < n, k0 + 32u + 8u * s_dec, k);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (act_lo < m && (tile_n * 32u + 8u * gn) < n) {
            const threadgroup float *my_x = x_tile + sg * 256u + u;
            const threadgroup float *w_cur0 = w_tile + lane;
            const threadgroup float *w_cur1 = w_tile + (lane ^ 8u);
            const threadgroup float *w_cur2 = w_tile + (lane ^ 16u);
            const threadgroup float *w_cur3 = w_tile + (lane ^ 24u);
            PROJECTION_UNROLL
            for (uint s = 0; s < 4; ++s) {
                const float xv0 = my_x[0u * 32u + 8u * s];
                const float xv1 = my_x[1u * 32u + 8u * s];
                const float xv2 = my_x[2u * 32u + 8u * s];
                const float xv3 = my_x[3u * 32u + 8u * s];
                const float xv4 = my_x[4u * 32u + 8u * s];
                const float xv5 = my_x[5u * 32u + 8u * s];
                const float xv6 = my_x[6u * 32u + 8u * s];
                const float xv7 = my_x[7u * 32u + 8u * s];
                PROJECTION_UNROLL
                for (uint j = 0; j < 8; ++j) {
                    const threadgroup float *w_ptr = ((j & 3u) == 0u) ? w_cur0
                        : (((j & 3u) == 1u) ? w_cur1 : (((j & 3u) == 2u) ? w_cur2 : w_cur3));
                    const float wv = w_ptr[(s * 8u + j) * 32u];
                    acc[0][j] = metal::fma(xv0, wv, acc[0][j]);
                    acc[1][j] = metal::fma(xv1, wv, acc[1][j]);
                    acc[2][j] = metal::fma(xv2, wv, acc[2][j]);
                    acc[3][j] = metal::fma(xv3, wv, acc[3][j]);
                    acc[4][j] = metal::fma(xv4, wv, acc[4][j]);
                    acc[5][j] = metal::fma(xv5, wv, acc[5][j]);
                    acc[6][j] = metal::fma(xv6, wv, acc[6][j]);
                    acc[7][j] = metal::fma(xv7, wv, acc[7][j]);
                }
            }
        }
    }
    if (has_tail) {
        const uint k0 = 32u * full_packets;
        threadgroup float *x_tile = base + (full_packets & 1u) * BUF_FLOATS;
        threadgroup float *w_tile = x_tile + 2048u;
        if (full_packets == 0u) {
            const uint col = 8u * (tid & 3u);
            x_pref = (x_row < m && col < k) ? in.words8(x_row, col) : uint4(0u);
            if (sg < 4u)
                w_pref.fetch_w_tail(w, dec_row, 0u, s_dec, dec_row < n, 8u * s_dec, k);
        }
        {
            float4 xe, xo;
            A::split8(x_pref, xe, xo);
            threadgroup float4 *xdst4 = reinterpret_cast<threadgroup float4 *>(x_tile) + 2u * tid;
            xdst4[0] = float4(xe.x, xo.x, xe.y, xo.y);
            xdst4[1] = float4(xe.z, xo.z, xe.w, xo.w);
        }
        if (sg < 4u) {
            float w_vals[8];
            w_pref.decode_w_tail(s_dec, k0 + 8u * s_dec, k, dec_row < n, w_vals);
            threadgroup float4 *wdst4 = reinterpret_cast<threadgroup float4 *>(w_tile) + 2u * w_slot;
            wdst4[0] = float4(w_vals[0], w_vals[1], w_vals[2], w_vals[3]);
            wdst4[1] = float4(w_vals[4], w_vals[5], w_vals[6], w_vals[7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (act_lo < m && (tile_n * 32u + 8u * gn) < n) {
            const threadgroup float *my_x = x_tile + sg * 256u + u;
            const threadgroup float *w_cur0 = w_tile + lane;
            const threadgroup float *w_cur1 = w_tile + (lane ^ 8u);
            const threadgroup float *w_cur2 = w_tile + (lane ^ 16u);
            const threadgroup float *w_cur3 = w_tile + (lane ^ 24u);
            PROJECTION_UNROLL
            for (uint s = 0; s < 4; ++s) {
                if (k0 + 8u * s + u < k) {
                    const float xv0 = my_x[0u * 32u + 8u * s];
                    const float xv1 = my_x[1u * 32u + 8u * s];
                    const float xv2 = my_x[2u * 32u + 8u * s];
                    const float xv3 = my_x[3u * 32u + 8u * s];
                    const float xv4 = my_x[4u * 32u + 8u * s];
                    const float xv5 = my_x[5u * 32u + 8u * s];
                    const float xv6 = my_x[6u * 32u + 8u * s];
                    const float xv7 = my_x[7u * 32u + 8u * s];
                    PROJECTION_UNROLL
                    for (uint j = 0; j < 8; ++j) {
                        const threadgroup float *w_ptr = ((j & 3u) == 0u) ? w_cur0
                            : (((j & 3u) == 1u) ? w_cur1 : (((j & 3u) == 2u) ? w_cur2 : w_cur3));
                        const float wv = w_ptr[(s * 8u + j) * 32u];
                        acc[0][j] = metal::fma(xv0, wv, acc[0][j]);
                        acc[1][j] = metal::fma(xv1, wv, acc[1][j]);
                        acc[2][j] = metal::fma(xv2, wv, acc[2][j]);
                        acc[3][j] = metal::fma(xv3, wv, acc[3][j]);
                        acc[4][j] = metal::fma(xv4, wv, acc[4][j]);
                        acc[5][j] = metal::fma(xv5, wv, acc[5][j]);
                        acc[6][j] = metal::fma(xv6, wv, acc[6][j]);
                        acc[7][j] = metal::fma(xv7, wv, acc[7][j]);
                    }
                }
            }
        }
    }
    if (act_lo < m) {
        PROJECTION_UNROLL
        for (uint i = 0; i < 8; ++i) {
            const uint out_r = act_lo + i;
            if (out_r < m) {
                PROJECTION_UNROLL
                for (uint j = 0; j < 8; ++j) {
                    const float sum = ordered_combine8(acc[i][j]);
                    const uint out_c = tile_n * 32u + 8u * gn + j;
                    if (u == 0u && out_c < n)
                        out.store(out_r, out_c, sum);
                }
            }
        }
    }
}

} // namespace projection

#ifndef PROJECTION_ORDERED_PUBLISH
#define PROJECTION_ORDERED_PUBLISH 0
#endif
#ifndef PROJECTION_ORDERED_PUB_OUT
#define PROJECTION_ORDERED_PUB_OUT element::Out<element::F16>(nullptr)
#define PROJECTION_ORDERED_PUB_S0 0ul
#define PROJECTION_ORDERED_PUB_S1 1ul
#endif
#ifndef PROJECTION_ORDERED_FIRST_TILE
#define PROJECTION_ORDERED_FIRST_TILE() ((void)0)
#endif
#ifndef PROJECTION_ORDERED_EPILOGUE
#define PROJECTION_ORDERED_EPILOGUE(ordered_tiles) ((void)0)
#endif
#ifndef PROJECTION_STAGE_EXTRA
#define PROJECTION_STAGE_EXTRA(row, shared_inv) ((void)0)
#endif

#define PROJECTION_ORDERED_STAGE_ARGS                                                                           \
    uint row [[threadgroup_position_in_grid]],                                                                  \
    uint sg [[simdgroup_index_in_threadgroup]],                                                                 \
    uint lane [[thread_index_in_simdgroup]]

#define PROJECTION_ORDERED_STAGE_BODY(IN_EXPR, OUT_EXPR, OUT_S0, OUT_S1)                                        \
    threadgroup float stage_scratch[8];                                                                         \
    const auto stage_in = (IN_EXPR);                                                                            \
    const float stage_inv = projection::ordered_stage_row(stage_in, (OUT_EXPR), (OUT_S0), (OUT_S1),             \
        row, uint(m), uint(PROJECTION_STAGE_K), stage_scratch, sg, lane);                                       \
    if (row < uint(m)) { PROJECTION_STAGE_EXTRA(row, stage_inv); }

#define PROJECTION_ORDERED_GEMV_ARGS                                                                            \
    threadgroup uchar *shared [[threadgroup(0)]],                                                               \
    uint3 position [[threadgroup_position_in_grid]],                                                            \
    uint sg [[simdgroup_index_in_threadgroup]],                                                                 \
    uint lane [[thread_index_in_simdgroup]]

#define PROJECTION_ORDERED_GEMV_SINGLE                                                                          \
    PROJECTION_OPERANDS(position.z * uint(PROJECTION_GROUPS > 1));                                              \
    PROJECTION_GEMV_INPUT(input);                                                                               \
    constexpr uint ord_per = simdgroups * 4u;                                                                        \
    const uint ordered_tiles = (uint(n) + ord_per - 1u) / ord_per;                                              \
    projection::ordered_gemv_run<packets::W0, (PROJECTION_ORDERED_PUBLISH != 0), simdgroups>(input, out, w,     \
        PROJECTION_ORDERED_PUB_OUT, PROJECTION_ORDERED_PUB_S0, PROJECTION_ORDERED_PUB_S1, position.x == 0u,     \
        m, n, k, position.x, shared, sg, lane);                                                                 \
    if (position.x == 0u) { PROJECTION_ORDERED_FIRST_TILE(); }                                                  \
    PROJECTION_ORDERED_EPILOGUE(ordered_tiles)

#define PROJECTION_ORDERED_GEMM_ARGS                                                                            \
    threadgroup uchar *shared [[threadgroup(0)]],                                                               \
    uint3 tile [[threadgroup_position_in_grid]],                                                                \
    uint sg [[simdgroup_index_in_threadgroup]],                                                                 \
    uint lane [[thread_index_in_simdgroup]]

#define PROJECTION_ORDERED_GEMM_SINGLE                                                                          \
    PROJECTION_OPERANDS(tile.z * uint(PROJECTION_GROUPS > 1));                                                  \
    projection::ordered_gemm_run<packets::W0>(in, out, w, m, n, k, tile.y, tile.x, shared, sg, lane)

#if SEISMIC_PROJECTION_SEGMENTS
#ifndef PROJECTION_SEGMENT_INPUT
#define PROJECTION_SEGMENT_INPUT(s, name) PROJECTION_GEMV_INPUT(name)
#endif
#ifndef PROJECTION_SEGMENT_NVFP4
#define PROJECTION_SEGMENT_NVFP4(s) NVFP4
#endif

#define PROJECTION_ORDERED_ADD_TILES(s, ORD_PER)                                                                \
    + (SEISMIC_SEGMENT_ORDERED_##s != 0 ? (uint(PROJECTION_N##s) + (ORD_PER) - 1u) / (ORD_PER) : 0u)

#define PROJECTION_ORDERED_SEG_FOUND(s, ORD_PER, UNORD_PER)                                                     \
    if (segment == s##u) {                                                                                      \
        const uint per_s = SEISMIC_SEGMENT_ORDERED_##s != 0 ? (ORD_PER) : (UNORD_PER);                          \
        const uint tiles_s = (uint(PROJECTION_N##s) + per_s - 1u) / per_s;                                      \
        if (local >= tiles_s) {                                                                                 \
            local -= tiles_s;                                                                                   \
            segment = s##u + 1u;                                                                                \
        }                                                                                                       \
    }

#define PROJECTION_ORDERED_SEG_GEMV_ONLY(s, ORD_PER)                                                            \
    else if (segment == s##u) {                                                                                 \
        if (PROJECTION_ZEROED(s)) {                                                                             \
            projection::zero_segment<PROJECTION_PAIRED(s)>(out##s, m, uint(PROJECTION_N##s), 0u, m,             \
                local * (ORD_PER), ORD_PER, sg * 32u + lane, simdgroups * 32u);                                 \
        } else {                                                                                                \
            PROJECTION_SEGMENT_INPUT(s, input_##s);                                                             \
            projection::ordered_gemv_run<PROJECTION_PACKET(s), (PROJECTION_ORDERED_PUBLISH != 0),               \
                simdgroups>(input_##s, out##s, w##s, PROJECTION_ORDERED_PUB_OUT, PROJECTION_ORDERED_PUB_S0,     \
                PROJECTION_ORDERED_PUB_S1, position.x == 0u, m, uint(PROJECTION_N##s), k, local, shared,        \
                sg, lane);                                                                                      \
        }                                                                                                       \
    }

#define PROJECTION_ORDERED_SEG_GEMV_MIXED(s, MAXM, ORD_PER, UNORD_PER)                                          \
    else if (segment == s##u) {                                                                                 \
        if constexpr (SEISMIC_SEGMENT_ORDERED_##s != 0) {                                                       \
            if (PROJECTION_ZEROED(s)) {                                                                         \
                projection::zero_segment<PROJECTION_PAIRED(s)>(out##s, m, uint(PROJECTION_N##s), 0u, m,         \
                    local * (ORD_PER), ORD_PER, sg * 32u + lane, simdgroups * 32u);                             \
            } else {                                                                                            \
                PROJECTION_SEGMENT_INPUT(s, input_##s);                                                         \
                projection::ordered_gemv_maxm_run<PROJECTION_PACKET(s), (PROJECTION_ORDERED_PUBLISH != 0),      \
                    (MAXM), simdgroups>(input_##s, out##s, w##s, PROJECTION_ORDERED_PUB_OUT,                    \
                    PROJECTION_ORDERED_PUB_S0, PROJECTION_ORDERED_PUB_S1, position.x == 0u, m,                  \
                    uint(PROJECTION_N##s), k, local, shared, sg, lane);                                         \
            }                                                                                                   \
        } else {                                                                                                \
            if (PROJECTION_ZEROED(s)) {                                                                         \
                projection::zero_segment<PROJECTION_PAIRED(s)>(out##s, m, uint(PROJECTION_N##s), 0u, m,         \
                    local * (UNORD_PER), UNORD_PER, sg * 32u + lane, simdgroups * 32u);                         \
            } else {                                                                                            \
                PROJECTION_SEGMENT_INPUT(s, input_##s);                                                         \
                projection::unordered_from_ordered_maxm_run<PROJECTION_PACKET(s), PROJECTION_PARTNER_PACKET(s), \
                    PROJECTION_PAIRED(s), ROWS, (MAXM), LANES, PROJECTION_SEGMENT_NVFP4(s), simdgroups>(        \
                    input_##s, out##s, w##s, PROJECTION_PARTNER(s), m, uint(PROJECTION_N##s), k, local, shared, \
                    sg, lane);                                                                                  \
            }                                                                                                   \
            return;                                                                                             \
        }                                                                                                       \
    }

#define PROJECTION_SEGMENTED_ORDERED_GEMV_ALL                                                                   \
    constexpr uint ord_per = simdgroups * 4u;                                                                        \
    PROJECTION_OPERANDS;                                                                                        \
    const uint ordered_tiles = 0u PROJECTION_EACH_SEGMENT(PROJECTION_ORDERED_ADD_TILES, ord_per);               \
    uint segment = 0u, local = position.x;                                                                      \
    PROJECTION_EACH_SEGMENT(PROJECTION_ORDERED_SEG_FOUND, ord_per, ord_per)                                     \
    PROJECTION_SEGMENT_CHAIN                                                                                    \
    PROJECTION_EACH_SEGMENT(PROJECTION_ORDERED_SEG_GEMV_ONLY, ord_per)                                          \
    else { return; }                                                                                            \
    if (position.x == 0u) { PROJECTION_ORDERED_FIRST_TILE(); }                                                  \
    PROJECTION_ORDERED_EPILOGUE(ordered_tiles)

#define PROJECTION_SEGMENTED_ORDERED_GEMV_MIXED(MAXM)                                                           \
    constexpr uint ord_per = projection::ordered_mixed_rows<(MAXM), simdgroups>();                                                                   \
    constexpr uint unord_per = simdgroups * ROWS * (32u / LANES);                                               \
    PROJECTION_OPERANDS;                                                                                        \
    const uint ordered_tiles = 0u PROJECTION_EACH_SEGMENT(PROJECTION_ORDERED_ADD_TILES, ord_per);               \
    uint segment = 0u, local = position.x;                                                                      \
    PROJECTION_EACH_SEGMENT(PROJECTION_ORDERED_SEG_FOUND, ord_per, unord_per)                                   \
    PROJECTION_SEGMENT_CHAIN                                                                                    \
    PROJECTION_EACH_SEGMENT(PROJECTION_ORDERED_SEG_GEMV_MIXED, (MAXM), ord_per, unord_per)                      \
    else { return; }                                                                                            \
    if (position.x == 0u) { PROJECTION_ORDERED_FIRST_TILE(); }                                                  \
    PROJECTION_ORDERED_EPILOGUE(ordered_tiles)

#define PROJECTION_ORDERED_SEG_GEMM(s, ...)                                                                     \
    else if (segment == s##u) {                                                                                 \
        if (PROJECTION_ZEROED(s))                                                                               \
            projection::zero_segment<PROJECTION_PAIRED(s)>(out##s, m, uint(PROJECTION_N##s), tile.y * 32u, 32u, \
                local * 32u, 32u, sg * 32u + lane, 256u);                                                       \
        else if constexpr (PROJECTION_N##s <= 1)                                                                \
            projection::ordered_gemm_col1_run<PROJECTION_PACKET(s)>(in, out##s, w##s, m, uint(PROJECTION_N##s), \
                k, tile.y, local, shared, sg, lane);                                                            \
        else                                                                                                    \
            projection::ordered_gemm_run<PROJECTION_PACKET(s)>(in, out##s, w##s, m, uint(PROJECTION_N##s),      \
                k, tile.y, local, shared, sg, lane);                                                            \
    }

#define PROJECTION_SEGMENTED_ORDERED_GEMM                                                                       \
    PROJECTION_OPERANDS;                                                                                        \
    PROJECTION_SEGMENT_FIND(tile.x, 32u);                                                                       \
    PROJECTION_SEGMENT_CHAIN                                                                                    \
    PROJECTION_EACH_SEGMENT(PROJECTION_ORDERED_SEG_GEMM, )
#endif
